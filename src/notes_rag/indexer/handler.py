"""The indexer Lambda: rebuild the index when the source bucket changes.

Ordering matters. The manifest diff runs first and, when nothing changed,
returns before downloading the index or constructing a Bedrock client - roughly
8,639 of ~8,640 monthly runs take that path and must stay cheap.

When something did change, every source is re-chunked, not just the changed
ones: `build_index` deletes any source path absent from the chunks it is given,
so a partial rebuild would delete the rest of the corpus. Chunking is local and
free; the expensive part is embedding, and that stays incremental because
`build_index` reuses any vector whose content_hash is already stored.
"""

import logging
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from notes_rag.embed.base import Embedder
from notes_rag.indexer.build import build_index, derive_backlinks
from notes_rag.indexer.collect import (
    SourceDocument,
    build_chunks,
    classify,
    unsupported_suffix_skip,
)
from notes_rag.indexer.manifest import Manifest
from notes_rag.sources.s3 import (
    download_file,
    get_bytes,
    get_json,
    head_exists,
    list_objects,
    put_json,
    upload_file,
)
from notes_rag.store.sqlite_vec import SqliteVecStore

logger = logging.getLogger(__name__)

DEFAULT_PREFIXES = ("summaries/", "transcripts/")


@dataclass(frozen=True)
class IndexerConfig:
    source_bucket: str
    index_bucket: str
    source_prefixes: tuple[str, ...] = DEFAULT_PREFIXES
    full_db_key: str = "index/full.db"
    public_db_key: str = "index/public.db"
    manifest_key: str = "index/manifest.json"
    dimensions: int = 1024
    bedrock_region: str = "us-east-2"
    vault_id: str = "Vault"
    work_dir: str = "/tmp"

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "IndexerConfig":
        """Build config from environment variables.

        SOURCE_BUCKET and INDEX_BUCKET are required; a missing one raises
        KeyError at cold start, which is the right time to find out.
        """
        prefixes = env.get("SOURCE_PREFIXES")
        return cls(
            source_bucket=env["SOURCE_BUCKET"],
            index_bucket=env["INDEX_BUCKET"],
            source_prefixes=(
                tuple(p.strip() for p in prefixes.split(",") if p.strip())
                if prefixes
                else DEFAULT_PREFIXES
            ),
            full_db_key=env.get("FULL_DB_KEY", "index/full.db"),
            public_db_key=env.get("PUBLIC_DB_KEY", "index/public.db"),
            manifest_key=env.get("MANIFEST_KEY", "index/manifest.json"),
            dimensions=int(env.get("EMBED_DIMENSIONS", "1024")),
            bedrock_region=env.get("BEDROCK_REGION", "us-east-2"),
            vault_id=env.get("VAULT_ID", "Vault"),
            work_dir=env.get("WORK_DIR", "/tmp"),
        )


@dataclass(frozen=True)
class IndexerResult:
    status: str
    changed: int = 0
    removed: int = 0
    chunks_written: int = 0
    vectors_embedded: int = 0
    vectors_reused: int = 0

    @classmethod
    def no_op(cls) -> "IndexerResult":
        return cls(status="no-op")

    def to_dict(self) -> dict:
        return asdict(self)


def run_index(config: IndexerConfig, *, s3, embedder: Embedder) -> IndexerResult:
    """List, diff, and rebuild if anything moved. Returns what happened."""
    objects = list_objects(s3, config.source_bucket, config.source_prefixes)
    previous = Manifest.from_dict(get_json(s3, config.index_bucket, config.manifest_key))
    diff = previous.diff(objects)

    if diff.is_empty:
        # An empty diff alone isn't enough: if an operator deletes full.db to
        # force a re-embed, or rolls back to an S3 object version that predates
        # one of the artifacts (a path infra/storage.tf explicitly advertises),
        # the manifest still matches and every run would return no-op forever,
        # never restoring what's missing. head_object is ~20ms, so checking
        # both keys keeps the common no-op path cheap. Only bother when the
        # manifest actually recorded a prior build - a genuine first run with
        # zero source objects has no artifacts either, and that is correctly
        # a no-op, not something to "restore".
        missing = (
            [
                name
                for name, key in (
                    ("full.db", config.full_db_key),
                    ("public.db", config.public_db_key),
                )
                if not head_exists(s3, config.index_bucket, key)
            ]
            if previous.etags
            else []
        )
        if not missing:
            logger.info("no source changes; skipping rebuild")
            return IndexerResult.no_op()
        logger.warning(
            "source unchanged but %s missing from the index bucket; rebuilding",
            " and ".join(missing),
        )

    logger.info("rebuilding: %d changed, %d removed", len(diff.changed), len(diff.removed))

    work = Path(config.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    full_db = work / "full.db"
    public_db = work / "public.db"
    # A container Lambda can reuse /tmp across invocations. An invocation
    # killed mid-write can leave a SQLite side file (-journal, -wal, -shm)
    # behind for whatever full.db/public.db were on disk at the time; clearing
    # only the two db files would let the next run's freshly downloaded
    # full.db open next to a stale hot journal for a different database.
    for stale in (*work.glob("full.db*"), *work.glob("public.db*")):
        stale.unlink(missing_ok=True)

    # The previous index is the embedding cache. Its absence is fine on a
    # genuine first run - build_index simply embeds everything - but if the
    # manifest says sources were already indexed, a missing full.db means this
    # rebuild silently re-embeds the whole corpus at full Bedrock cost, with no
    # trace but vectors_reused: 0.
    had_previous_index = download_file(s3, config.index_bucket, config.full_db_key, full_db)
    if not had_previous_index and previous.etags:
        logger.warning(
            "no previous index found at %s; the full corpus will be re-embedded",
            config.full_db_key,
        )

    # Decide from the key alone, before fetching anything: a large object under
    # a watched prefix with an unsupported suffix must never be pulled into
    # memory just to discover that. That is what let one such object wedge the
    # index permanently - MemoryError before put_json, so the manifest never
    # advances and the next tick lists and dies on the same object forever.
    suffix_skips = []
    documents = []
    for obj in objects:
        skip = unsupported_suffix_skip(obj.key)
        if skip is not None:
            suffix_skips.append(skip)
            continue
        documents.append(
            SourceDocument(source_path=obj.key, raw=get_bytes(s3, config.source_bucket, obj.key))
        )

    collected = classify(documents)
    chunks, unpairable = build_chunks(collected, vault_id=config.vault_id)
    for path, reason in [*suffix_skips, *collected.skipped, *unpairable]:
        logger.warning("skipping %s: %s", path, reason)

    chunks = derive_backlinks(chunks)

    store = SqliteVecStore(full_db, dimensions=config.dimensions)
    try:
        stats = build_index(chunks, store, embedder)
        store.copy_filtered(public_db, corpus="video")
    finally:
        store.close()

    upload_file(s3, config.index_bucket, config.full_db_key, full_db)
    upload_file(s3, config.index_bucket, config.public_db_key, public_db)
    put_json(s3, config.index_bucket, config.manifest_key, Manifest.of(objects).to_dict())

    return IndexerResult(
        status="rebuilt",
        changed=len(diff.changed),
        removed=len(diff.removed),
        chunks_written=stats.chunks_written,
        vectors_embedded=stats.vectors_embedded,
        vectors_reused=stats.vectors_reused,
    )


def lambda_handler(event, context) -> dict:
    """Entry point. `event` is ignored: the scheduler sends nothing useful, and
    an on-demand `aws lambda invoke` should behave identically."""
    logging.getLogger().setLevel(logging.INFO)

    # Imported here rather than at module scope so that importing this module -
    # which the unit tests do - never constructs an AWS client.
    import boto3

    from notes_rag.embed.bedrock import TitanEmbedder

    config = IndexerConfig.from_env(os.environ)
    result = run_index(
        config,
        s3=boto3.client("s3"),
        embedder=TitanEmbedder(region=config.bedrock_region, dimensions=config.dimensions),
    )
    logger.info("indexer result: %s", result.to_dict())
    return result.to_dict()
