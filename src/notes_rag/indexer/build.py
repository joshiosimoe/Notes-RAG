"""Index assembly with an embedding cache.

Spec §3: a full re-embed on every rebuild costs ~$72/month at a 5-minute
trigger cadence, so incremental embedding is mandatory rather than an
optimisation. The cache is the store itself — a chunk whose content_hash is
already present reuses its vector instead of calling Bedrock.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

from notes_rag.embed.base import Embedder
from notes_rag.models import Chunk
from notes_rag.store.base import VectorStore


@dataclass(frozen=True)
class BuildStats:
    chunks_written: int
    vectors_embedded: int
    vectors_reused: int
    paths_deleted: int


def derive_backlinks(chunks: Sequence[Chunk]) -> list[Chunk]:
    """Populate `backlinks` by inverting `links_to` across the whole chunk set.

    Wikilinks name a note, not a path, so targets are matched against each
    source path's stem. That stem is only unique *within* a vault - two
    vaults can both contain a "README.md" - so the lookup and accumulator are
    keyed on (vault_id, stem), not stem alone. Keying on stem alone would let
    a link in one vault silently resolve to an unrelated note of the same
    name in another vault, merging their backlinks with no error or warning.
    Links to notes that do not exist in the corpus (or vault) are ignored
    rather than recorded as dangling edges.
    """
    keys = {(chunk.vault_id, PurePosixPath(chunk.source_path).stem) for chunk in chunks}
    inbound: dict[tuple[str | None, str], set[str]] = {key: set() for key in keys}

    for chunk in chunks:
        source_stem = PurePosixPath(chunk.source_path).stem
        for target in chunk.links_to:
            key = (chunk.vault_id, target)
            if key in inbound and target != source_stem:
                inbound[key].add(source_stem)

    return [
        replace(
            chunk,
            backlinks=tuple(
                sorted(inbound[(chunk.vault_id, PurePosixPath(chunk.source_path).stem)])
            ),
        )
        for chunk in chunks
    ]


def build_index(chunks: Sequence[Chunk], store: VectorStore, embedder: Embedder) -> BuildStats:
    """Write `chunks` into `store`, embedding only what is not already cached.

    Any source path present in the store but absent from `chunks` is deleted —
    this is how renames and deletions are handled, since a rename appears as a
    delete plus an add.
    """
    incoming_paths = {chunk.source_path for chunk in chunks}

    # Read the cache BEFORE deleting anything. The store is the cache, so
    # deleting a path's rows also destroys its vectors — reading afterwards
    # would report a 0% reuse rate and re-embed the entire corpus every run.
    cached = store.cached_vectors({chunk.content_hash for chunk in chunks})

    # Embed BEFORE any delete, too. embed() is the flaky, expensive step (a
    # real Bedrock call can throttle or time out); if it raises, the store
    # must still hold everything it held on entry — not be left emptied with
    # no vectors to write back.
    to_embed = [chunk for chunk in chunks if chunk.content_hash not in cached]
    if to_embed:
        fresh = embedder.embed([chunk.text for chunk in to_embed])
        for chunk, vector in zip(to_embed, fresh, strict=True):
            cached[chunk.content_hash] = vector

    # Every path in stale_paths came straight out of all_source_paths(), so
    # each one is guaranteed to currently exist - paths_deleted counts them
    # directly rather than trusting replace()'s return value, which also
    # counts incoming paths that get cleared-and-rewritten below.
    stale_paths = store.all_source_paths() - incoming_paths
    paths_deleted = len(stale_paths)

    # Delete stale paths AND incoming paths, then insert, as one atomic
    # write: a source whose chunk count shrank would otherwise leave orphaned
    # rows behind (upsert only replaces by id), and doing the deletes and the
    # insert in separate transactions is exactly what left the store emptied
    # when upsert failed after the deletes had already committed.
    store.replace(
        stale_paths | incoming_paths,
        list(chunks),
        [cached[chunk.content_hash] for chunk in chunks],
    )

    if not chunks:
        return BuildStats(0, 0, 0, paths_deleted)

    return BuildStats(
        chunks_written=len(chunks),
        vectors_embedded=len(to_embed),
        vectors_reused=len(chunks) - len(to_embed),
        paths_deleted=paths_deleted,
    )
