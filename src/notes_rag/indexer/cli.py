"""CLI: build a local sqlite-vec index from a directory of source artifacts.

This is the wiring the rest of the package is missing - `chunk_markdown`,
`chunk_video_summary`, and `chunk_video_transcript` are otherwise only ever
called from their own tests. Walks `source`, dispatches each file to the
chunker matching its shape (Video Vault summary JSON, Video Vault transcript
JSON, or an Obsidian markdown note), derives backlinks across the whole set,
and writes the result via `build_index`.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from notes_rag.embed.base import Embedder
from notes_rag.embed.fake import FakeEmbedder
from notes_rag.indexer.build import BuildStats, build_index, derive_backlinks
from notes_rag.indexer.collect import (
    Skip,
    SourceDocument,
    build_chunks,
    classify,
    unsupported_suffix_skip,
)
from notes_rag.store.sqlite_vec import SqliteVecStore


def _read_documents(source: Path, vault_id: str) -> tuple[list[SourceDocument], list[Skip]]:
    """Every file under `source`, as SourceDocuments with source-relative paths,
    plus the skips for files whose suffix nothing here reads.

    `source_path` is posix-separated because it ends up in `Chunk.source_path`,
    which must match the S3 key layout the indexer Lambda produces.

    The suffix is decided from the relative path alone, before `read_bytes()`
    runs, so an unsupported file is never loaded into memory just to find out
    it will be skipped - mirrors the same guard in the Lambda handler.

    Every document carries the CLI's --vault-id. Locally there is one vault and
    no S3 prefix, so display_path and source_path are the same value.
    """
    documents: list[SourceDocument] = []
    skipped: list[Skip] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source).as_posix()
        skip = unsupported_suffix_skip(rel)
        if skip is not None:
            skipped.append(skip)
            continue
        documents.append(
            SourceDocument(
                source_path=rel,
                raw=path.read_bytes(),
                vault_id=vault_id,
                display_path=rel,
            )
        )
    return documents, skipped


def _print_stats(stats: BuildStats) -> None:
    print(f"chunks_written:   {stats.chunks_written}")
    print(f"vectors_embedded: {stats.vectors_embedded}")
    print(f"vectors_reused:   {stats.vectors_reused}")
    print(f"paths_deleted:    {stats.paths_deleted}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a local sqlite-vec index from Video Vault and Obsidian source files."
    )
    parser.add_argument(
        "source", help="directory to walk for summary/transcript JSON and markdown notes"
    )
    parser.add_argument("--out", required=True, help="path to write the .db index")
    parser.add_argument(
        "--vault-id", required=True, help="vault id recorded on markdown-derived chunks"
    )
    parser.add_argument(
        "--dimensions", type=int, default=1024, help="embedding vector width (default: 1024)"
    )
    parser.add_argument(
        "--fake-embedder",
        action="store_true",
        help="use the deterministic FakeEmbedder instead of TitanEmbedder "
        "(no AWS credentials required; for tests and dry runs)",
    )
    args = parser.parse_args(argv)

    documents, suffix_skips = _read_documents(Path(args.source), args.vault_id)
    collected = classify(documents)
    chunks, unpairable = build_chunks(collected)
    for path, reason in [*suffix_skips, *collected.skipped, *unpairable]:
        print(f"skipping {path}: {reason}", file=sys.stderr)
    chunks = derive_backlinks(chunks)

    embedder: Embedder
    if args.fake_embedder:
        embedder = FakeEmbedder(dimensions=args.dimensions)
    else:
        # Lazy import, exactly like eval/run.py's main(): importing this
        # module must never touch boto3, only constructing a real embedder
        # does.
        from notes_rag.embed.bedrock import TitanEmbedder

        embedder = TitanEmbedder(dimensions=args.dimensions)

    store = SqliteVecStore(args.out, dimensions=args.dimensions)
    try:
        stats = build_index(chunks, store, embedder)
    finally:
        store.close()

    _print_stats(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
