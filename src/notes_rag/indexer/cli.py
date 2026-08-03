"""CLI: build a local sqlite-vec index from a directory of source artifacts.

This is the wiring the rest of the package is missing - `chunk_markdown`,
`chunk_video_summary`, and `chunk_video_transcript` are otherwise only ever
called from their own tests. Walks `source`, dispatches each file to the
chunker matching its shape (Video Vault summary JSON, Video Vault transcript
JSON, or an Obsidian markdown note), derives backlinks across the whole set,
and writes the result via `build_index`.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from notes_rag.chunkers.markdown import chunk_markdown
from notes_rag.chunkers.video_summary import chunk_video_summary
from notes_rag.chunkers.video_transcript import chunk_video_transcript
from notes_rag.embed.base import Embedder
from notes_rag.embed.fake import FakeEmbedder
from notes_rag.indexer.build import BuildStats, build_index, derive_backlinks
from notes_rag.models import Chunk
from notes_rag.store.sqlite_vec import SqliteVecStore

SummaryDoc = tuple[dict, str]
TranscriptDoc = tuple[dict, str]
MarkdownDoc = tuple[str, str]


def _collect(source: Path) -> tuple[list[SummaryDoc], list[TranscriptDoc], list[MarkdownDoc]]:
    """Walk `source` and classify every file by artifact shape.

    Returns (summaries, transcripts, markdown_notes), each a list of
    (parsed content, source_path) pairs. `source_path` is `source`-relative
    and posix-separated - it's what ends up in `Chunk.source_path`.

    Summaries and transcripts are both JSON but distinguished by shape:
    a summary has a top-level `summary` object, a transcript has `segments`.
    Anything else - malformed JSON object, unrecognized shape, non-JSON,
    non-markdown - is skipped with a warning rather than aborting the run.
    """
    summaries: list[SummaryDoc] = []
    transcripts: list[TranscriptDoc] = []
    markdown_notes: list[MarkdownDoc] = []

    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source).as_posix()

        if path.suffix == ".md":
            markdown_notes.append((path.read_text(), rel))
            continue
        if path.suffix != ".json":
            continue

        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            print(f"skipping {rel}: JSON top level is not an object", file=sys.stderr)
        elif isinstance(data.get("summary"), dict) and "video_id" in data:
            summaries.append((data, rel))
        elif isinstance(data.get("segments"), list) and "video_id" in data:
            transcripts.append((data, rel))
        else:
            print(f"skipping {rel}: unrecognized JSON shape", file=sys.stderr)

    return summaries, transcripts, markdown_notes


def _build_chunks(
    summaries: list[SummaryDoc],
    transcripts: list[TranscriptDoc],
    markdown_notes: list[MarkdownDoc],
    *,
    vault_id: str,
) -> list[Chunk]:
    by_video_id = {summary["video_id"]: summary for summary, _ in summaries}

    chunks: list[Chunk] = []
    for summary, path in summaries:
        chunks.extend(chunk_video_summary(summary, source_path=path))

    for transcript, path in transcripts:
        video_id = transcript.get("video_id")
        summary = by_video_id.get(video_id)
        if summary is None:
            # chunk_video_transcript needs the summary for title/channel/url -
            # without it there's nothing to build citation fields from.
            print(
                f"skipping {path}: no summary found for video_id={video_id!r}",
                file=sys.stderr,
            )
            continue
        chunks.extend(chunk_video_transcript(transcript, summary, source_path=path))

    for text, path in markdown_notes:
        chunks.extend(chunk_markdown(text, source_path=path, vault_id=vault_id))

    return chunks


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

    summaries, transcripts, markdown_notes = _collect(Path(args.source))
    chunks = _build_chunks(summaries, transcripts, markdown_notes, vault_id=args.vault_id)
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
