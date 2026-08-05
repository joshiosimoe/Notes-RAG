"""Turn raw source bytes into chunks, wherever those bytes came from.

The CLI reads files off disk and the indexer Lambda reads objects out of S3,
but both need the same artifact-shape dispatch and the same transcript-to-summary
pairing. Both build `SourceDocument`s and call in here.

Nothing in this module does IO or writes to stderr: skips are returned as
(source_path, reason) pairs so the caller can print them, log them, or assert
on them in a test.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass

from notes_rag.chunkers.markdown import chunk_markdown
from notes_rag.chunkers.video_summary import chunk_video_summary
from notes_rag.chunkers.video_transcript import chunk_video_transcript
from notes_rag.models import Chunk

Skip = tuple[str, str]


@dataclass(frozen=True)
class SourceDocument:
    source_path: str
    raw: bytes


@dataclass(frozen=True)
class CollectedDocuments:
    summaries: tuple[tuple[dict, str], ...]
    transcripts: tuple[tuple[dict, str], ...]
    markdown_notes: tuple[tuple[str, str], ...]
    skipped: tuple[Skip, ...]


def classify(documents: Sequence[SourceDocument]) -> CollectedDocuments:
    """Sort documents by artifact shape.

    Summaries and transcripts are both JSON, distinguished by shape: a summary
    has a top-level `summary` object, a transcript has `segments`. Anything
    unrecognized is skipped with a reason rather than aborting the run - one bad
    object must not stop the whole index from rebuilding.
    """
    summaries: list[tuple[dict, str]] = []
    transcripts: list[tuple[dict, str]] = []
    markdown_notes: list[tuple[str, str]] = []
    skipped: list[Skip] = []

    for document in documents:
        path = document.source_path

        if path.endswith(".md"):
            markdown_notes.append((document.raw.decode("utf-8", errors="replace"), path))
            continue
        if not path.endswith(".json"):
            skipped.append((path, "unhandled file suffix"))
            continue

        try:
            data = json.loads(document.raw)
        except json.JSONDecodeError as error:
            skipped.append((path, f"malformed JSON: {error}"))
            continue

        if not isinstance(data, dict):
            skipped.append((path, "JSON top level is not an object"))
        elif isinstance(data.get("summary"), dict) and "video_id" in data:
            summaries.append((data, path))
        elif isinstance(data.get("segments"), list) and "video_id" in data:
            transcripts.append((data, path))
        else:
            skipped.append((path, "unrecognized JSON shape"))

    return CollectedDocuments(
        summaries=tuple(summaries),
        transcripts=tuple(transcripts),
        markdown_notes=tuple(markdown_notes),
        skipped=tuple(skipped),
    )


def build_chunks(
    collected: CollectedDocuments, *, vault_id: str
) -> tuple[list[Chunk], tuple[Skip, ...]]:
    """Chunk everything classified. Returns (chunks, skips from this stage).

    The returned skips are only those discovered while chunking - a transcript
    with no matching summary. `collected.skipped` from classification is the
    caller's to report; the two are kept separate so a caller can tell a
    malformed file from an unpairable one.
    """
    by_video_id = {summary["video_id"]: summary for summary, _ in collected.summaries}

    chunks: list[Chunk] = []
    skipped: list[Skip] = []

    for summary, path in collected.summaries:
        chunks.extend(chunk_video_summary(summary, source_path=path))

    for transcript, path in collected.transcripts:
        video_id = transcript.get("video_id")
        summary = by_video_id.get(video_id)
        if summary is None:
            # chunk_video_transcript needs the summary for title/channel/url -
            # without it there is nothing to build citation fields from.
            skipped.append((path, f"no summary found for video_id={video_id!r}"))
            continue
        chunks.extend(chunk_video_transcript(transcript, summary, source_path=path))

    for text, path in collected.markdown_notes:
        chunks.extend(chunk_markdown(text, source_path=path, vault_id=vault_id))

    return chunks, tuple(skipped)
