"""Chunk a Video Vault transcript, aligned to the summary's section boundaries.

Splitting on `summary.sections[].start_seconds` rather than a fixed window means
every transcript chunk inherits a real timestamp, so citations deep-link into the
video without a second source of truth for boundaries.
"""

from notes_rag.chunkers.normalizer import normalize
from notes_rag.models import Chunk


def chunk_video_transcript(transcript: dict, summary: dict, *, source_path: str) -> list[Chunk]:
    segments = transcript.get("segments") or []
    if not segments:
        return []

    video_id = summary["video_id"]
    title = summary["title"]
    channel = summary["channel"]
    url = summary["url"]

    boundaries = _boundaries(summary)
    buckets: dict[int, list[str]] = {start: [] for start, _ in boundaries}

    for segment in segments:
        start = int(segment["start_seconds"])
        bucket_start = _bucket_for(start, boundaries)
        buckets[bucket_start].append(segment["text"])

    chunks: list[Chunk] = []
    for ordinal, (start, heading) in enumerate(boundaries):
        texts = buckets[start]
        if not texts:
            continue
        chunks.append(
            Chunk(
                id=f"video-transcript:{source_path}#{ordinal}",
                corpus="video",
                vault_id=None,
                source_path=source_path,
                chunk_type="transcript",
                title=title,
                heading=heading,
                context=f"{title} — {channel} — {heading} (transcript)",
                text=" ".join(texts),
                content_hash="",
                video_id=video_id,
                start_seconds=start,
                url=url,
            )
        )

    return normalize(chunks)


def _boundaries(summary: dict) -> list[tuple[int, str]]:
    """Return (start_seconds, heading) pairs, always starting at 0."""
    sections = summary.get("summary", {}).get("sections") or []
    pairs = sorted((int(section["start_seconds"]), section["title"]) for section in sections)
    if not pairs or pairs[0][0] != 0:
        pairs.insert(0, (0, "Opening"))
    return pairs


def _bucket_for(start: int, boundaries: list[tuple[int, str]]) -> int:
    """The last boundary at or before `start`."""
    chosen = boundaries[0][0]
    for boundary_start, _ in boundaries:
        if boundary_start <= start:
            chosen = boundary_start
        else:
            break
    return chosen
