"""Chunk a Video Vault summary artifact.

Artifact shape is fixed by Video Vault (see docs/rag/BRIEF.md §3). It is
self-contained: title, channel, and url are repeated on every object, so
chunking needs no external lookup.
"""

from notes_rag.chunkers.normalizer import normalize
from notes_rag.models import Chunk


def chunk_video_summary(summary: dict, *, source_path: str) -> list[Chunk]:
    video_id = summary["video_id"]
    title = summary["title"]
    channel = summary["channel"]
    url = summary["url"]
    body = summary["summary"]

    chunks: list[Chunk] = [
        _make(
            ordinal=0,
            heading="Overview",
            text=_overview_text(body),
            start_seconds=0,
            video_id=video_id,
            title=title,
            channel=channel,
            url=url,
            source_path=source_path,
        )
    ]

    for index, section in enumerate(body.get("sections") or [], start=1):
        chunks.append(
            _make(
                ordinal=index,
                heading=section["title"],
                text=section["summary"],
                start_seconds=int(section["start_seconds"]),
                video_id=video_id,
                title=title,
                channel=channel,
                url=url,
                source_path=source_path,
            )
        )

    return normalize(chunks)


def _overview_text(body: dict) -> str:
    parts = [body.get("verdict", ""), body.get("tldr", "")]
    takeaways = body.get("takeaways") or []
    if takeaways:
        parts.append("\n".join(f"- {item}" for item in takeaways))
    return "\n\n".join(part for part in parts if part)


def _make(
    *,
    ordinal: int,
    heading: str,
    text: str,
    start_seconds: int,
    video_id: str,
    title: str,
    channel: str,
    url: str,
    source_path: str,
) -> Chunk:
    return Chunk(
        id=f"video:{source_path}#{ordinal}",
        corpus="video",
        vault_id=None,
        source_path=source_path,
        chunk_type="summary",
        title=title,
        heading=heading,
        context=f"{title} — {channel} — {heading}",
        text=text,
        content_hash="",
        video_id=video_id,
        start_seconds=start_seconds,
        url=url,
    )
