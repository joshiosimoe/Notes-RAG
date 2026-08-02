"""Core data types shared by every chunker, store, and indexer."""

from dataclasses import dataclass, field


def estimate_tokens(text: str) -> int:
    """Rough token estimate used only for merge/split thresholds.

    Deliberately not a real tokenizer: this drives size decisions in the
    normalizer, where a 4-chars-per-token approximation is accurate enough
    and avoids a tokenizer dependency in the Lambda bundle.
    """
    return len(text) // 4


@dataclass(frozen=True)
class Chunk:
    """One embeddable unit of source material.

    `context` is the human-readable prefix prepended to `text` before
    embedding (see chunkers/normalizer.py). `content_hash` is computed over
    the *prefixed* text, because that is what actually gets embedded — it is
    the cache key for reusing vectors across index rebuilds.
    """

    id: str
    corpus: str  # "video" | "note" | "material"
    vault_id: str | None
    source_path: str
    chunk_type: str  # "summary" | "transcript" | "note" | "slide" | "sheet" | "page"
    title: str
    heading: str | None
    context: str
    text: str
    content_hash: str
    video_id: str | None = None
    start_seconds: int | None = None
    url: str | None = None
    links_to: tuple[str, ...] = field(default=())
    backlinks: tuple[str, ...] = field(default=())
