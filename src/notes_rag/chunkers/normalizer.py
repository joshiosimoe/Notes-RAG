"""Size normalization and context prefixing, applied to every chunker's output."""

import hashlib
from dataclasses import replace

from notes_rag.models import Chunk, estimate_tokens

DEFAULT_MIN_TOKENS = 150
DEFAULT_MAX_TOKENS = 800


def normalize(
    chunks: list[Chunk],
    *,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[Chunk]:
    """Merge undersized chunks, split oversized ones, then prefix and hash.

    Order is load-bearing: merge/split run on raw text, the context prefix is
    applied afterwards, and the hash is taken last over the prefixed text —
    which is exactly the string that gets embedded.
    """
    if not chunks:
        return []

    merged = _merge_small(chunks, min_tokens)
    split = _split_large(merged, max_tokens)
    return [_finalize(chunk) for chunk in split]


def _merge_small(chunks: list[Chunk], min_tokens: int) -> list[Chunk]:
    out: list[Chunk] = []
    pending: Chunk | None = None

    for chunk in chunks:
        current = (
            chunk if pending is None else replace(pending, text=f"{pending.text}\n\n{chunk.text}")
        )
        if estimate_tokens(current.text) < min_tokens:
            pending = current
        else:
            out.append(current)
            pending = None

    if pending is not None:
        # Nothing left to merge forward into: fold into the previous chunk if
        # there is one, otherwise keep it as its own undersized chunk.
        if out:
            out[-1] = replace(out[-1], text=f"{out[-1].text}\n\n{pending.text}")
        else:
            out.append(pending)

    return out


def _split_large(chunks: list[Chunk], max_tokens: int) -> list[Chunk]:
    out: list[Chunk] = []
    for chunk in chunks:
        if estimate_tokens(chunk.text) <= max_tokens:
            out.append(chunk)
            continue

        parts = _split_on_paragraphs(chunk.text, max_tokens)
        if len(parts) == 1:
            # No usable paragraph boundary. Emit intact rather than cutting
            # mid-sentence; an oversized chunk retrieves worse than a
            # truncated one reads.
            out.append(chunk)
            continue

        for index, part in enumerate(parts):
            out.append(replace(chunk, id=f"{chunk.id}.{index}", text=part))
    return out


def _split_on_paragraphs(text: str, max_tokens: int) -> list[str]:
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        return [text]

    parts: list[str] = []
    buffer: list[str] = []
    for paragraph in paragraphs:
        candidate = buffer + [paragraph]
        if buffer and estimate_tokens("\n\n".join(candidate)) > max_tokens:
            parts.append("\n\n".join(buffer))
            buffer = [paragraph]
        else:
            buffer = candidate
    if buffer:
        parts.append("\n\n".join(buffer))
    return parts


def _finalize(chunk: Chunk) -> Chunk:
    prefixed = f"{chunk.context}\n\n{chunk.text}" if chunk.context else chunk.text
    digest = hashlib.sha256(prefixed.encode()).hexdigest()
    return replace(chunk, text=prefixed, content_hash=digest)
