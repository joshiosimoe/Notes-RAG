"""Chunk an Obsidian markdown note on headings, capturing its wikilink graph.

`links_to` is populated but unused by retrieval in v1 (spec §2 decision 15):
capturing it now makes gated expansion a query-side change later, with no
re-index. `backlinks` is the inverse relation and is filled in at index-build
time, once every note in the corpus is known.
"""

import re
from pathlib import PurePosixPath

import yaml

from notes_rag.chunkers.normalizer import normalize
from notes_rag.models import Chunk

# [[Note]] | [[Note|alias]] | [[Note#Heading]] | ![[Note]]
# Captures the note name only, discarding any #heading anchor and |alias.
_WIKILINK = re.compile(r"!?\[\[([^\]\|#]+)(?:#[^\]\|]*)?(?:\|[^\]]*)?\]\]")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def extract_wikilinks(text: str) -> tuple[str, ...]:
    """Return wikilink targets in document order, deduplicated."""
    seen: dict[str, None] = {}
    for match in _WIKILINK.finditer(text):
        seen.setdefault(match.group(1).strip(), None)
    return tuple(seen)


def chunk_markdown(text: str, *, source_path: str, vault_id: str) -> list[Chunk]:
    frontmatter, body = _split_frontmatter(text)
    if not body.strip():
        return []

    title = frontmatter.get("title") or PurePosixPath(source_path).stem
    links = extract_wikilinks(body)

    chunks: list[Chunk] = []
    for ordinal, (heading, section_text) in enumerate(_split_on_headings(body)):
        label = heading or title
        chunks.append(
            Chunk(
                id=f"note:{source_path}#{ordinal}",
                corpus="note",
                vault_id=vault_id,
                source_path=source_path,
                chunk_type="note",
                title=title,
                heading=heading,
                context=f"{vault_id} / {source_path} / {label}",
                text=section_text,
                content_hash="",
                links_to=links,
            )
        )

    return normalize(chunks)


def _split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    raw = text[4:end]
    rest = text[end + 4 :].lstrip("\n")
    try:
        parsed = yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        parsed = {}
    return (parsed if isinstance(parsed, dict) else {}), rest


def _split_on_headings(body: str) -> list[tuple[str | None, str]]:
    """Split into (heading, text) pairs. Text before the first heading gets None.

    The heading line is kept in the section body as well as in the `heading`
    field. It is real content and should be searchable — and if the normalizer
    merges several small sections together, only the first chunk's context
    prefix survives, so a heading that lived only in the prefix would vanish.
    """
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in body.splitlines():
        match = _HEADING.match(line)
        if match:
            sections.append((match.group(2).strip(), [line]))
        else:
            sections[-1][1].append(line)

    out: list[tuple[str | None, str]] = []
    for heading, lines in sections:
        joined = "\n".join(lines).strip()
        if joined:
            out.append((heading, joined))
    return out
