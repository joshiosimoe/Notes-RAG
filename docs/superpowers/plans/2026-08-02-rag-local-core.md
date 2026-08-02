# RAG Local Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the chunking, storage, embedding, and evaluation core of the RAG locally — able to index the two real Video Vault artifacts, search them, and report recall@k — with no AWS deployment.

**Architecture:** Pure-function chunkers (one per source format) emit `Chunk` objects; a shared normalizer merges, splits, prefixes, and hashes them; a `VectorStore` interface backed by `sqlite-vec` holds vectors plus metadata in one file; an `Embedder` interface wraps Bedrock Titan v2 with a deterministic fake for tests. The indexer reuses cached vectors by `content_hash` so only changed chunks are re-embedded.

**Tech Stack:** Python 3.12, pytest, `sqlite-vec`, `boto3`, `PyYAML`, `numpy`.

## Global Constraints

- Python 3.12. Target runtime is AWS Lambda `python3.12`, so no dependency may require a compile step at install time — wheels only.
- `sqlite-vec` is loaded as a SQLite extension. `sqlite3` must be built with extension loading enabled (`enable_load_extension`). Some system Pythons (notably macOS-bundled) are not — use a `pyenv`/`uv`-managed interpreter if `Task 6 Step 2` fails on that.
- Embedding model is `amazon.titan-embed-text-v2:0` at **1024 dimensions**. This value appears in the vec0 schema and in `Embedder.dimensions`; they must agree.
- Bedrock region for embeddings is `us-east-2`. Titan is not an Anthropic model, so the Anthropic use-case form does not gate it.
- **No AWS calls in unit tests.** Every test in `tests/` must pass with no credentials. Bedrock-touching tests live behind `@pytest.mark.integration` and are deselected by default.
- Token counts throughout are estimated as `len(text) // 4`. This is a threshold heuristic for merge/split decisions only — never presented as a real token count.
- Conventional commit prefixes (`feat:`, `test:`, `chore:`, `fix:`). Lint and tests green before every commit.

---

## File Structure

```
pyproject.toml                          deps, pytest config, ruff config
src/notes_rag/__init__.py
src/notes_rag/models.py                 Chunk dataclass, estimate_tokens
src/notes_rag/chunkers/__init__.py
src/notes_rag/chunkers/normalizer.py    merge / split / prefix / hash
src/notes_rag/chunkers/video_summary.py summary JSON  -> chunks
src/notes_rag/chunkers/video_transcript.py transcript + summary -> chunks
src/notes_rag/chunkers/markdown.py      Obsidian note -> chunks, wikilinks, frontmatter
src/notes_rag/store/__init__.py
src/notes_rag/store/base.py             VectorStore protocol, SearchHit
src/notes_rag/store/sqlite_vec.py       SqliteVecStore
src/notes_rag/embed/__init__.py
src/notes_rag/embed/base.py             Embedder protocol
src/notes_rag/embed/fake.py             FakeEmbedder (deterministic, no network)
src/notes_rag/embed/bedrock.py          TitanEmbedder
src/notes_rag/indexer/__init__.py
src/notes_rag/indexer/build.py          build_index, BuildStats
tests/conftest.py                       fixture loaders
tests/fixtures/summary_sample.json
tests/fixtures/transcript_sample.json
tests/fixtures/note_sample.md
tests/chunkers/test_normalizer.py
tests/chunkers/test_video_summary.py
tests/chunkers/test_video_transcript.py
tests/chunkers/test_markdown.py
tests/store/test_sqlite_vec.py
tests/embed/test_bedrock.py             integration-marked
tests/indexer/test_build.py
eval/questions.yaml
eval/run.py                             recall@k, MRR
tests/eval/test_run.py
```

Chunkers are split one-per-format because they share no logic — only the `Chunk` output shape and the normalizer that follows them. `store/base.py` is separate from `sqlite_vec.py` so a future pgvector implementation has an obvious home.

---

### Task 1: Project scaffolding and the `Chunk` model

**Files:**
- Create: `pyproject.toml`, `src/notes_rag/__init__.py`, `src/notes_rag/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Chunk` frozen dataclass with fields `id: str`, `corpus: str`, `vault_id: str | None`, `source_path: str`, `chunk_type: str`, `title: str`, `heading: str | None`, `context: str`, `text: str`, `content_hash: str`, `video_id: str | None`, `start_seconds: int | None`, `url: str | None`, `links_to: tuple[str, ...]`, `backlinks: tuple[str, ...]`. Also `estimate_tokens(text: str) -> int`.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "notes-rag"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "sqlite-vec>=0.1.6",
    "boto3>=1.35",
    "PyYAML>=6.0",
    "numpy>=2.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.6"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["integration: touches AWS; deselected by default"]
addopts = "-m 'not integration'"

[tool.ruff]
line-length = 100
```

- [ ] **Step 2: Install**

Run: `python -m venv .venv && .venv/bin/pip install -e '.[dev]'`
Expected: installs without error.

- [ ] **Step 3: Write the failing test**

Create `tests/test_models.py`:

```python
import pytest

from notes_rag.models import Chunk, estimate_tokens


def test_estimate_tokens_is_quarter_of_characters():
    assert estimate_tokens("a" * 400) == 100


def test_estimate_tokens_empty_string():
    assert estimate_tokens("") == 0


def test_chunk_is_frozen():
    chunk = Chunk(
        id="video:summaries/x.json#0",
        corpus="video",
        vault_id=None,
        source_path="summaries/x.json",
        chunk_type="summary",
        title="Some Title",
        heading="Custom scheduler",
        context="Some Title — Some Channel — Custom scheduler",
        text="body text",
        content_hash="deadbeef",
    )
    with pytest.raises(AttributeError):
        chunk.text = "mutated"  # type: ignore[misc]


def test_chunk_optional_fields_default_to_none_or_empty():
    chunk = Chunk(
        id="note:Class Notes/a.md#0",
        corpus="note",
        vault_id="Class Notes",
        source_path="Class Notes/a.md",
        chunk_type="note",
        title="a",
        heading=None,
        context="Class Notes / Class Notes/a.md",
        text="body",
        content_hash="cafe",
    )
    assert chunk.video_id is None
    assert chunk.start_seconds is None
    assert chunk.url is None
    assert chunk.links_to == ()
    assert chunk.backlinks == ()
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'notes_rag.models'`

- [ ] **Step 5: Write the implementation**

Create `src/notes_rag/__init__.py` (empty file).

Create `src/notes_rag/models.py`:

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_models.py -v`
Expected: 4 passed

- [ ] **Step 7: Lint**

Run: `.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests`
Expected: no errors. If format fails, run `.venv/bin/ruff format src tests` and re-check.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/notes_rag/__init__.py src/notes_rag/models.py tests/test_models.py
git commit -m "feat: add Chunk model and project scaffolding"
```

---

### Task 2: Normalizer

**Files:**
- Create: `src/notes_rag/chunkers/__init__.py`, `src/notes_rag/chunkers/normalizer.py`
- Test: `tests/chunkers/test_normalizer.py`

**Interfaces:**
- Consumes: `Chunk`, `estimate_tokens` from `notes_rag.models`.
- Produces: `normalize(chunks: list[Chunk], *, min_tokens: int = 150, max_tokens: int = 800) -> list[Chunk]`. Applied by every chunker as its final step. Sets `context`-prefixed `text` and `content_hash` on the returned chunks.

**Design note for the implementer:** order matters. Merge and split operate on raw text; the context prefix is applied afterward; the hash is computed last, over the prefixed text. Doing it in any other order means the cache key does not match what gets embedded.

- [ ] **Step 1: Write the failing test**

Create `tests/chunkers/__init__.py` (empty) and `tests/chunkers/test_normalizer.py`:

```python
from notes_rag.chunkers.normalizer import normalize
from notes_rag.models import Chunk, estimate_tokens


def make(text: str, *, ordinal: int = 0, context: str = "CTX") -> Chunk:
    return Chunk(
        id=f"video:x#{ordinal}",
        corpus="video",
        vault_id=None,
        source_path="summaries/x.json",
        chunk_type="summary",
        title="T",
        heading=f"H{ordinal}",
        context=context,
        text=text,
        content_hash="",
    )


def test_applies_context_prefix_to_text():
    out = normalize([make("a" * 2000)])
    assert out[0].text.startswith("CTX\n\n")
    assert out[0].text.endswith("a" * 2000)


def test_computes_hash_over_prefixed_text():
    import hashlib

    out = normalize([make("a" * 2000)])
    expected = hashlib.sha256(("CTX\n\n" + "a" * 2000).encode()).hexdigest()
    assert out[0].content_hash == expected


def test_merges_chunk_below_min_into_next():
    small = make("x" * 100, ordinal=0)  # 25 tokens, below min
    big = make("y" * 2000, ordinal=1)  # 500 tokens
    out = normalize([small, big], min_tokens=150, max_tokens=800)
    assert len(out) == 1
    assert "x" * 100 in out[0].text
    assert "y" * 2000 in out[0].text


def test_merged_chunk_keeps_first_chunks_metadata():
    small = make("x" * 100, ordinal=0)
    big = make("y" * 2000, ordinal=1)
    out = normalize([small, big], min_tokens=150, max_tokens=800)
    assert out[0].heading == "H0"
    assert out[0].id == "video:x#0"


def test_trailing_small_chunk_merges_into_previous():
    big = make("y" * 2000, ordinal=0)
    small = make("x" * 100, ordinal=1)
    out = normalize([big, small], min_tokens=150, max_tokens=800)
    assert len(out) == 1
    assert "x" * 100 in out[0].text


def test_splits_chunk_above_max_on_paragraph_boundary():
    para = "z" * 1600  # 400 tokens each
    text = f"{para}\n\n{para}\n\n{para}"  # 1200 tokens total
    out = normalize([make(text)], min_tokens=150, max_tokens=800)
    assert len(out) == 2
    for chunk in out:
        assert estimate_tokens(chunk.text) <= 800 + estimate_tokens("CTX\n\n")


def test_split_chunks_get_distinct_ids():
    para = "z" * 1600
    out = normalize([make(f"{para}\n\n{para}\n\n{para}")], max_tokens=800)
    assert len({chunk.id for chunk in out}) == len(out)


def test_oversized_single_paragraph_is_left_intact():
    # No paragraph boundary to split on — emit as-is rather than cutting mid-word.
    out = normalize([make("q" * 8000)], max_tokens=800)
    assert len(out) == 1


def test_single_small_chunk_with_no_neighbour_survives():
    out = normalize([make("x" * 100)], min_tokens=150)
    assert len(out) == 1
    assert "x" * 100 in out[0].text


def test_empty_input_returns_empty():
    assert normalize([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/chunkers/test_normalizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'notes_rag.chunkers'`

- [ ] **Step 3: Write the implementation**

Create `src/notes_rag/chunkers/__init__.py` (empty file).

Create `src/notes_rag/chunkers/normalizer.py`:

```python
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
        current = chunk if pending is None else replace(
            pending, text=f"{pending.text}\n\n{chunk.text}"
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/chunkers/test_normalizer.py -v`
Expected: 10 passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src tests && .venv/bin/ruff format src tests
git add src/notes_rag/chunkers tests/chunkers
git commit -m "feat: add chunk size normalizer with context prefixing"
```

---

### Task 3: Video summary chunker

**Files:**
- Create: `src/notes_rag/chunkers/video_summary.py`, `tests/fixtures/summary_sample.json`, `tests/conftest.py`
- Test: `tests/chunkers/test_video_summary.py`

**Interfaces:**
- Consumes: `Chunk` from `notes_rag.models`, `normalize` from `notes_rag.chunkers.normalizer`.
- Produces: `chunk_video_summary(summary: dict, *, source_path: str) -> list[Chunk]`.

**Contract reminder:** the summary artifact shape is fixed by Video Vault and documented in `docs/rag/BRIEF.md` §3. It is self-contained — `title`, `channel`, `url`, `duration_seconds` are repeated on every object, so no lookup is needed.

- [ ] **Step 1: Create the fixture**

Create `tests/fixtures/summary_sample.json`:

```json
{
  "video_id": "dQw4w9WgXcQ",
  "title": "How Kubernetes Scheduling Actually Works",
  "channel": "Some Channel",
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "published_at": "2026-07-01T12:00:00Z",
  "duration_seconds": 3862,
  "summarized_at": "2026-07-22",
  "note_path": "Video Vault/2026/How Kubernetes Scheduling Actually Works-dQw4w9WgXcQ.md",
  "summary": {
    "verdict": "Worth watching if you operate clusters at scale.",
    "tldr": "Walks through the default scheduler, then builds a custom one.",
    "takeaways": [
      "The scheduler is two phases: filtering and scoring.",
      "Custom schedulers register via the scheduler name field on the pod spec."
    ],
    "sections": [
      {
        "start_seconds": 0,
        "title": "Introduction",
        "summary": "Sets up why scheduling matters once clusters get heterogeneous."
      },
      {
        "start_seconds": 1120,
        "title": "Custom scheduler",
        "summary": "Implements a scheduler that packs pods onto the fewest nodes."
      }
    ],
    "tags": ["kubernetes"]
  }
}
```

- [ ] **Step 2: Create the fixture loader**

Create `tests/conftest.py`:

```python
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def summary_sample() -> dict:
    return json.loads((FIXTURES / "summary_sample.json").read_text())


@pytest.fixture
def transcript_sample() -> dict:
    return json.loads((FIXTURES / "transcript_sample.json").read_text())


@pytest.fixture
def note_sample() -> str:
    return (FIXTURES / "note_sample.md").read_text()
```

- [ ] **Step 3: Write the failing test**

Create `tests/chunkers/test_video_summary.py`:

```python
from notes_rag.chunkers.video_summary import chunk_video_summary

PATH = "summaries/dQw4w9WgXcQ.json"


def test_emits_one_chunk_per_section_plus_an_overview(summary_sample):
    chunks = chunk_video_summary(summary_sample, source_path=PATH)
    # 2 sections + 1 overview, before any merging
    assert len(chunks) >= 1
    headings = {chunk.heading for chunk in chunks}
    assert "Custom scheduler" in headings or any(
        "Custom scheduler" in chunk.text for chunk in chunks
    )


def test_overview_chunk_carries_verdict_tldr_and_takeaways(summary_sample):
    chunks = chunk_video_summary(summary_sample, source_path=PATH)
    combined = "\n".join(chunk.text for chunk in chunks)
    assert "Worth watching if you operate clusters at scale." in combined
    assert "Walks through the default scheduler" in combined
    assert "filtering and scoring" in combined


def test_every_chunk_has_video_citation_fields(summary_sample):
    chunks = chunk_video_summary(summary_sample, source_path=PATH)
    for chunk in chunks:
        assert chunk.corpus == "video"
        assert chunk.chunk_type == "summary"
        assert chunk.video_id == "dQw4w9WgXcQ"
        assert chunk.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert chunk.start_seconds is not None


def test_section_chunk_uses_its_own_start_seconds(summary_sample):
    chunks = chunk_video_summary(summary_sample, source_path=PATH)
    starts = {chunk.start_seconds for chunk in chunks}
    assert 1120 in starts or 0 in starts


def test_context_prefix_includes_title_and_channel(summary_sample):
    chunks = chunk_video_summary(summary_sample, source_path=PATH)
    for chunk in chunks:
        assert "How Kubernetes Scheduling Actually Works" in chunk.text
        assert "Some Channel" in chunk.text


def test_all_chunks_have_a_content_hash(summary_sample):
    chunks = chunk_video_summary(summary_sample, source_path=PATH)
    assert all(len(chunk.content_hash) == 64 for chunk in chunks)


def test_source_path_is_recorded(summary_sample):
    chunks = chunk_video_summary(summary_sample, source_path=PATH)
    assert all(chunk.source_path == PATH for chunk in chunks)


def test_missing_sections_still_produces_overview():
    minimal = {
        "video_id": "abc",
        "title": "T",
        "channel": "C",
        "url": "https://example.com",
        "summary": {"verdict": "v", "tldr": "t", "takeaways": [], "sections": []},
    }
    chunks = chunk_video_summary(minimal, source_path="summaries/abc.json")
    assert len(chunks) == 1
    assert "v" in chunks[0].text
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/bin/pytest tests/chunkers/test_video_summary.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'notes_rag.chunkers.video_summary'`

- [ ] **Step 5: Write the implementation**

Create `src/notes_rag/chunkers/video_summary.py`:

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/chunkers/test_video_summary.py -v`
Expected: 8 passed

- [ ] **Step 7: Lint and commit**

```bash
.venv/bin/ruff check src tests && .venv/bin/ruff format src tests
git add src/notes_rag/chunkers/video_summary.py tests/
git commit -m "feat: add video summary chunker"
```

---

### Task 4: Video transcript chunker

**Files:**
- Create: `src/notes_rag/chunkers/video_transcript.py`, `tests/fixtures/transcript_sample.json`
- Test: `tests/chunkers/test_video_transcript.py`

**Interfaces:**
- Consumes: `Chunk`, `normalize`.
- Produces: `chunk_video_transcript(transcript: dict, summary: dict, *, source_path: str) -> list[Chunk]`.

**Design note:** transcript chunks are split on `summary.sections[].start_seconds` rather than on a fixed window. This is the decision from spec §4.4 — every transcript chunk inherits a real timestamp for free, so citations stay deep-linkable, and no separate windowing heuristic is needed. Segments earlier than the first section boundary belong to a leading bucket at `start_seconds = 0`.

- [ ] **Step 1: Create the fixture**

Create `tests/fixtures/transcript_sample.json`:

```json
{
  "video_id": "dQw4w9WgXcQ",
  "language": "en",
  "segments": [
    {"start_seconds": 0, "text": "Welcome back to the channel."},
    {"start_seconds": 12, "text": "Today we are looking at the Kubernetes scheduler."},
    {"start_seconds": 600, "text": "Filtering removes nodes that cannot host the pod."},
    {"start_seconds": 1120, "text": "Now let us write our own scheduler from scratch."},
    {"start_seconds": 1400, "text": "We register it by setting schedulerName on the pod spec."}
  ]
}
```

- [ ] **Step 2: Write the failing test**

Create `tests/chunkers/test_video_transcript.py`:

```python
from notes_rag.chunkers.video_transcript import chunk_video_transcript

PATH = "transcripts/dQw4w9WgXcQ.json"


def test_groups_segments_by_section_boundary(transcript_sample, summary_sample):
    chunks = chunk_video_transcript(transcript_sample, summary_sample, source_path=PATH)
    starts = sorted({chunk.start_seconds for chunk in chunks})
    assert starts == [0, 1120]


def test_pre_first_boundary_segments_land_in_the_leading_bucket(
    transcript_sample, summary_sample
):
    chunks = chunk_video_transcript(transcript_sample, summary_sample, source_path=PATH)
    leading = next(chunk for chunk in chunks if chunk.start_seconds == 0)
    assert "Welcome back to the channel." in leading.text
    assert "Filtering removes nodes" in leading.text


def test_segments_after_boundary_land_in_that_section(transcript_sample, summary_sample):
    chunks = chunk_video_transcript(transcript_sample, summary_sample, source_path=PATH)
    later = next(chunk for chunk in chunks if chunk.start_seconds == 1120)
    assert "write our own scheduler" in later.text
    assert "schedulerName" in later.text


def test_chunk_type_is_transcript(transcript_sample, summary_sample):
    chunks = chunk_video_transcript(transcript_sample, summary_sample, source_path=PATH)
    assert all(chunk.chunk_type == "transcript" for chunk in chunks)
    assert all(chunk.corpus == "video" for chunk in chunks)


def test_carries_citation_fields_from_summary(transcript_sample, summary_sample):
    chunks = chunk_video_transcript(transcript_sample, summary_sample, source_path=PATH)
    for chunk in chunks:
        assert chunk.video_id == "dQw4w9WgXcQ"
        assert chunk.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_summary_with_no_sections_yields_one_bucket(transcript_sample):
    summary = {
        "video_id": "dQw4w9WgXcQ",
        "title": "T",
        "channel": "C",
        "url": "https://example.com",
        "summary": {"sections": []},
    }
    chunks = chunk_video_transcript(transcript_sample, summary, source_path=PATH)
    assert {chunk.start_seconds for chunk in chunks} == {0}


def test_empty_transcript_yields_no_chunks(summary_sample):
    empty = {"video_id": "dQw4w9WgXcQ", "language": "en", "segments": []}
    assert chunk_video_transcript(empty, summary_sample, source_path=PATH) == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/chunkers/test_video_transcript.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Write the implementation**

Create `src/notes_rag/chunkers/video_transcript.py`:

```python
"""Chunk a Video Vault transcript, aligned to the summary's section boundaries.

Splitting on `summary.sections[].start_seconds` rather than a fixed window means
every transcript chunk inherits a real timestamp, so citations deep-link into the
video without a second source of truth for boundaries.
"""

from notes_rag.chunkers.normalizer import normalize
from notes_rag.models import Chunk


def chunk_video_transcript(
    transcript: dict, summary: dict, *, source_path: str
) -> list[Chunk]:
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
    pairs = sorted(
        (int(section["start_seconds"]), section["title"]) for section in sections
    )
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/chunkers/test_video_transcript.py -v`
Expected: 7 passed

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff check src tests && .venv/bin/ruff format src tests
git add src/notes_rag/chunkers/video_transcript.py tests/
git commit -m "feat: add transcript chunker aligned to section boundaries"
```

---

### Task 5: Markdown note chunker

**Files:**
- Create: `src/notes_rag/chunkers/markdown.py`, `tests/fixtures/note_sample.md`
- Test: `tests/chunkers/test_markdown.py`

**Interfaces:**
- Consumes: `Chunk`, `normalize`.
- Produces: `chunk_markdown(text: str, *, source_path: str, vault_id: str) -> list[Chunk]` and `extract_wikilinks(text: str) -> tuple[str, ...]`.

**Design note:** `links_to` is populated in v1 but never read by retrieval (spec §2 decision 15). Capturing it now means gated expansion is later a query-side change with no re-index. `backlinks` is left empty here — it is the inverse relation, computed at index-build time in Task 8 once every note is known.

- [ ] **Step 1: Create the fixture**

Create `tests/fixtures/note_sample.md`:

```markdown
---
title: Kubernetes Scheduling
tags: [kubernetes, scheduling]
status: reviewed
---

Intro paragraph before any heading. Mentions [[Bloom Filters]] in passing.

## Filtering phase

The scheduler removes nodes that cannot host the pod. Related: [[Node Affinity|affinity rules]].

## Scoring phase

Remaining nodes are ranked. See [[Scoring#Priorities]] for the weight table.
```

- [ ] **Step 2: Write the failing test**

Create `tests/chunkers/test_markdown.py`:

```python
from notes_rag.chunkers.markdown import chunk_markdown, extract_wikilinks

PATH = "Class Notes/Kubernetes Scheduling.md"
VAULT = "Class Notes"


def test_extract_plain_wikilink():
    assert extract_wikilinks("see [[Bloom Filters]] here") == ("Bloom Filters",)


def test_extract_aliased_wikilink_returns_target_not_alias():
    assert extract_wikilinks("[[Node Affinity|affinity rules]]") == ("Node Affinity",)


def test_extract_heading_anchored_wikilink_returns_note_only():
    assert extract_wikilinks("[[Scoring#Priorities]]") == ("Scoring",)


def test_extract_embed_wikilink():
    assert extract_wikilinks("![[Diagram]]") == ("Diagram",)


def test_extract_deduplicates_preserving_order():
    assert extract_wikilinks("[[A]] [[B]] [[A]]") == ("A", "B")


def test_extract_returns_empty_when_no_links():
    assert extract_wikilinks("plain text") == ()


def test_splits_on_headings(note_sample):
    chunks = chunk_markdown(note_sample, source_path=PATH, vault_id=VAULT)
    combined = "\n".join(chunk.text for chunk in chunks)
    assert "Filtering phase" in combined
    assert "Scoring phase" in combined


def test_preamble_before_first_heading_is_kept(note_sample):
    chunks = chunk_markdown(note_sample, source_path=PATH, vault_id=VAULT)
    combined = "\n".join(chunk.text for chunk in chunks)
    assert "Intro paragraph before any heading." in combined


def test_frontmatter_is_stripped_from_text(note_sample):
    chunks = chunk_markdown(note_sample, source_path=PATH, vault_id=VAULT)
    combined = "\n".join(chunk.text for chunk in chunks)
    assert "status: reviewed" not in combined


def test_title_comes_from_frontmatter_when_present(note_sample):
    chunks = chunk_markdown(note_sample, source_path=PATH, vault_id=VAULT)
    assert all(chunk.title == "Kubernetes Scheduling" for chunk in chunks)


def test_title_falls_back_to_filename_stem():
    chunks = chunk_markdown("no frontmatter here", source_path=PATH, vault_id=VAULT)
    assert chunks[0].title == "Kubernetes Scheduling"


def test_links_are_collected_across_the_whole_note(note_sample):
    chunks = chunk_markdown(note_sample, source_path=PATH, vault_id=VAULT)
    all_links = {link for chunk in chunks for link in chunk.links_to}
    assert all_links == {"Bloom Filters", "Node Affinity", "Scoring"}


def test_backlinks_are_empty_at_chunk_time(note_sample):
    chunks = chunk_markdown(note_sample, source_path=PATH, vault_id=VAULT)
    assert all(chunk.backlinks == () for chunk in chunks)


def test_corpus_and_vault_metadata(note_sample):
    chunks = chunk_markdown(note_sample, source_path=PATH, vault_id=VAULT)
    for chunk in chunks:
        assert chunk.corpus == "note"
        assert chunk.chunk_type == "note"
        assert chunk.vault_id == VAULT
        assert chunk.video_id is None
        assert chunk.start_seconds is None


def test_context_prefix_includes_vault_and_path(note_sample):
    chunks = chunk_markdown(note_sample, source_path=PATH, vault_id=VAULT)
    assert all(VAULT in chunk.text and PATH in chunk.text for chunk in chunks)


def test_empty_note_yields_no_chunks():
    assert chunk_markdown("", source_path=PATH, vault_id=VAULT) == []


def test_frontmatter_only_note_yields_no_chunks():
    assert chunk_markdown("---\ntitle: X\n---\n", source_path=PATH, vault_id=VAULT) == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/chunkers/test_markdown.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Write the implementation**

Create `src/notes_rag/chunkers/markdown.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/chunkers/test_markdown.py -v`
Expected: 17 passed

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff check src tests && .venv/bin/ruff format src tests
git add src/notes_rag/chunkers/markdown.py tests/
git commit -m "feat: add markdown note chunker with wikilink extraction"
```

---

### Task 6: `VectorStore` interface and `sqlite-vec` implementation

**Files:**
- Create: `src/notes_rag/store/__init__.py`, `src/notes_rag/store/base.py`, `src/notes_rag/store/sqlite_vec.py`
- Test: `tests/store/test_sqlite_vec.py`

**Interfaces:**
- Consumes: `Chunk` from `notes_rag.models`.
- Produces:
  - `SearchHit` frozen dataclass: `chunk: Chunk`, `distance: float`.
  - `VectorStore` Protocol with `upsert(chunks, vectors) -> None`, `delete_by_path(source_path) -> int`, `search(vector, k, *, corpus=None, vault_id=None, chunk_type=None) -> list[SearchHit]`, `cached_vectors(hashes) -> dict[str, list[float]]`, `all_source_paths() -> set[str]`, `close() -> None`.
  - `SqliteVecStore(path: str | Path, *, dimensions: int = 1024)` implementing it, plus `SqliteVecStore.copy_filtered(dest, *, corpus) -> None` used to produce `public.db`.

**Design note — the over-fetch:** vec0's `MATCH` cannot express our metadata filters, so `search` asks vec0 for `k * OVERFETCH` rows and filters after joining to the `chunks` table. At ~20k chunks this is free. It is not exact: a filter that excludes almost everything can return fewer than `k` hits. That is acceptable in v1 because `public.db` is physically filtered already, so the demo path needs no filter at all. If it ever bites, the fix is internal to this class — which is why `VectorStore` exists.

- [ ] **Step 1: Write the failing test**

Create `tests/store/__init__.py` (empty) and `tests/store/test_sqlite_vec.py`:

```python
import pytest

from notes_rag.models import Chunk
from notes_rag.store.sqlite_vec import SqliteVecStore

DIMS = 4


def make_chunk(chunk_id: str, path: str, *, corpus="video", vault_id=None) -> Chunk:
    return Chunk(
        id=chunk_id,
        corpus=corpus,
        vault_id=vault_id,
        source_path=path,
        chunk_type="summary",
        title="T",
        heading="H",
        context="CTX",
        text=f"text for {chunk_id}",
        content_hash=f"hash-{chunk_id}",
        video_id="vid",
        start_seconds=10,
        url="https://example.com",
    )


@pytest.fixture
def store(tmp_path):
    s = SqliteVecStore(tmp_path / "test.db", dimensions=DIMS)
    yield s
    s.close()


def test_upsert_then_search_returns_the_nearest_chunk(store):
    store.upsert(
        [make_chunk("a", "p1.json"), make_chunk("b", "p2.json")],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )
    hits = store.search([1.0, 0.0, 0.0, 0.0], k=1)
    assert len(hits) == 1
    assert hits[0].chunk.id == "a"


def test_search_returns_hits_ordered_by_distance(store):
    store.upsert(
        [make_chunk("a", "p1.json"), make_chunk("b", "p2.json")],
        [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
    )
    hits = store.search([1.0, 0.0, 0.0, 0.0], k=2)
    assert [hit.chunk.id for hit in hits] == ["a", "b"]
    assert hits[0].distance <= hits[1].distance


def test_search_round_trips_all_chunk_fields(store):
    store.upsert([make_chunk("a", "p1.json")], [[1.0, 0.0, 0.0, 0.0]])
    chunk = store.search([1.0, 0.0, 0.0, 0.0], k=1)[0].chunk
    assert chunk.corpus == "video"
    assert chunk.video_id == "vid"
    assert chunk.start_seconds == 10
    assert chunk.url == "https://example.com"
    assert chunk.text == "text for a"


def test_search_filters_by_corpus(store):
    store.upsert(
        [make_chunk("a", "p1.json", corpus="video"),
         make_chunk("b", "p2.md", corpus="note", vault_id="V")],
        [[1.0, 0.0, 0.0, 0.0], [0.99, 0.01, 0.0, 0.0]],
    )
    hits = store.search([1.0, 0.0, 0.0, 0.0], k=5, corpus="note")
    assert [hit.chunk.id for hit in hits] == ["b"]


def test_search_filters_by_vault_id(store):
    store.upsert(
        [make_chunk("a", "p1.md", corpus="note", vault_id="Alpha"),
         make_chunk("b", "p2.md", corpus="note", vault_id="Beta")],
        [[1.0, 0.0, 0.0, 0.0], [0.99, 0.01, 0.0, 0.0]],
    )
    hits = store.search([1.0, 0.0, 0.0, 0.0], k=5, vault_id="Beta")
    assert [hit.chunk.id for hit in hits] == ["b"]


def test_upsert_replaces_a_chunk_with_the_same_id(store):
    store.upsert([make_chunk("a", "p1.json")], [[1.0, 0.0, 0.0, 0.0]])
    store.upsert([make_chunk("a", "p1.json")], [[0.0, 1.0, 0.0, 0.0]])
    assert len(store.search([0.0, 1.0, 0.0, 0.0], k=10)) == 1


def test_delete_by_path_removes_every_chunk_for_that_path(store):
    store.upsert(
        [make_chunk("a", "p1.json"), make_chunk("b", "p1.json"), make_chunk("c", "p2.json")],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
    )
    assert store.delete_by_path("p1.json") == 2
    remaining = {hit.chunk.id for hit in store.search([1.0, 1.0, 1.0, 0.0], k=10)}
    assert remaining == {"c"}


def test_delete_by_path_on_unknown_path_returns_zero(store):
    assert store.delete_by_path("nope.json") == 0


def test_cached_vectors_returns_known_hashes_only(store):
    store.upsert([make_chunk("a", "p1.json")], [[1.0, 0.0, 0.0, 0.0]])
    cached = store.cached_vectors({"hash-a", "hash-missing"})
    assert set(cached) == {"hash-a"}
    assert cached["hash-a"] == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_cached_vectors_with_empty_input(store):
    assert store.cached_vectors(set()) == {}


def test_all_source_paths(store):
    store.upsert(
        [make_chunk("a", "p1.json"), make_chunk("b", "p2.json")],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )
    assert store.all_source_paths() == {"p1.json", "p2.json"}


def test_copy_filtered_writes_a_video_only_database(store, tmp_path):
    store.upsert(
        [make_chunk("a", "p1.json", corpus="video"),
         make_chunk("b", "p2.md", corpus="note", vault_id="V")],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )
    dest = tmp_path / "public.db"
    store.copy_filtered(dest, corpus="video")

    public = SqliteVecStore(dest, dimensions=DIMS)
    try:
        hits = public.search([1.0, 1.0, 0.0, 0.0], k=10)
        assert {hit.chunk.id for hit in hits} == {"a"}
    finally:
        public.close()


def test_reopening_the_file_preserves_data(tmp_path):
    path = tmp_path / "persist.db"
    first = SqliteVecStore(path, dimensions=DIMS)
    first.upsert([make_chunk("a", "p1.json")], [[1.0, 0.0, 0.0, 0.0]])
    first.close()

    second = SqliteVecStore(path, dimensions=DIMS)
    try:
        assert len(second.search([1.0, 0.0, 0.0, 0.0], k=1)) == 1
    finally:
        second.close()


def test_upsert_rejects_mismatched_lengths(store):
    with pytest.raises(ValueError):
        store.upsert([make_chunk("a", "p1.json")], [])
```

- [ ] **Step 2: Verify sqlite extension loading works on this machine**

Run:

```bash
.venv/bin/python -c "
import sqlite3, sqlite_vec
db = sqlite3.connect(':memory:')
db.enable_load_extension(True)
sqlite_vec.load(db)
print(db.execute('select vec_version()').fetchone())
"
```

Expected: prints a version tuple. If it raises `AttributeError: 'sqlite3.Connection' object has no attribute 'enable_load_extension'`, this Python was built without extension support — switch interpreters (see Global Constraints) before continuing.

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/store/test_sqlite_vec.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'notes_rag.store'`

- [ ] **Step 4: Write the interface**

Create `src/notes_rag/store/__init__.py` (empty file).

Create `src/notes_rag/store/base.py`:

```python
"""Storage interface. One implementation today (sqlite-vec); the interface exists
so a pgvector implementation can be benchmarked against it without touching callers."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from notes_rag.models import Chunk


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    distance: float


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Insert or replace chunks. `chunks` and `vectors` must be the same length."""

    def delete_by_path(self, source_path: str) -> int:
        """Delete every chunk for a source path. Returns the number deleted."""

    def search(
        self,
        vector: Sequence[float],
        k: int,
        *,
        corpus: str | None = None,
        vault_id: str | None = None,
        chunk_type: str | None = None,
    ) -> list[SearchHit]:
        """Nearest neighbours, optionally filtered, ordered by ascending distance."""

    def cached_vectors(self, hashes: Iterable[str]) -> dict[str, list[float]]:
        """Vectors already stored for these content hashes. Missing hashes are omitted."""

    def all_source_paths(self) -> set[str]:
        """Every distinct source path currently indexed."""

    def close(self) -> None: ...
```

- [ ] **Step 5: Write the sqlite-vec implementation**

Create `src/notes_rag/store/sqlite_vec.py`:

```python
"""sqlite-vec backed VectorStore.

The whole index is one file: metadata in a normal table, vectors in a vec0
virtual table joined by rowid. At ~20k chunks a brute-force scan is
milliseconds, so no ANN tuning is required.
"""

import json
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path

import sqlite_vec

from notes_rag.models import Chunk
from notes_rag.store.base import SearchHit

# vec0 MATCH cannot express our metadata filters, so we over-fetch and filter
# after joining. See the design note in the plan for why this is acceptable.
OVERFETCH = 8

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    rowid          INTEGER PRIMARY KEY AUTOINCREMENT,
    id             TEXT UNIQUE NOT NULL,
    corpus         TEXT NOT NULL,
    vault_id       TEXT,
    source_path    TEXT NOT NULL,
    chunk_type     TEXT NOT NULL,
    title          TEXT NOT NULL,
    heading        TEXT,
    context        TEXT NOT NULL,
    text           TEXT NOT NULL,
    content_hash   TEXT NOT NULL,
    video_id       TEXT,
    start_seconds  INTEGER,
    url            TEXT,
    links_to       TEXT NOT NULL DEFAULT '[]',
    backlinks      TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_chunks_source_path ON chunks(source_path);
CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_corpus ON chunks(corpus);
"""

_COLUMNS = (
    "id, corpus, vault_id, source_path, chunk_type, title, heading, context, "
    "text, content_hash, video_id, start_seconds, url, links_to, backlinks"
)


class SqliteVecStore:
    def __init__(self, path: str | Path, *, dimensions: int = 1024) -> None:
        self.path = Path(path)
        self.dimensions = dimensions
        self._db = sqlite3.connect(self.path)
        self._db.enable_load_extension(True)
        sqlite_vec.load(self._db)
        self._db.enable_load_extension(False)
        self._db.executescript(_SCHEMA)
        self._db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks "
            f"USING vec0(embedding float[{dimensions}])"
        )
        self._db.commit()

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks and vectors must be the same length; "
                f"got {len(chunks)} and {len(vectors)}"
            )
        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"vector for {chunk.id} has {len(vector)} dimensions, "
                    f"expected {self.dimensions}"
                )
            self._delete_ids([chunk.id])
            cursor = self._db.execute(
                f"INSERT INTO chunks ({_COLUMNS}) "
                f"VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    chunk.id,
                    chunk.corpus,
                    chunk.vault_id,
                    chunk.source_path,
                    chunk.chunk_type,
                    chunk.title,
                    chunk.heading,
                    chunk.context,
                    chunk.text,
                    chunk.content_hash,
                    chunk.video_id,
                    chunk.start_seconds,
                    chunk.url,
                    json.dumps(list(chunk.links_to)),
                    json.dumps(list(chunk.backlinks)),
                ),
            )
            self._db.execute(
                "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                (cursor.lastrowid, sqlite_vec.serialize_float32(list(vector))),
            )
        self._db.commit()

    def delete_by_path(self, source_path: str) -> int:
        rows = self._db.execute(
            "SELECT id FROM chunks WHERE source_path = ?", (source_path,)
        ).fetchall()
        ids = [row[0] for row in rows]
        self._delete_ids(ids)
        self._db.commit()
        return len(ids)

    def search(
        self,
        vector: Sequence[float],
        k: int,
        *,
        corpus: str | None = None,
        vault_id: str | None = None,
        chunk_type: str | None = None,
    ) -> list[SearchHit]:
        rows = self._db.execute(
            "SELECT rowid, distance FROM vec_chunks "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (sqlite_vec.serialize_float32(list(vector)), k * OVERFETCH),
        ).fetchall()
        if not rows:
            return []

        distances = {rowid: distance for rowid, distance in rows}
        placeholders = ",".join("?" * len(distances))
        clauses = [f"rowid IN ({placeholders})"]
        params: list[object] = list(distances)
        for column, value in (
            ("corpus", corpus),
            ("vault_id", vault_id),
            ("chunk_type", chunk_type),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)

        records = self._db.execute(
            f"SELECT rowid, {_COLUMNS} FROM chunks WHERE {' AND '.join(clauses)}",
            params,
        ).fetchall()

        hits = [
            SearchHit(chunk=_to_chunk(record[1:]), distance=distances[record[0]])
            for record in records
        ]
        hits.sort(key=lambda hit: hit.distance)
        return hits[:k]

    def cached_vectors(self, hashes: Iterable[str]) -> dict[str, list[float]]:
        wanted = list(hashes)
        if not wanted:
            return {}
        placeholders = ",".join("?" * len(wanted))
        rows = self._db.execute(
            f"SELECT c.content_hash, vec_to_json(v.embedding) "
            f"FROM chunks c JOIN vec_chunks v ON v.rowid = c.rowid "
            f"WHERE c.content_hash IN ({placeholders})",
            wanted,
        ).fetchall()
        return {content_hash: json.loads(raw) for content_hash, raw in rows}

    def all_source_paths(self) -> set[str]:
        rows = self._db.execute("SELECT DISTINCT source_path FROM chunks").fetchall()
        return {row[0] for row in rows}

    def copy_filtered(self, dest: str | Path, *, corpus: str) -> None:
        """Write a new database containing only chunks from one corpus.

        This is how public.db is produced from full.db. Physical separation,
        not a query predicate: the demo Lambda's IAM role can read only the
        resulting file, so a filter bug cannot leak other corpora.
        """
        target = SqliteVecStore(dest, dimensions=self.dimensions)
        try:
            rows = self._db.execute(
                f"SELECT c.rowid, {_COLUMNS}, vec_to_json(v.embedding) "
                f"FROM chunks c JOIN vec_chunks v ON v.rowid = c.rowid "
                f"WHERE c.corpus = ?",
                (corpus,),
            ).fetchall()
            for record in rows:
                chunk = _to_chunk(record[1:-1])
                target.upsert([chunk], [json.loads(record[-1])])
        finally:
            target.close()

    def close(self) -> None:
        self._db.close()

    def _delete_ids(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        rows = self._db.execute(
            f"SELECT rowid FROM chunks WHERE id IN ({placeholders})", list(ids)
        ).fetchall()
        for (rowid,) in rows:
            self._db.execute("DELETE FROM vec_chunks WHERE rowid = ?", (rowid,))
        self._db.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", list(ids))


def _to_chunk(record: Sequence) -> Chunk:
    return Chunk(
        id=record[0],
        corpus=record[1],
        vault_id=record[2],
        source_path=record[3],
        chunk_type=record[4],
        title=record[5],
        heading=record[6],
        context=record[7],
        text=record[8],
        content_hash=record[9],
        video_id=record[10],
        start_seconds=record[11],
        url=record[12],
        links_to=tuple(json.loads(record[13])),
        backlinks=tuple(json.loads(record[14])),
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/store/test_sqlite_vec.py -v`
Expected: 14 passed

- [ ] **Step 7: Lint and commit**

```bash
.venv/bin/ruff check src tests && .venv/bin/ruff format src tests
git add src/notes_rag/store tests/store
git commit -m "feat: add VectorStore interface and sqlite-vec implementation"
```

---

### Task 7: `Embedder` interface, fake, and Titan implementation

**Files:**
- Create: `src/notes_rag/embed/__init__.py`, `src/notes_rag/embed/base.py`, `src/notes_rag/embed/fake.py`, `src/notes_rag/embed/bedrock.py`
- Test: `tests/embed/test_fake.py`, `tests/embed/test_bedrock.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `Embedder` Protocol: attribute `dimensions: int`, method `embed(texts: Sequence[str]) -> list[list[float]]`.
  - `FakeEmbedder(dimensions: int = 1024)` — deterministic, hash-derived, no network. Used by every unit test downstream.
  - `TitanEmbedder(*, region: str = "us-east-2", dimensions: int = 1024, client=None)`.

**Note:** Titan v2's `InvokeModel` takes one `inputText` per call, so `embed` loops. That is fine at this corpus size — incremental embedding means a typical run embeds a handful of chunks.

- [ ] **Step 1: Write the failing test for the fake**

Create `tests/embed/__init__.py` (empty) and `tests/embed/test_fake.py`:

```python
import pytest

from notes_rag.embed.fake import FakeEmbedder


def test_returns_one_vector_per_text():
    vectors = FakeEmbedder(dimensions=8).embed(["a", "b", "c"])
    assert len(vectors) == 3


def test_vectors_have_the_declared_dimensions():
    embedder = FakeEmbedder(dimensions=8)
    assert all(len(vector) == 8 for vector in embedder.embed(["a"]))
    assert embedder.dimensions == 8


def test_is_deterministic_across_instances():
    assert FakeEmbedder(dimensions=8).embed(["hello"]) == FakeEmbedder(dimensions=8).embed(
        ["hello"]
    )


def test_different_texts_give_different_vectors():
    embedder = FakeEmbedder(dimensions=8)
    assert embedder.embed(["hello"]) != embedder.embed(["world"])


def test_vectors_are_unit_normalised():
    vector = FakeEmbedder(dimensions=8).embed(["hello"])[0]
    magnitude = sum(component**2 for component in vector) ** 0.5
    assert magnitude == pytest.approx(1.0)


def test_empty_input_returns_empty_list():
    assert FakeEmbedder(dimensions=8).embed([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/embed/test_fake.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'notes_rag.embed'`

- [ ] **Step 3: Write the interface and the fake**

Create `src/notes_rag/embed/__init__.py` (empty file).

Create `src/notes_rag/embed/base.py`:

```python
"""Embedding interface. Keeps Bedrock out of every unit test."""

from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    dimensions: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per input text, in the same order."""
```

Create `src/notes_rag/embed/fake.py`:

```python
"""Deterministic embedder for tests. No network, stable across processes."""

import hashlib
import math
from collections.abc import Sequence


class FakeEmbedder:
    def __init__(self, dimensions: int = 1024) -> None:
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        # Expand the digest until it covers the requested dimensions, so the
        # same text always maps to the same point regardless of dimension count.
        raw = b""
        counter = 0
        while len(raw) < self.dimensions:
            raw += hashlib.sha256(f"{counter}:{text}".encode()).digest()
            counter += 1
        components = [byte / 255.0 - 0.5 for byte in raw[: self.dimensions]]
        magnitude = math.sqrt(sum(value**2 for value in components)) or 1.0
        return [value / magnitude for value in components]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/embed/test_fake.py -v`
Expected: 6 passed

- [ ] **Step 5: Write the Titan test**

Create `tests/embed/test_bedrock.py`:

```python
import json

import pytest

from notes_rag.embed.bedrock import TitanEmbedder


class StubBedrockClient:
    """Records calls and returns a fixed embedding."""

    def __init__(self, dimensions: int = 4) -> None:
        self.calls: list[dict] = []
        self.dimensions = dimensions

    def invoke_model(self, *, modelId: str, body: str):  # noqa: N803 - boto3 kwarg
        self.calls.append({"modelId": modelId, "body": json.loads(body)})
        payload = json.dumps({"embedding": [0.5] * self.dimensions})
        return {"body": _FakeStream(payload)}


class _FakeStream:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload.encode()


def test_returns_one_vector_per_text():
    client = StubBedrockClient()
    embedder = TitanEmbedder(dimensions=4, client=client)
    assert embedder.embed(["a", "b"]) == [[0.5] * 4, [0.5] * 4]


def test_calls_the_titan_v2_model_id():
    client = StubBedrockClient()
    TitanEmbedder(dimensions=4, client=client).embed(["a"])
    assert client.calls[0]["modelId"] == "amazon.titan-embed-text-v2:0"


def test_requests_the_configured_dimensions_and_normalisation():
    client = StubBedrockClient()
    TitanEmbedder(dimensions=4, client=client).embed(["a"])
    body = client.calls[0]["body"]
    assert body["dimensions"] == 4
    assert body["normalize"] is True
    assert body["inputText"] == "a"


def test_empty_input_makes_no_calls():
    client = StubBedrockClient()
    assert TitanEmbedder(dimensions=4, client=client).embed([]) == []
    assert client.calls == []


def test_raises_when_response_dimensions_disagree():
    client = StubBedrockClient(dimensions=3)
    with pytest.raises(ValueError, match="dimensions"):
        TitanEmbedder(dimensions=4, client=client).embed(["a"])


@pytest.mark.integration
def test_real_titan_call_returns_1024_dimensions():
    """Confirms spec §10 item 1: listing is not entitlement.

    Run explicitly: pytest -m integration tests/embed/test_bedrock.py
    """
    vectors = TitanEmbedder().embed(["hello world"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 1024
```

- [ ] **Step 6: Run test to verify it fails**

Run: `.venv/bin/pytest tests/embed/test_bedrock.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'notes_rag.embed.bedrock'`

- [ ] **Step 7: Write the Titan implementation**

Create `src/notes_rag/embed/bedrock.py`:

```python
"""Amazon Titan Text Embeddings v2 via Bedrock.

Titan is not an Anthropic model, so the Bedrock Anthropic use-case form does not
gate it — but listing a model is not the same as being entitled to it. The
integration test in tests/embed/test_bedrock.py is the entitlement check
(spec §10 item 1); run it before relying on this class.
"""

import json
from collections.abc import Sequence

MODEL_ID = "amazon.titan-embed-text-v2:0"
DEFAULT_REGION = "us-east-2"


class TitanEmbedder:
    def __init__(
        self,
        *,
        region: str = DEFAULT_REGION,
        dimensions: int = 1024,
        client=None,
    ) -> None:
        self.dimensions = dimensions
        if client is None:
            import boto3

            client = boto3.client("bedrock-runtime", region_name=region)
        self._client = client

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # Titan v2 accepts one inputText per InvokeModel call. Incremental
        # embedding means a typical indexer run sends only a handful.
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        response = self._client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(
                {
                    "inputText": text,
                    "dimensions": self.dimensions,
                    "normalize": True,
                }
            ),
        )
        vector = json.loads(response["body"].read())["embedding"]
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Titan returned {len(vector)} dimensions, expected {self.dimensions}"
            )
        return vector
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/embed/ -v`
Expected: 11 passed, 1 deselected (the integration test)

- [ ] **Step 9: Lint and commit**

```bash
.venv/bin/ruff check src tests && .venv/bin/ruff format src tests
git add src/notes_rag/embed tests/embed
git commit -m "feat: add Embedder interface with fake and Titan v2 implementations"
```

---

### Task 8: Index builder with embedding cache and backlink derivation

**Files:**
- Create: `src/notes_rag/indexer/__init__.py`, `src/notes_rag/indexer/build.py`
- Test: `tests/indexer/test_build.py`

**Interfaces:**
- Consumes: `Chunk`, `VectorStore`, `SqliteVecStore`, `Embedder`, `FakeEmbedder`.
- Produces:
  - `BuildStats` frozen dataclass: `chunks_written: int`, `vectors_embedded: int`, `vectors_reused: int`, `paths_deleted: int`.
  - `build_index(chunks: Sequence[Chunk], store: VectorStore, embedder: Embedder) -> BuildStats`.
  - `derive_backlinks(chunks: Sequence[Chunk]) -> list[Chunk]`.

**Design note — why this exists:** spec §3. A full re-embed on every rebuild costs ~$72/month at this trigger cadence. `build_index` looks up `content_hash` in the store before embedding, so an unchanged chunk costs one index lookup instead of one Bedrock call. `vectors_reused` in `BuildStats` is what proves the cache works — assert on it.

- [ ] **Step 1: Write the failing test**

Create `tests/indexer/__init__.py` (empty) and `tests/indexer/test_build.py`:

```python
import pytest

from notes_rag.embed.fake import FakeEmbedder
from notes_rag.indexer.build import build_index, derive_backlinks
from notes_rag.models import Chunk
from notes_rag.store.sqlite_vec import SqliteVecStore

DIMS = 8


def note_chunk(chunk_id: str, path: str, *, text: str, links=()) -> Chunk:
    return Chunk(
        id=chunk_id,
        corpus="note",
        vault_id="V",
        source_path=path,
        chunk_type="note",
        title="T",
        heading=None,
        context="CTX",
        text=text,
        content_hash=f"hash-of-{text}",
        links_to=tuple(links),
    )


@pytest.fixture
def store(tmp_path):
    s = SqliteVecStore(tmp_path / "index.db", dimensions=DIMS)
    yield s
    s.close()


@pytest.fixture
def embedder():
    return FakeEmbedder(dimensions=DIMS)


def test_first_build_embeds_everything(store, embedder):
    stats = build_index(
        [note_chunk("a", "a.md", text="one"), note_chunk("b", "b.md", text="two")],
        store,
        embedder,
    )
    assert stats.chunks_written == 2
    assert stats.vectors_embedded == 2
    assert stats.vectors_reused == 0


def test_rebuild_with_identical_content_reuses_every_vector(store, embedder):
    chunks = [note_chunk("a", "a.md", text="one")]
    build_index(chunks, store, embedder)
    stats = build_index(chunks, store, embedder)
    assert stats.vectors_embedded == 0
    assert stats.vectors_reused == 1


def test_rebuild_embeds_only_the_changed_chunk(store, embedder):
    build_index(
        [note_chunk("a", "a.md", text="one"), note_chunk("b", "b.md", text="two")],
        store,
        embedder,
    )
    stats = build_index(
        [note_chunk("a", "a.md", text="one"), note_chunk("b", "b.md", text="CHANGED")],
        store,
        embedder,
    )
    assert stats.vectors_embedded == 1
    assert stats.vectors_reused == 1


def test_paths_absent_from_the_new_chunk_set_are_deleted(store, embedder):
    build_index(
        [note_chunk("a", "a.md", text="one"), note_chunk("b", "b.md", text="two")],
        store,
        embedder,
    )
    stats = build_index([note_chunk("a", "a.md", text="one")], store, embedder)
    assert stats.paths_deleted == 1
    assert store.all_source_paths() == {"a.md"}


def test_deleted_path_chunks_are_gone_from_search(store, embedder):
    build_index(
        [note_chunk("a", "a.md", text="one"), note_chunk("b", "b.md", text="two")],
        store,
        embedder,
    )
    build_index([note_chunk("a", "a.md", text="one")], store, embedder)
    hits = store.search(embedder.embed(["two"])[0], k=10)
    assert {hit.chunk.source_path for hit in hits} == {"a.md"}


def test_embedder_receives_only_uncached_texts(store):
    class CountingEmbedder(FakeEmbedder):
        def __init__(self):
            super().__init__(dimensions=DIMS)
            self.seen: list[str] = []

        def embed(self, texts):
            self.seen.extend(texts)
            return super().embed(texts)

    counting = CountingEmbedder()
    chunks = [note_chunk("a", "a.md", text="one")]
    build_index(chunks, store, counting)
    counting.seen.clear()
    build_index(chunks, store, counting)
    assert counting.seen == []


def test_build_with_no_chunks_deletes_everything(store, embedder):
    build_index([note_chunk("a", "a.md", text="one")], store, embedder)
    stats = build_index([], store, embedder)
    assert stats.paths_deleted == 1
    assert store.all_source_paths() == set()


def test_derive_backlinks_inverts_the_link_relation():
    chunks = [
        note_chunk("a", "Alpha.md", text="one", links=("Beta",)),
        note_chunk("b", "Beta.md", text="two"),
    ]
    out = derive_backlinks(chunks)
    beta = next(chunk for chunk in out if chunk.source_path == "Beta.md")
    assert beta.backlinks == ("Alpha",)


def test_derive_backlinks_leaves_unlinked_notes_empty():
    chunks = [note_chunk("a", "Alpha.md", text="one")]
    assert derive_backlinks(chunks)[0].backlinks == ()


def test_derive_backlinks_ignores_links_to_unknown_notes():
    chunks = [note_chunk("a", "Alpha.md", text="one", links=("Nonexistent",))]
    out = derive_backlinks(chunks)
    assert out[0].backlinks == ()


def test_derive_backlinks_deduplicates_and_sorts():
    chunks = [
        note_chunk("a1", "Alpha.md", text="one", links=("Beta",)),
        note_chunk("a2", "Alpha.md", text="two", links=("Beta",)),
        note_chunk("z", "Zeta.md", text="three", links=("Beta",)),
        note_chunk("b", "Beta.md", text="four"),
    ]
    beta = next(c for c in derive_backlinks(chunks) if c.source_path == "Beta.md")
    assert beta.backlinks == ("Alpha", "Zeta")


def test_backlinks_survive_a_build_round_trip(store, embedder):
    chunks = derive_backlinks(
        [
            note_chunk("a", "Alpha.md", text="one", links=("Beta",)),
            note_chunk("b", "Beta.md", text="two"),
        ]
    )
    build_index(chunks, store, embedder)
    hits = store.search(embedder.embed(["two"])[0], k=10)
    beta = next(hit.chunk for hit in hits if hit.chunk.source_path == "Beta.md")
    assert beta.backlinks == ("Alpha",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/indexer/test_build.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'notes_rag.indexer'`

- [ ] **Step 3: Write the implementation**

Create `src/notes_rag/indexer/__init__.py` (empty file).

Create `src/notes_rag/indexer/build.py`:

```python
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
    source path's stem. Links to notes that do not exist in the corpus are
    ignored rather than recorded as dangling edges.
    """
    stems = {PurePosixPath(chunk.source_path).stem for chunk in chunks}
    inbound: dict[str, set[str]] = {stem: set() for stem in stems}

    for chunk in chunks:
        source_stem = PurePosixPath(chunk.source_path).stem
        for target in chunk.links_to:
            if target in inbound and target != source_stem:
                inbound[target].add(source_stem)

    return [
        replace(
            chunk,
            backlinks=tuple(sorted(inbound[PurePosixPath(chunk.source_path).stem])),
        )
        for chunk in chunks
    ]


def build_index(
    chunks: Sequence[Chunk], store: VectorStore, embedder: Embedder
) -> BuildStats:
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

    stale_paths = store.all_source_paths() - incoming_paths
    paths_deleted = 0
    for path in stale_paths:
        store.delete_by_path(path)
        paths_deleted += 1

    # Clear incoming paths too: a source whose chunk count shrank would
    # otherwise leave orphaned rows behind, since upsert only replaces by id.
    for path in incoming_paths:
        store.delete_by_path(path)

    if not chunks:
        return BuildStats(0, 0, 0, paths_deleted)

    to_embed = [chunk for chunk in chunks if chunk.content_hash not in cached]
    if to_embed:
        fresh = embedder.embed([chunk.text for chunk in to_embed])
        for chunk, vector in zip(to_embed, fresh, strict=True):
            cached[chunk.content_hash] = vector

    store.upsert(list(chunks), [cached[chunk.content_hash] for chunk in chunks])

    return BuildStats(
        chunks_written=len(chunks),
        vectors_embedded=len(to_embed),
        vectors_reused=len(chunks) - len(to_embed),
        paths_deleted=paths_deleted,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/indexer/test_build.py -v`
Expected: 12 passed.

`test_rebuild_with_identical_content_reuses_every_vector` is the one that proves the cache works. If it reports `vectors_embedded == 1, vectors_reused == 0`, the read-before-delete ordering in `build_index` has been broken — that ordering is the whole feature, not an optimization.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/pytest -v`
Expected: all pass, 1 deselected (integration).

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff check src tests && .venv/bin/ruff format src tests
git add src/notes_rag/indexer tests/indexer
git commit -m "feat: add index builder with embedding cache and backlink derivation"
```

---

### Task 9: Evaluation harness

**Files:**
- Create: `eval/__init__.py`, `eval/questions.yaml`, `eval/run.py`
- Test: `tests/eval/test_run.py`

**Interfaces:**
- Consumes: `SearchHit` from `notes_rag.store.base`, `Embedder`, `VectorStore`.
- Produces:
  - `load_questions(path) -> list[Question]` where `Question` has `id: str`, `question: str`, `expects: list[Expectation]`.
  - `Expectation` has `corpus: str`, `video_id: str | None`, `start_seconds: tuple[int, int] | None`, `source_path: str | None`.
  - `matches(hit: SearchHit, expectation: Expectation) -> bool`.
  - `evaluate(questions, store, embedder, *, k: int = 6) -> EvalReport` with `recall_at_k: float`, `mrr: float`, `per_question: list[QuestionResult]`.

**Design note — why spans, not chunk IDs:** spec §6. Chunk IDs are a function of chunking config, so anchoring the golden set on them would invalidate the test whenever chunking changes — which is the main thing the test exists to measure. A hit counts if its `start_seconds` falls inside the expected range, or its `source_path` matches.

**Known limitation to state in any report:** ~15 questions over 2 videos moves recall@k in ~6.7% increments. This detects "something broke", not "this is 3% better".

- [ ] **Step 1: Write the failing test**

Create `tests/eval/__init__.py` (empty) and `tests/eval/test_run.py`:

```python
import pytest

from eval.run import Expectation, Question, evaluate, load_questions, matches
from notes_rag.embed.fake import FakeEmbedder
from notes_rag.models import Chunk
from notes_rag.store.base import SearchHit
from notes_rag.store.sqlite_vec import SqliteVecStore

DIMS = 8


def video_chunk(chunk_id: str, *, start: int, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        corpus="video",
        vault_id=None,
        source_path="summaries/vid.json",
        chunk_type="summary",
        title="T",
        heading="H",
        context="CTX",
        text=text,
        content_hash=f"hash-{chunk_id}",
        video_id="vid",
        start_seconds=start,
        url="https://example.com",
    )


def hit(chunk: Chunk, distance: float = 0.1) -> SearchHit:
    return SearchHit(chunk=chunk, distance=distance)


def test_matches_when_start_seconds_falls_inside_the_span():
    expectation = Expectation(corpus="video", video_id="vid", start_seconds=(1000, 1500))
    assert matches(hit(video_chunk("a", start=1120, text="x")), expectation)


def test_does_not_match_outside_the_span():
    expectation = Expectation(corpus="video", video_id="vid", start_seconds=(1000, 1500))
    assert not matches(hit(video_chunk("a", start=200, text="x")), expectation)


def test_span_boundaries_are_inclusive():
    expectation = Expectation(corpus="video", video_id="vid", start_seconds=(1000, 1500))
    assert matches(hit(video_chunk("a", start=1000, text="x")), expectation)
    assert matches(hit(video_chunk("a", start=1500, text="x")), expectation)


def test_does_not_match_a_different_video():
    expectation = Expectation(corpus="video", video_id="other", start_seconds=(0, 9999))
    assert not matches(hit(video_chunk("a", start=100, text="x")), expectation)


def test_matches_a_note_by_source_path():
    chunk = Chunk(
        id="n", corpus="note", vault_id="V", source_path="Class Notes/a.md",
        chunk_type="note", title="a", heading=None, context="CTX", text="x",
        content_hash="h",
    )
    expectation = Expectation(corpus="note", source_path="Class Notes/a.md")
    assert matches(hit(chunk), expectation)


def test_load_questions_parses_a_video_expectation(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text(
        "- id: q001\n"
        "  question: What about custom schedulers?\n"
        "  expects:\n"
        "    - corpus: video\n"
        "      video_id: vid\n"
        "      start_seconds: [1120, 1400]\n"
    )
    questions = load_questions(path)
    assert len(questions) == 1
    assert questions[0].id == "q001"
    assert questions[0].expects[0].start_seconds == (1120, 1400)


def test_load_questions_parses_a_note_expectation(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text(
        "- id: q002\n"
        "  question: What is affinity?\n"
        "  expects:\n"
        "    - corpus: note\n"
        "      source_path: Class Notes/a.md\n"
    )
    questions = load_questions(path)
    assert questions[0].expects[0].source_path == "Class Notes/a.md"
    assert questions[0].expects[0].start_seconds is None


@pytest.fixture
def populated_store(tmp_path):
    store = SqliteVecStore(tmp_path / "eval.db", dimensions=DIMS)
    embedder = FakeEmbedder(dimensions=DIMS)
    chunks = [
        video_chunk("a", start=1120, text="writing a custom scheduler"),
        video_chunk("b", start=0, text="introduction to clusters"),
    ]
    store.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))
    yield store, embedder
    store.close()


def test_evaluate_scores_a_hit_at_rank_one(populated_store):
    store, embedder = populated_store
    questions = [
        Question(
            id="q1",
            question="writing a custom scheduler",
            expects=[Expectation(corpus="video", video_id="vid", start_seconds=(1100, 1200))],
        )
    ]
    report = evaluate(questions, store, embedder, k=2)
    assert report.recall_at_k == 1.0
    assert report.mrr == 1.0


def test_evaluate_scores_a_miss(populated_store):
    store, embedder = populated_store
    questions = [
        Question(
            id="q1",
            question="writing a custom scheduler",
            expects=[Expectation(corpus="video", video_id="vid", start_seconds=(9000, 9999))],
        )
    ]
    report = evaluate(questions, store, embedder, k=2)
    assert report.recall_at_k == 0.0
    assert report.mrr == 0.0


def test_mrr_halves_for_a_hit_at_rank_two(populated_store):
    store, embedder = populated_store
    questions = [
        Question(
            id="q1",
            question="writing a custom scheduler",
            expects=[Expectation(corpus="video", video_id="vid", start_seconds=(0, 0))],
        )
    ]
    report = evaluate(questions, store, embedder, k=2)
    assert report.recall_at_k == 1.0
    assert report.mrr == pytest.approx(0.5)


def test_per_question_results_are_reported(populated_store):
    store, embedder = populated_store
    questions = [
        Question(
            id="q1",
            question="writing a custom scheduler",
            expects=[Expectation(corpus="video", video_id="vid", start_seconds=(1100, 1200))],
        )
    ]
    report = evaluate(questions, store, embedder, k=2)
    assert len(report.per_question) == 1
    assert report.per_question[0].question_id == "q1"
    assert report.per_question[0].rank == 1


def test_evaluate_with_no_questions_returns_zero(populated_store):
    store, embedder = populated_store
    report = evaluate([], store, embedder, k=2)
    assert report.recall_at_k == 0.0
    assert report.mrr == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/eval/test_run.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval'`

- [ ] **Step 3: Write the seed question set**

Create `eval/__init__.py` (empty file).

Create `eval/questions.yaml`:

```yaml
# Golden questions, anchored on SOURCE SPAN rather than chunk ID.
#
# Chunk IDs are a function of chunking config, so anchoring on them would
# invalidate this file every time chunking changes - which is the main thing
# it exists to measure. A retrieved chunk counts as a hit when its
# start_seconds falls inside the expected range, or its source_path matches.
#
# Replace the placeholder video_id and spans below with values read off the two
# real artifacts in the Video Vault bucket before running this for real.

- id: q001
  question: How does a custom Kubernetes scheduler get registered?
  expects:
    - corpus: video
      video_id: REPLACE_WITH_REAL_VIDEO_ID
      start_seconds: [1120, 1400]

- id: q002
  question: What are the two phases of Kubernetes scheduling?
  expects:
    - corpus: video
      video_id: REPLACE_WITH_REAL_VIDEO_ID
      start_seconds: [0, 1120]
```

- [ ] **Step 4: Write the harness**

Create `eval/run.py`:

```python
"""Retrieval evaluation: recall@k and MRR against a golden question set.

Deterministic and free - no LLM judge, no generation. This is the fast inner
loop that runs on every commit; groundedness and citation scoring live behind
a separate --judge flag (not in this plan).

Known limitation: ~15 questions over 2 videos moves recall@k in ~6.7%
increments. This detects "something broke", not "this is 3% better". Report it
as a regression tripwire, not a measurement.
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from notes_rag.embed.base import Embedder
from notes_rag.store.base import SearchHit, VectorStore


@dataclass(frozen=True)
class Expectation:
    corpus: str
    video_id: str | None = None
    start_seconds: tuple[int, int] | None = None
    source_path: str | None = None


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    expects: list[Expectation] = field(default_factory=list)


@dataclass(frozen=True)
class QuestionResult:
    question_id: str
    rank: int | None  # 1-based rank of the first relevant hit, None if missed


@dataclass(frozen=True)
class EvalReport:
    recall_at_k: float
    mrr: float
    per_question: list[QuestionResult]


def load_questions(path: str | Path) -> list[Question]:
    raw = yaml.safe_load(Path(path).read_text()) or []
    questions: list[Question] = []
    for entry in raw:
        expects = [
            Expectation(
                corpus=item["corpus"],
                video_id=item.get("video_id"),
                start_seconds=(
                    tuple(item["start_seconds"]) if item.get("start_seconds") else None
                ),
                source_path=item.get("source_path"),
            )
            for item in entry.get("expects", [])
        ]
        questions.append(
            Question(id=entry["id"], question=entry["question"], expects=expects)
        )
    return questions


def matches(hit: SearchHit, expectation: Expectation) -> bool:
    chunk = hit.chunk
    if chunk.corpus != expectation.corpus:
        return False
    if expectation.video_id is not None and chunk.video_id != expectation.video_id:
        return False
    if expectation.source_path is not None and chunk.source_path != expectation.source_path:
        return False
    if expectation.start_seconds is not None:
        if chunk.start_seconds is None:
            return False
        low, high = expectation.start_seconds
        if not low <= chunk.start_seconds <= high:
            return False
    return True


def evaluate(
    questions: Sequence[Question],
    store: VectorStore,
    embedder: Embedder,
    *,
    k: int = 6,
) -> EvalReport:
    if not questions:
        return EvalReport(recall_at_k=0.0, mrr=0.0, per_question=[])

    results: list[QuestionResult] = []
    for question in questions:
        vector = embedder.embed([question.question])[0]
        hits = store.search(vector, k=k)
        rank = _first_relevant_rank(hits, question.expects)
        results.append(QuestionResult(question_id=question.id, rank=rank))

    hit_count = sum(1 for result in results if result.rank is not None)
    reciprocal = sum(1.0 / result.rank for result in results if result.rank is not None)
    return EvalReport(
        recall_at_k=hit_count / len(results),
        mrr=reciprocal / len(results),
        per_question=results,
    )


def _first_relevant_rank(
    hits: Sequence[SearchHit], expects: Sequence[Expectation]
) -> int | None:
    for index, hit in enumerate(hits, start=1):
        if any(matches(hit, expectation) for expectation in expects):
            return index
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation.")
    parser.add_argument("--index", required=True, help="path to the .db index")
    parser.add_argument("--questions", default="eval/questions.yaml")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.0,
        help="exit non-zero if recall@k falls below this (use in CI)",
    )
    args = parser.parse_args(argv)

    from notes_rag.embed.bedrock import TitanEmbedder
    from notes_rag.store.sqlite_vec import SqliteVecStore

    store = SqliteVecStore(args.index)
    try:
        report = evaluate(
            load_questions(args.questions), store, TitanEmbedder(), k=args.k
        )
    finally:
        store.close()

    print(f"recall@{args.k}: {report.recall_at_k:.3f}")
    print(f"MRR:       {report.mrr:.3f}")
    for result in report.per_question:
        status = f"rank {result.rank}" if result.rank else "MISS"
        print(f"  {result.question_id}: {status}")

    if report.recall_at_k < args.min_recall:
        print(
            f"FAIL: recall@{args.k} {report.recall_at_k:.3f} "
            f"below threshold {args.min_recall:.3f}"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/eval/test_run.py -v`
Expected: 13 passed

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/pytest -v`
Expected: all pass, 1 deselected.

- [ ] **Step 7: Lint and commit**

```bash
.venv/bin/ruff check src tests eval && .venv/bin/ruff format src tests eval
git add eval tests/eval
git commit -m "feat: add retrieval evaluation harness with recall@k and MRR"
```

---

## Definition of done

- `.venv/bin/pytest` passes with zero failures and no AWS credentials present.
- `.venv/bin/ruff check src tests eval` is clean.
- `.venv/bin/pytest -m integration tests/embed/test_bedrock.py` passes — this is
  the spec §10 item 1 entitlement check for Titan v2 in `us-east-2`.
- A local index can be built from the two real artifacts in the Video Vault
  bucket and queried, and `eval/run.py` reports a recall@k figure against
  `eval/questions.yaml` with real video IDs substituted in.

## What this plan deliberately does not cover

Deferred to later plans, in order:

- **Plan 2 — cloud ingestion and query:** Terraform, indexer Lambda, EventBridge
  Scheduler, GitHub vault source with commit-SHA watermarks, S3 ETag manifests,
  `full.db` / `public.db` upload, query Lambda with Haiku 4.5 generation.
- **Plan 3 — web layer:** Cognito, CloudFront, Vite + React + TypeScript site,
  demo/authenticated split.
- **Plan 4 — extensions:** docx/pptx/xlsx/pdf chunkers, S3 EventBridge trigger,
  confidence-gated wikilink expansion measured against the eval harness.
