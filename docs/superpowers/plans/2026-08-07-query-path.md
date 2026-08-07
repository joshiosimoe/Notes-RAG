# Query Path Implementation Plan (Plan 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a question into a grounded, cited answer from the deployed index, invocable end-to-end with `aws lambda invoke`.

**Architecture:** A Lambda downloads the SQLite index from S3 (re-downloading only when its ETag moved), embeds the question with the existing Titan embedder, searches the vector store, and either refuses — when nothing clears a distance threshold — or asks Haiku 4.5 to answer from numbered context blocks. The model emits context *indices*, never URLs; the handler maps those indices back to citation objects built from stored chunk metadata, so every link is real by construction. A second, near-identical entrypoint serves the public demo over `public.db`; the two differ only in which artifact key they read, and their isolation is enforced by IAM, not by this code.

**Tech Stack:** Python 3.12, `anthropic[bedrock]` (`AnthropicBedrock` client), boto3, sqlite-vec, pytest, Terraform.

## Global Constraints

Every task's requirements implicitly include this section.

- **Python `>=3.12`.** Ruff `line-length = 100`; `ruff check .` must be clean before every commit.
- **Region `us-east-2`, account `207423186995`.**
- **Generation model — decision 29:** client `anthropic.AnthropicBedrock`, model id `us.anthropic.claude-haiku-4-5-20251001-v1:0`. The bare `anthropic.claude-haiku-4-5-20251001-v1:0` is rejected by the API (no in-region availability in `us-east-2`); `AnthropicBedrockMantle` 403s on this account. Do not "simplify" either of these back.
- **Request shape — decision 27:** omit `thinking`, omit `output_config`/`effort`, omit `temperature`/`top_p`/`top_k`. `effort` errors on Haiku 4.5, and omitting `thinking` is how thinking is disabled on this model family.
- **Retrieval — decision 28:** `k=6`, single-turn, non-streaming. k=6 is what the eval baseline (recall@6 1.000, MRR 0.967) was measured at; changing it silently invalidates the only retrieval number the project has.
- **Citations — decision 24:** assembled server-side from context indices. The model never emits a URL or a timestamp. A cited index not present in the context is dropped, the answer is kept, and a warning is logged (§6.4).
- **Refusal is not an error:** no hit clearing the threshold returns HTTP 200 with a refusal string and zero citations (decision 25).
- **Tests touching AWS** are marked `@pytest.mark.integration` and are deselected by default via `addopts = "-m 'not integration'"`. Unit tests must never construct an AWS client.
- **Terraform:** never pipe `terraform` through `head` — a SIGPIPE strands the S3 state lock and `force-unlock` is blocked by the safety classifier. Redirect to a file and read the file.
- **WSL/DrvFS:** every file under `/mnt/c` reports mode `777`, so `ls -l` and `test -x` cannot verify an executable bit. Use `git ls-files -s <path>`; fix with `git update-index --chmod=+x`.

## Spec drift to correct as you go

The spec was written before the code it describes was read. Three names in it are wrong; use the real ones:

| Spec says | Reality |
|---|---|
| "the existing `BedrockEmbedder`" (§6.2 step 2) | The class is `notes_rag.embed.bedrock.TitanEmbedder` |
| Implies an ETag HEAD helper exists (§6.2 step 1) | `sources/s3.py` has only `head_exists`, which returns a bool. Task 1 adds `head_etag`. |
| Implies notes carry a vault-relative path | They do not. `Chunk.source_path` is the full S3 key and `display_path` is never stored. Task 4 derives the vault-relative path from `source_path` + `vault_id`. |

## File Structure

| File | Responsibility |
|---|---|
| `src/notes_rag/sources/s3.py` *(modify)* | Add `head_etag` — HEAD returning the quote-stripped ETag, or `None` when absent. |
| `src/notes_rag/query/__init__.py` *(create)* | Package marker. |
| `src/notes_rag/query/errors.py` *(create)* | `QueryError` taxonomy carrying the HTTP status Plan 5 will map. |
| `src/notes_rag/query/artifact.py` *(create)* | ETag-cached download of the index to `/tmp` (decision 26). |
| `src/notes_rag/query/retrieve.py` *(create)* | Embed the question, search, apply the distance threshold. |
| `src/notes_rag/query/citations.py` *(create)* | Build citation objects from chunks — video deep links and `obsidian://` URLs. |
| `src/notes_rag/query/prompt.py` *(create)* | Prompt assembly and citation-index parsing. Pure; no network. |
| `src/notes_rag/query/generate.py` *(create)* | The Bedrock call and its error classification. |
| `src/notes_rag/query/handler.py` *(create)* | Config, orchestration, the §6.3 response, and the `lambda_handler` entrypoint. |
| `src/notes_rag/demo/__init__.py` *(create)* | Package marker. |
| `src/notes_rag/demo/handler.py` *(create)* | Demo entrypoint — same orchestration, `public.db`, `corpus_scope="public"`. |
| `tests/conftest.py` *(modify)* | `StubS3.head_object` must return an ETag. |
| `scripts/build_lambda.sh` *(modify)* | Bundle `anthropic`. |
| `infra/query.tf` *(create)* | Both Lambdas, their roles, log groups. |
| `infra/iam.tf` *(modify)* | Generation grant — profile ARN plus one foundation-model ARN per destination region. |

Tests mirror the source tree under `tests/query/` and `tests/demo/`.

---

### Task 1: ETag-cached artifact download

Decision 26: the indexer republishes every five minutes, so a warm container that downloaded once serves an arbitrarily stale index for its entire life with no symptom. A ~20ms HEAD per invocation buys correctness on a request path already spending seconds in Bedrock.

**Files:**
- Modify: `src/notes_rag/sources/s3.py` (add `head_etag` after `head_exists`, ~line 92)
- Create: `src/notes_rag/query/__init__.py`
- Create: `src/notes_rag/query/errors.py`
- Create: `src/notes_rag/query/artifact.py`
- Modify: `tests/conftest.py:102-106` (`StubS3.head_object`)
- Test: `tests/query/__init__.py`, `tests/query/test_artifact.py`

**Interfaces:**
- Consumes: `notes_rag.sources.s3.get_bytes` (existing).
- Produces:
  - `s3.head_etag(client, bucket: str, key: str) -> str | None`
  - `errors.QueryError(Exception)` with class attributes `http_status: int` and `retry_after: int | None = None`
  - `errors.ArtifactMissing(QueryError)` — `http_status = 503`, `retry_after = 60`
  - `errors.EmbeddingFailed(QueryError)` — `http_status = 502`
  - `errors.UpstreamThrottled(QueryError)` — `http_status = 503`, `retry_after = 5`
  - `errors.GenerationFailed(QueryError)` — `http_status = 502`
  - `artifact.ArtifactCache(bucket: str, key: str, dest: Path)` with `.ensure_current(s3) -> Path` and `.etag: str | None`

- [ ] **Step 1: Write the failing tests**

Create `tests/query/__init__.py` as an empty file, then `tests/query/test_artifact.py`:

```python
import pytest

from notes_rag.query.artifact import ArtifactCache
from notes_rag.query.errors import ArtifactMissing
from notes_rag.sources.s3 import head_etag


def test_head_etag_returns_quote_free_etag(make_s3):
    s3 = make_s3({"index/full.db": b"payload"})
    etag = head_etag(s3, "bucket", "index/full.db")
    assert etag is not None
    assert '"' not in etag


def test_head_etag_returns_none_for_missing_key(make_s3):
    s3 = make_s3({})
    assert head_etag(s3, "bucket", "index/full.db") is None


def test_first_call_downloads(tmp_path, make_s3):
    s3 = make_s3({"index/full.db": b"payload"})
    cache = ArtifactCache(bucket="bucket", key="index/full.db", dest=tmp_path / "full.db")
    path = cache.ensure_current(s3)
    assert path.read_bytes() == b"payload"


def test_unchanged_etag_skips_the_download(tmp_path, make_s3):
    s3 = make_s3({"index/full.db": b"payload"})
    cache = ArtifactCache(bucket="bucket", key="index/full.db", dest=tmp_path / "full.db")
    cache.ensure_current(s3)

    # Corrupt the local copy. A second ensure_current must NOT repair it:
    # that is the proof the download was skipped rather than merely fast.
    (tmp_path / "full.db").write_bytes(b"local-copy-untouched")
    cache.ensure_current(s3)
    assert (tmp_path / "full.db").read_bytes() == b"local-copy-untouched"


def test_changed_etag_redownloads(tmp_path, make_s3):
    s3 = make_s3({"index/full.db": b"payload"})
    cache = ArtifactCache(bucket="bucket", key="index/full.db", dest=tmp_path / "full.db")
    cache.ensure_current(s3)

    s3.objects["index/full.db"] = b"rebuilt"
    assert cache.ensure_current(s3).read_bytes() == b"rebuilt"


def test_missing_local_file_redownloads_even_when_etag_matches(tmp_path, make_s3):
    """A container can lose /tmp between invocations while the cache object survives."""
    s3 = make_s3({"index/full.db": b"payload"})
    cache = ArtifactCache(bucket="bucket", key="index/full.db", dest=tmp_path / "full.db")
    cache.ensure_current(s3)

    (tmp_path / "full.db").unlink()
    assert cache.ensure_current(s3).read_bytes() == b"payload"


def test_missing_artifact_raises_artifact_missing(tmp_path, make_s3):
    s3 = make_s3({})
    cache = ArtifactCache(bucket="bucket", key="index/full.db", dest=tmp_path / "full.db")
    with pytest.raises(ArtifactMissing) as caught:
        cache.ensure_current(s3)
    assert caught.value.http_status == 503
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/mnt/c/Users/joshi/Notes Rag" && uv run pytest tests/query/test_artifact.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'notes_rag.query'`

- [ ] **Step 3: Make `StubS3.head_object` return an ETag**

The stub's `head_object` currently returns a bare `{}`, so `head_etag` could never see one. Replace `tests/conftest.py:102-106` with:

```python
    def head_object(self, **kwargs):
        key = kwargs["Key"]
        if key not in self.objects:
            raise _ClientError("404")
        # Same body-derived digest list_objects_v2 returns, so a HEAD and a
        # LIST of the same object agree - which is what makes an ETag usable
        # as a change detector rather than just a non-empty string.
        return {"ETag": f'"{hashlib.sha256(self.objects[key]).hexdigest()}"'}
```

- [ ] **Step 4: Add `head_etag` to `sources/s3.py`**

Insert after `head_exists` (which ends at line 91):

```python
def head_etag(client, bucket: str, key: str) -> str | None:
    """`key`'s quote-free ETag, or None when the key does not exist.

    The bool-returning `head_exists` above answers a different question and is
    kept: the indexer only needs to know whether an artifact is present, while
    the query path needs to know whether it *changed*. Same HTTP call, and the
    same bare-404 handling - a missing artifact is an expected state here too,
    because the query path can outrace a rebuild.
    """
    try:
        response = client.head_object(Bucket=bucket, Key=key)
    except client.exceptions.ClientError as error:
        if error.response.get("Error", {}).get("Code") == "404":
            return None
        raise
    return _strip_quotes(response["ETag"])
```

- [ ] **Step 5: Write `src/notes_rag/query/errors.py`**

```python
"""Failure taxonomy for the query path (spec §6.4).

Each error carries the HTTP status it should become. Plan 4 has no HTTP
surface - `lambda_handler` renders these into its own return value - but Plan
5's API Gateway adapter maps the same attribute, so the status lives with the
error rather than in a translation table that has to be kept in sync.

A refusal is deliberately absent: no hit clearing the distance threshold is a
successful 200 with zero citations (decision 25), not a failure.
"""


class QueryError(Exception):
    """Base for every failure with a defined HTTP rendering."""

    http_status: int = 500
    retry_after: int | None = None


class ArtifactMissing(QueryError):
    """The index is absent from S3. The indexer restores it within one tick."""

    http_status = 503
    retry_after = 60


class EmbeddingFailed(QueryError):
    """Titan could not embed the question, so it was never searchable."""

    http_status = 502


class UpstreamThrottled(QueryError):
    """Bedrock throttled us. Distinct from GenerationFailed: retrying works."""

    http_status = 503
    retry_after = 5


class GenerationFailed(QueryError):
    """Bedrock failed in a way retrying will not fix."""

    http_status = 502
```

- [ ] **Step 6: Write `src/notes_rag/query/artifact.py`**

```python
"""ETag-gated download of the index artifact (decision 26).

A Lambda execution environment is reused across invocations, so an instance
that downloaded the index once would serve it for its entire life. The indexer
republishes every five minutes, and a stale index has no symptom - it just
quietly answers from an old corpus. A HEAD costs ~20ms against a request path
already spending seconds in Bedrock, so it runs every invocation.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from notes_rag.query.errors import ArtifactMissing
from notes_rag.sources.s3 import get_bytes, head_etag

logger = logging.getLogger(__name__)


@dataclass
class ArtifactCache:
    """One S3 object mirrored to a local path, refreshed when its ETag moves.

    Mutable by design: `etag` is the memory that survives between invocations
    on a warm container, and is the whole point of the class.
    """

    bucket: str
    key: str
    dest: Path
    etag: str | None = field(default=None)

    def ensure_current(self, s3) -> Path:
        """Return a local path holding the current artifact, downloading if needed."""
        remote = head_etag(s3, self.bucket, self.key)
        if remote is None:
            raise ArtifactMissing(f"{self.bucket}/{self.key} is not in S3")

        # The ETag check alone is not enough. /tmp is not guaranteed to
        # outlive an invocation even when the Python process does, so a
        # matching ETag with no local file must still download.
        if remote == self.etag and self.dest.exists():
            return self.dest

        logger.info("downloading %s/%s (etag %s)", self.bucket, self.key, remote)
        raw = get_bytes(s3, self.bucket, self.key)
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        self.dest.write_bytes(raw)
        # Set last: an exception above must leave `etag` describing whatever
        # is actually on disk, not what we hoped to put there.
        self.etag = remote
        return self.dest
```

Create `src/notes_rag/query/__init__.py` as an empty file.

- [ ] **Step 7: Run the tests**

Run: `cd "/mnt/c/Users/joshi/Notes Rag" && uv run pytest tests/query/test_artifact.py -v`
Expected: PASS (7 tests)

- [ ] **Step 8: Run the full suite and the linter**

Run: `cd "/mnt/c/Users/joshi/Notes Rag" && uv run pytest -q && uv run ruff check .`
Expected: all pre-existing tests still pass (the conftest change touches every S3 test), ruff clean.

- [ ] **Step 9: Commit**

```bash
cd "/mnt/c/Users/joshi/Notes Rag"
git add src/notes_rag/query tests/query src/notes_rag/sources/s3.py tests/conftest.py
git commit -m "feat: cache the index artifact against its S3 ETag

The query Lambda reuses its execution environment, so an instance that
downloaded the index once would serve it until the container died - and
because the indexer republishes every five minutes, a stale index answers
from an old corpus with no symptom at all.

head_etag joins head_exists rather than replacing it: the indexer asks
whether an artifact is present, the query path asks whether it changed.

The cache checks the local file's existence as well as the ETag, because
/tmp is not guaranteed to outlive an invocation even when the process is.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Retrieval with a distance threshold

Decision 25: an out-of-corpus question has no grounded answer available, so generating one invites a hallucination and pays for the privilege. Refusing before Bedrock is both the cheapest and the most accurate response.

**Files:**
- Create: `src/notes_rag/query/retrieve.py`
- Test: `tests/query/test_retrieve.py`

**Interfaces:**
- Consumes: `store.base.VectorStore.search`, `store.base.SearchHit`, `embed.base.Embedder`, `errors.EmbeddingFailed`.
- Produces:
  - `retrieve.DEFAULT_DISTANCE_THRESHOLD: float` (placeholder `1.0` here; Task 3 replaces it with a measured value)
  - `retrieve.Retrieval(hits: list[SearchHit], refused: bool)` — frozen dataclass
  - `retrieve.retrieve(question: str, store, embedder, *, k: int = 6, threshold: float = DEFAULT_DISTANCE_THRESHOLD) -> Retrieval`

- [ ] **Step 1: Write the failing tests**

Create `tests/query/test_retrieve.py`:

```python
import pytest

from notes_rag.models import Chunk
from notes_rag.query.errors import EmbeddingFailed
from notes_rag.query.retrieve import Retrieval, retrieve
from notes_rag.store.base import SearchHit


def make_chunk(chunk_id: str = "c1") -> Chunk:
    return Chunk(
        id=chunk_id,
        corpus="video",
        vault_id=None,
        source_path="summaries/v.json",
        chunk_type="summary",
        title="T",
        heading=None,
        context="ctx",
        text="body",
        content_hash="h",
    )


class StubStore:
    """Returns a fixed hit list; records the k it was called with."""

    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.calls: list[int] = []

    def search(self, vector, k, **kwargs):
        self.calls.append(k)
        return self.hits[:k]


class StubEmbedder:
    dimensions = 8

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.texts: list[str] = []

    def embed(self, texts):
        if self.error is not None:
            raise self.error
        self.texts.extend(texts)
        return [[0.0] * self.dimensions for _ in texts]


def test_returns_hits_that_clear_the_threshold():
    store = StubStore([SearchHit(chunk=make_chunk(), distance=0.4)])
    result = retrieve("q", store, StubEmbedder(), threshold=1.0)
    assert result.refused is False
    assert [hit.chunk.id for hit in result.hits] == ["c1"]


def test_refuses_when_the_best_hit_is_worse_than_the_threshold():
    store = StubStore([SearchHit(chunk=make_chunk(), distance=1.4)])
    result = retrieve("q", store, StubEmbedder(), threshold=1.0)
    assert result == Retrieval(hits=[], refused=True)


def test_threshold_is_inclusive_at_the_boundary():
    """A hit exactly at the threshold is kept. Documented so a later `<` vs
    `<=` change is a deliberate decision rather than an accident."""
    store = StubStore([SearchHit(chunk=make_chunk(), distance=1.0)])
    assert retrieve("q", store, StubEmbedder(), threshold=1.0).refused is False


def test_drops_individual_hits_past_the_threshold():
    """The best hit clearing the bar does not license passing the whole list
    to the model - a weak chunk in context is a hallucination invitation."""
    store = StubStore(
        [
            SearchHit(chunk=make_chunk("good"), distance=0.3),
            SearchHit(chunk=make_chunk("weak"), distance=1.8),
        ]
    )
    result = retrieve("q", store, StubEmbedder(), threshold=1.0)
    assert [hit.chunk.id for hit in result.hits] == ["good"]


def test_empty_index_refuses_rather_than_raising():
    assert retrieve("q", StubStore([]), StubEmbedder(), threshold=1.0).refused is True


def test_searches_with_k_six_by_default():
    store = StubStore([])
    retrieve("q", store, StubEmbedder())
    assert store.calls == [6]


def test_embeds_the_question_verbatim():
    embedder = StubEmbedder()
    retrieve("what is a monad?", StubStore([]), embedder)
    assert embedder.texts == ["what is a monad?"]


def test_embedding_failure_becomes_embedding_failed():
    embedder = StubEmbedder(error=RuntimeError("bedrock down"))
    with pytest.raises(EmbeddingFailed) as caught:
        retrieve("q", StubStore([]), embedder)
    assert caught.value.http_status == 502
    assert isinstance(caught.value.__cause__, RuntimeError)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/mnt/c/Users/joshi/Notes Rag" && uv run pytest tests/query/test_retrieve.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'notes_rag.query.retrieve'`

- [ ] **Step 3: Write `src/notes_rag/query/retrieve.py`**

```python
"""Question -> vectors -> hits, with the refusal gate in front of Bedrock.

The threshold is the only thing standing between an out-of-corpus question and
a confidently wrong answer, so it is applied here rather than left to the
prompt: a model given six irrelevant chunks and asked to be careful will still
often produce something. Refusing costs nothing and cannot hallucinate.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from notes_rag.embed.base import Embedder
from notes_rag.query.errors import EmbeddingFailed
from notes_rag.store.base import SearchHit, VectorStore

logger = logging.getLogger(__name__)

# Placeholder until Task 3 measures it against the golden set. sqlite-vec's
# vec0 uses L2, and Titan v2 returns normalized vectors, so distance lies in
# [0, 2] and relates to cosine similarity as d = sqrt(2 * (1 - cos)).
DEFAULT_DISTANCE_THRESHOLD = 1.0

DEFAULT_K = 6


@dataclass(frozen=True)
class Retrieval:
    hits: list[SearchHit]
    refused: bool


def retrieve(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    *,
    k: int = DEFAULT_K,
    threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> Retrieval:
    """Embed, search, and drop everything past `threshold`.

    Weak hits are dropped individually rather than the list being taken
    wholesale once its best member passes. A question that matches one chunk
    well and five poorly should be answered from the one - the other five add
    only tokens and opportunities to go wrong.
    """
    try:
        vector = embedder.embed([question])[0]
    except Exception as error:
        # The question was never searchable, which is a different failure from
        # a generation problem and gets its own status (§6.4).
        raise EmbeddingFailed("could not embed the question") from error

    hits: Sequence[SearchHit] = store.search(vector, k=k)
    kept = [hit for hit in hits if hit.distance <= threshold]

    if not kept:
        best = f"{hits[0].distance:.3f}" if hits else "no hits"
        logger.info("refusing: best distance %s is past threshold %.3f", best, threshold)
        return Retrieval(hits=[], refused=True)

    return Retrieval(hits=list(kept), refused=False)
```

- [ ] **Step 4: Run the tests**

Run: `cd "/mnt/c/Users/joshi/Notes Rag" && uv run pytest tests/query/test_retrieve.py -v && uv run ruff check .`
Expected: PASS (8 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
cd "/mnt/c/Users/joshi/Notes Rag"
git add src/notes_rag/query/retrieve.py tests/query/test_retrieve.py
git commit -m "feat: retrieve with a distance threshold that refuses before Bedrock

An out-of-corpus question has no grounded answer available, so generating
one invites a hallucination and pays for the privilege. The gate sits in
front of the model rather than inside the prompt because a model handed six
irrelevant chunks and told to be careful will still often answer.

Weak hits are dropped individually rather than the list being taken whole
once its best member passes: a question matching one chunk well and five
poorly should be answered from the one.

The threshold value here is a placeholder - it is measured next.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Measure the distance threshold

Spec §6.2 step 4 is explicit that this number is measured, not guessed: too tight refuses answerable questions, too loose defeats decision 25. This task produces the number and the evidence for it.

**Files:**
- Create: `scripts/measure_threshold.py`
- Modify: `src/notes_rag/query/retrieve.py` (the `DEFAULT_DISTANCE_THRESHOLD` constant)
- Modify: `eval/questions.yaml` (append the out-of-corpus question block)

**Interfaces:**
- Consumes: `eval.run.load_questions`, `notes_rag.store.sqlite_vec.SqliteVecStore`, `notes_rag.embed.bedrock.TitanEmbedder`.
- Produces: a measured `DEFAULT_DISTANCE_THRESHOLD` and a recorded rationale.

- [ ] **Step 1: Add out-of-corpus questions to the golden set**

The golden set has 15 in-corpus questions and no negatives, so it can measure recall but not the refusal boundary. Append to `eval/questions.yaml`:

```yaml
# --- Out-of-corpus negatives (added 2026-08-07, Plan 4 Task 3) ---
#
# These have NO `expects` block, deliberately. eval/run.py scores a question
# by whether a hit matches an expectation, so a question with none can never
# count as a hit and would drag recall@k to 0 if it were scored - which is why
# `evaluate` must keep ignoring questions with empty `expects` (it already
# does: `_first_relevant_rank` returns None over an empty `expects`, and that
# is a MISS). They exist for scripts/measure_threshold.py, which reads the
# raw distances rather than the ranks.
#
# Chosen to be plausible study questions about subjects the corpus does not
# cover - not gibberish. A threshold that only rejects keyboard-mashing is
# not a threshold.
- id: oo-cooking
  question: What temperature should I sous-vide a ribeye to get medium rare?
- id: oo-tax
  question: How do I file a self-assessment tax return in the UK?
- id: oo-medical
  question: What are the first-line treatments for type 2 diabetes?
- id: oo-sports
  question: Who won the 2018 FIFA World Cup final and what was the score?
- id: oo-gardening
  question: When should I prune wisteria in a temperate climate?
```

- [ ] **Step 2: Write the measurement script**

Create `scripts/measure_threshold.py`:

```python
"""Measure the retrieval distance distribution to choose a refusal threshold.

Prints the best-hit distance for every golden question, split into questions
the corpus should answer and questions it should not. The threshold belongs in
the gap between the two distributions; this script's job is to show whether a
gap exists at all.

Usage:
    uv run python scripts/measure_threshold.py --index build/full.db
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.run import load_questions  # noqa: E402
from notes_rag.embed.bedrock import TitanEmbedder  # noqa: E402
from notes_rag.store.sqlite_vec import SqliteVecStore  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--questions", default="eval/questions.yaml")
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args(argv)

    store = SqliteVecStore(args.index)
    embedder = TitanEmbedder()
    try:
        in_corpus: list[tuple[str, float]] = []
        out_corpus: list[tuple[str, float]] = []
        for question in load_questions(args.questions):
            hits = store.search(embedder.embed([question.question])[0], k=args.k)
            if not hits:
                print(f"  {question.id}: NO HITS")
                continue
            bucket = in_corpus if question.expects else out_corpus
            bucket.append((question.id, hits[0].distance))
    finally:
        store.close()

    for label, rows in (("IN-CORPUS", in_corpus), ("OUT-OF-CORPUS", out_corpus)):
        print(f"\n{label} (n={len(rows)})")
        for qid, distance in sorted(rows, key=lambda row: row[1]):
            print(f"  {distance:.4f}  {qid}")

    if not in_corpus or not out_corpus:
        print("\nNeed both populations to recommend a threshold.")
        return 1

    worst_good = max(distance for _, distance in in_corpus)
    best_bad = min(distance for _, distance in out_corpus)
    print(f"\nworst in-corpus:  {worst_good:.4f}")
    print(f"best out-of-corpus: {best_bad:.4f}")

    if worst_good >= best_bad:
        print("\nOVERLAP - no threshold separates these. Do not guess one; "
              "investigate the overlapping questions before setting a value.")
        return 1

    print(f"suggested threshold (midpoint): {(worst_good + best_bad) / 2:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Build a local index to measure against**

If `build/full.db` does not already exist, download the deployed one:

```bash
cd "/mnt/c/Users/joshi/Notes Rag"
aws s3 cp "s3://$(cd infra && terraform output -raw index_bucket)/index/full.db" build/full.db
```

If `terraform output` has no such output, get the bucket from `aws s3 ls | grep notes-rag`.

- [ ] **Step 4: Run the measurement**

Run: `cd "/mnt/c/Users/joshi/Notes Rag" && uv run python scripts/measure_threshold.py --index build/full.db`

This makes 20 real Titan calls (~$0.00002). Expected: two separated populations and a suggested midpoint. **If it reports OVERLAP, stop and report that** — do not pick a number anyway. An overlap means either the negatives are not actually out-of-corpus or retrieval is not discriminating, and both need a human decision.

- [ ] **Step 5: Set the measured threshold**

In `src/notes_rag/query/retrieve.py`, replace the placeholder constant with the measured value, and replace the comment with the evidence. Use the **suggested midpoint rounded to two decimals**, and record the actual observed numbers:

```python
# Measured 2026-08-07 against build/full.db (see scripts/measure_threshold.py).
# vec0 uses L2 and Titan v2 returns normalized vectors, so distance lies in
# [0, 2] and relates to cosine similarity as d = sqrt(2 * (1 - cos)).
#
#   worst in-corpus best hit:    <FILL IN from the run>
#   best out-of-corpus best hit: <FILL IN from the run>
#
# Set at the midpoint of that gap. Re-measure when the corpus grows: this is a
# 20-question sample over a small corpus, so treat it as a starting point with
# evidence rather than a tuned constant.
DEFAULT_DISTANCE_THRESHOLD = <FILL IN>
```

- [ ] **Step 6: Confirm the existing eval still passes with the negatives added**

The negatives have no `expects`, so they score as MISS and drag `recall@k` down. Confirm the number and record it — this is expected, not a regression:

Run: `cd "/mnt/c/Users/joshi/Notes Rag" && uv run python eval/run.py --index build/full.db`
Expected: `recall@6` is now `15/20 = 0.750`, and every `oo-*` question shows MISS while all 15 original questions still show a rank.

If any originally-passing question now MISSes, that *is* a regression — stop and report it.

- [ ] **Step 7: Note the recall change in questions.yaml**

Append to the baseline comment block at the top of `eval/questions.yaml`:

```
# Baseline 2026-08-07 (Plan 4 Task 3): five out-of-corpus negatives added for
# threshold calibration. They carry no `expects` and therefore always score as
# MISS, so headline recall@6 drops from 1.000 to 0.750 (15/20) with no change
# in retrieval quality. Compare in-corpus questions only, or use --min-recall
# 0.75 in CI.
```

- [ ] **Step 8: Verify and commit**

Run: `cd "/mnt/c/Users/joshi/Notes Rag" && uv run pytest -q && uv run ruff check .`
Expected: PASS, ruff clean. `tests/query/test_retrieve.py` passes its own explicit thresholds, so the constant change cannot break it.

```bash
cd "/mnt/c/Users/joshi/Notes Rag"
git add scripts/measure_threshold.py eval/questions.yaml src/notes_rag/query/retrieve.py
git commit -m "feat: measure the refusal threshold instead of guessing it

The spec is explicit that this number is measured: set too tight it refuses
answerable questions, set too loose it defeats the refusal decision entirely.

The golden set had only positives, so it could measure recall but not the
refusal boundary. Five out-of-corpus negatives make the second population
exist. They are plausible study questions about uncovered subjects rather
than gibberish - a threshold that only rejects keyboard-mashing is not a
threshold.

The negatives carry no expects and therefore always score MISS, so headline
recall@6 moves 1.000 -> 0.750 with no change in retrieval quality. Recorded
at the top of questions.yaml so the next reader does not chase it.

The script refuses to suggest a value when the two distributions overlap,
rather than splitting a difference that has no gap in it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Citation building

Decision 24: a fabricated deep link is worse than no link — it looks authoritative and fails silently. Every emitted link is built here from stored chunk metadata, so it is real by construction.

**Files:**
- Create: `src/notes_rag/query/citations.py`
- Test: `tests/query/test_citations.py`

**Interfaces:**
- Consumes: `notes_rag.models.Chunk`.
- Produces:
  - `citations.Citation` — frozen dataclass: `kind: str`, `title: str`, `url: str | None`, `source_path: str`, `chunk_id: str`, `start_seconds: int | None = None`
  - `citations.Citation.to_dict() -> dict` — omits `start_seconds` when `None` (spec §6.3 shows it on video citations only)
  - `citations.build_citation(chunk: Chunk) -> Citation`
  - `citations.vault_relative_path(source_path: str, vault_id: str) -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/query/test_citations.py`:

```python
from notes_rag.models import Chunk
from notes_rag.query.citations import Citation, build_citation, vault_relative_path


def video_chunk(**overrides) -> Chunk:
    base = dict(
        id="video:summaries/x.json#2",
        corpus="video",
        vault_id=None,
        source_path="summaries/x.json",
        chunk_type="summary",
        title="Compilers, Part 3",
        heading="Register allocation",
        context="ctx",
        text="body",
        content_hash="h",
        video_id="abc123",
        start_seconds=1120,
        url="https://www.youtube.com/watch?v=abc123",
    )
    base.update(overrides)
    return Chunk(**base)


def note_chunk(**overrides) -> Chunk:
    base = dict(
        id="note:notes/joshiosimoe/Algorithms/Dijkstra.md#0",
        corpus="note",
        vault_id="joshiosimoe",
        source_path="notes/joshiosimoe/Algorithms/Dijkstra.md",
        chunk_type="note",
        title="Dijkstra",
        heading=None,
        context="ctx",
        text="body",
        content_hash="h",
    )
    base.update(overrides)
    return Chunk(**base)


def test_video_citation_appends_the_timestamp():
    citation = build_citation(video_chunk())
    assert citation.kind == "video"
    assert citation.url == "https://www.youtube.com/watch?v=abc123&t=1120"
    assert citation.start_seconds == 1120


def test_video_url_without_a_query_string_uses_a_question_mark():
    chunk = video_chunk(url="https://example.com/watch")
    assert build_citation(chunk).url == "https://example.com/watch?t=1120"


def test_video_at_zero_seconds_still_gets_a_timestamp():
    """t=0 is a real, meaningful deep link. A falsy check here would drop it."""
    chunk = video_chunk(start_seconds=0)
    assert build_citation(chunk).url.endswith("&t=0")


def test_video_without_a_start_time_links_to_the_video():
    chunk = video_chunk(start_seconds=None)
    citation = build_citation(chunk)
    assert citation.url == "https://www.youtube.com/watch?v=abc123"
    assert citation.start_seconds is None


def test_video_without_a_url_yields_no_url_rather_than_a_broken_one():
    citation = build_citation(video_chunk(url=None, start_seconds=90))
    assert citation.url is None


def test_note_citation_builds_an_obsidian_uri():
    citation = build_citation(note_chunk())
    assert citation.kind == "note"
    assert citation.url == (
        "obsidian://open?vault=joshiosimoe&file=Algorithms%2FDijkstra"
    )
    assert citation.start_seconds is None


def test_note_uri_percent_encodes_spaces_and_punctuation():
    chunk = note_chunk(
        vault_id="my vault",
        source_path="notes/my vault/Week 1 & 2.md",
    )
    citation = build_citation(chunk)
    assert "vault=my%20vault" in citation.url
    assert "file=Week%201%20%26%202" in citation.url


def test_note_without_a_vault_id_yields_no_url():
    """An obsidian:// URI is meaningless without a vault name. No link beats
    a link that opens the wrong vault or nothing at all."""
    assert build_citation(note_chunk(vault_id=None)).url is None


def test_vault_relative_path_strips_the_sync_prefix_and_extension():
    assert (
        vault_relative_path("notes/joshiosimoe/Algorithms/Dijkstra.md", "joshiosimoe")
        == "Algorithms/Dijkstra"
    )


def test_vault_relative_path_uses_the_last_matching_segment():
    """A vault named after a folder that also appears earlier in the key must
    not truncate at the first occurrence."""
    assert (
        vault_relative_path("notes/notes/Sub/File.md", "notes") == "Sub/File"
    )


def test_vault_relative_path_falls_back_to_the_stem_when_the_prefix_is_absent():
    assert vault_relative_path("Dijkstra.md", "joshiosimoe") == "Dijkstra"


def test_to_dict_omits_start_seconds_for_notes():
    payload = build_citation(note_chunk()).to_dict()
    assert "start_seconds" not in payload
    assert payload["kind"] == "note"


def test_to_dict_includes_start_seconds_for_videos():
    assert build_citation(video_chunk()).to_dict()["start_seconds"] == 1120


def test_citation_carries_the_chunk_id_for_debugging():
    assert build_citation(video_chunk()).chunk_id == "video:summaries/x.json#2"


def test_note_kind_falls_back_to_the_corpus_name():
    """A future corpus (`material`) must not silently render as a note."""
    citation = build_citation(note_chunk(corpus="material", vault_id=None))
    assert citation.kind == "material"
    assert isinstance(citation, Citation)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/mnt/c/Users/joshi/Notes Rag" && uv run pytest tests/query/test_citations.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'notes_rag.query.citations'`

- [ ] **Step 3: Write `src/notes_rag/query/citations.py`**

```python
"""Turn a retrieved chunk into a citation the frontend can link to.

Decision 24: the model never emits a URL or a timestamp. It cites the index of
a context block, and everything a user can click is constructed here from
metadata the indexer stored. A fabricated deep link is worse than no link -
it looks authoritative and fails silently - so the only way to make that
failure impossible is to leave the model no opportunity to produce one.

Every constructor here prefers `url=None` over a guess. A citation with no
link still names its source and is honest; a wrong link is not.
"""

from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import quote

from notes_rag.models import Chunk


@dataclass(frozen=True)
class Citation:
    kind: str
    title: str
    url: str | None
    source_path: str
    chunk_id: str
    start_seconds: int | None = None

    def to_dict(self) -> dict:
        payload = {
            "kind": self.kind,
            "title": self.title,
            "url": self.url,
            "source_path": self.source_path,
            "chunk_id": self.chunk_id,
        }
        # Spec §6.3 carries start_seconds on video citations only. Emitting a
        # null for notes would invite the frontend to render "at 0:00".
        if self.start_seconds is not None:
            payload["start_seconds"] = self.start_seconds
        return payload


def vault_relative_path(source_path: str, vault_id: str) -> str:
    """`source_path` as Obsidian knows the file: no sync prefix, no extension.

    The indexer stores the full S3 key (`notes/<vault_id>/Some/Note.md`)
    because that is the index identity, and never stores the vault-relative
    display path - so it is reconstructed here from the two fields that are
    stored. Splitting on the vault segment rather than a hardcoded `notes/`
    keeps this working if the sync prefix ever changes.

    rsplit, not split: a vault named after a folder that also appears earlier
    in the key (`notes/notes/Sub/File.md`) must truncate at the vault, not at
    the first coincidence.
    """
    marker = f"/{vault_id}/"
    _, separator, tail = source_path.rpartition(marker)
    relative = tail if separator else source_path
    return str(PurePosixPath(relative).with_suffix(""))


def _video_url(chunk: Chunk) -> str | None:
    if not chunk.url:
        return None
    if chunk.start_seconds is None:
        return chunk.url
    # `is None` above rather than a truthiness check: t=0 is a real deep link
    # to the start of the video, not a missing value.
    joiner = "&" if "?" in chunk.url else "?"
    return f"{chunk.url}{joiner}t={chunk.start_seconds}"


def _note_url(chunk: Chunk) -> str | None:
    if not chunk.vault_id:
        return None
    relative = vault_relative_path(chunk.source_path, chunk.vault_id)
    # `safe=""` so that path separators are encoded too: Obsidian's `file`
    # parameter is a single opaque value, and an unencoded `/` in a note name
    # would be indistinguishable from a folder boundary.
    return (
        f"obsidian://open?vault={quote(chunk.vault_id, safe='')}"
        f"&file={quote(relative, safe='')}"
    )


def build_citation(chunk: Chunk) -> Citation:
    """One citation for one chunk. `kind` is the corpus name, so a corpus
    added later renders as itself rather than being mislabelled a note."""
    if chunk.corpus == "video":
        return Citation(
            kind="video",
            title=chunk.title,
            url=_video_url(chunk),
            source_path=chunk.source_path,
            chunk_id=chunk.id,
            start_seconds=chunk.start_seconds,
        )
    return Citation(
        kind=chunk.corpus,
        title=chunk.title,
        url=_note_url(chunk),
        source_path=chunk.source_path,
        chunk_id=chunk.id,
    )
```

- [ ] **Step 4: Run the tests**

Run: `cd "/mnt/c/Users/joshi/Notes Rag" && uv run pytest tests/query/test_citations.py -v && uv run ruff check .`
Expected: PASS (14 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
cd "/mnt/c/Users/joshi/Notes Rag"
git add src/notes_rag/query/citations.py tests/query/test_citations.py
git commit -m "feat: build citations from stored metadata, never from the model

A fabricated deep link is worse than no link: it looks authoritative and
fails silently. The only way to make that impossible is to leave the model
no opportunity to produce one, so every URL a user can click is constructed
here from what the indexer stored.

Notes needed a vault-relative path that is nowhere in the index - the
chunker keeps the full S3 key as the index identity and drops the display
path - so it is reconstructed from source_path and vault_id. rpartition
rather than partition, so a vault named after a folder appearing earlier in
the key truncates at the vault.

Both constructors prefer no URL to a guessed one, and t=0 is treated as the
real deep link it is rather than a missing value.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Prompt assembly and citation-index parsing

The two pure halves of `generate.py`'s job, split out so they are testable without a Bedrock client. §6.4's last row lives here: a cited index the model invented is the one failure decision 24 cannot prevent, only contain.

**Files:**
- Create: `src/notes_rag/query/prompt.py`
- Test: `tests/query/test_prompt.py`

**Interfaces:**
- Consumes: `store.base.SearchHit`.
- Produces:
  - `prompt.SYSTEM_PROMPT: str`
  - `prompt.REFUSAL_TEXT: str`
  - `prompt.build_user_message(question: str, hits: Sequence[SearchHit]) -> str`
  - `prompt.parse_answer(raw: str, context_count: int) -> ParsedAnswer`
  - `prompt.ParsedAnswer(text: str, indices: list[int], dropped: list[int])` — frozen dataclass, `indices` 0-based and deduplicated in first-appearance order

- [ ] **Step 1: Write the failing tests**

Create `tests/query/test_prompt.py`:

```python
from notes_rag.models import Chunk
from notes_rag.query.prompt import (
    SYSTEM_PROMPT,
    build_user_message,
    parse_answer,
)
from notes_rag.store.base import SearchHit


def hit(text: str, context: str = "ctx", chunk_id: str = "c") -> SearchHit:
    return SearchHit(
        chunk=Chunk(
            id=chunk_id,
            corpus="video",
            vault_id=None,
            source_path="summaries/x.json",
            chunk_type="summary",
            title="T",
            heading=None,
            context=context,
            text=text,
            content_hash="h",
        ),
        distance=0.2,
    )


def test_system_prompt_forbids_urls_and_timestamps():
    lowered = SYSTEM_PROMPT.lower()
    assert "url" in lowered
    assert "[1]" in SYSTEM_PROMPT


def test_user_message_numbers_context_blocks_from_one():
    message = build_user_message("q", [hit("alpha"), hit("beta")])
    assert "[1]" in message
    assert "[2]" in message
    assert message.index("[1]") < message.index("[2]")


def test_user_message_includes_context_prefix_and_text():
    message = build_user_message("q", [hit("the body text", context="Course / Lecture 1")])
    assert "Course / Lecture 1" in message
    assert "the body text" in message


def test_user_message_includes_the_question():
    assert "what is a monad?" in build_user_message("what is a monad?", [hit("a")])


def test_parses_a_single_citation():
    parsed = parse_answer("Registers are allocated greedily [1].", context_count=3)
    assert parsed.indices == [0]
    assert parsed.dropped == []


def test_parses_multiple_and_deduplicates_in_first_appearance_order():
    parsed = parse_answer("A [3] then B [1] then C [3].", context_count=3)
    assert parsed.indices == [2, 0]


def test_parses_a_comma_separated_group():
    parsed = parse_answer("Both sources agree [1, 3].", context_count=3)
    assert parsed.indices == [0, 2]


def test_drops_an_index_past_the_context_count():
    """The one failure decision 24 cannot prevent, only contain."""
    parsed = parse_answer("As shown [7].", context_count=3)
    assert parsed.indices == []
    assert parsed.dropped == [7]
    assert parsed.text == "As shown [7]."


def test_drops_index_zero():
    """Context blocks are 1-based in the prompt, so [0] is never valid."""
    parsed = parse_answer("See [0].", context_count=3)
    assert parsed.indices == []
    assert parsed.dropped == [0]


def test_keeps_valid_indices_alongside_invalid_ones():
    parsed = parse_answer("First [1], second [9].", context_count=2)
    assert parsed.indices == [0]
    assert parsed.dropped == [9]


def test_answer_text_is_preserved_verbatim():
    raw = "  Leading and trailing space is stripped.  [1]  "
    assert parse_answer(raw, context_count=1).text == "Leading and trailing space is stripped.  [1]"


def test_no_citations_yields_empty_indices():
    parsed = parse_answer("I could not find that.", context_count=3)
    assert parsed.indices == []
    assert parsed.dropped == []


def test_bracketed_non_numbers_are_ignored():
    parsed = parse_answer("An aside [see also] and a cite [1].", context_count=2)
    assert parsed.indices == [0]
    assert parsed.dropped == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/mnt/c/Users/joshi/Notes Rag" && uv run pytest tests/query/test_prompt.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'notes_rag.query.prompt'`

- [ ] **Step 3: Write `src/notes_rag/query/prompt.py`**

```python
"""Prompt assembly and citation-index parsing. Pure - no network, no clients.

The model's only citation vocabulary is `[n]`, referring to a numbered context
block. It never sees a URL, a timestamp, or a file path it could echo back,
which is what makes decision 24's guarantee structural rather than a matter of
the model behaving. The residual risk - a number that refers to nothing - is
contained here rather than prevented, because it cannot be prevented.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from notes_rag.store.base import SearchHit

# Any bracketed run of digits and separators: [1], [2,3], [1, 2, 3].
_CITATION_GROUP = re.compile(r"\[([\d\s,]+)\]")

SYSTEM_PROMPT = """\
You answer questions about a student's study material using only the numbered \
context blocks provided.

Ground every claim in the context. If the context does not contain the answer, \
say so plainly rather than filling the gap from general knowledge - the person \
asking wants to know what their own material says.

Cite the context blocks you used by their number in square brackets, like [1] \
or [2, 3], placed at the end of the sentence they support. Cite only blocks you \
actually drew on.

Never write a URL, a video timestamp, a file path, or a link of any kind. The \
application builds those from the block numbers you cite; anything link-shaped \
you write will be wrong. Refer to sources by their number alone.

Be concise and direct. Answer the question that was asked."""

REFUSAL_TEXT = (
    "I could not find anything in your study material that answers that. "
    "It may not be covered by the notes and videos currently indexed."
)


@dataclass(frozen=True)
class ParsedAnswer:
    text: str
    indices: list[int]  # 0-based, deduplicated, first-appearance order
    dropped: list[int]  # 1-based numbers the model cited that do not exist


def build_user_message(question: str, hits: Sequence[SearchHit]) -> str:
    """The question plus numbered context blocks.

    Each block leads with the chunk's `context` prefix - the same
    human-readable breadcrumb that was embedded - because it tells the model
    which lecture or note it is reading without a separate metadata channel.
    """
    blocks = []
    for number, hit in enumerate(hits, start=1):
        blocks.append(f"[{number}] {hit.chunk.context}\n{hit.chunk.text}")
    joined = "\n\n".join(blocks)
    return f"Context:\n\n{joined}\n\nQuestion: {question}"


def parse_answer(raw: str, context_count: int) -> ParsedAnswer:
    """Extract cited block numbers, separating the valid from the invented.

    Returns 0-based indices into the hit list. The answer text is returned
    unmodified apart from surrounding whitespace: rewriting it to strip a bad
    citation marker risks corrupting a sentence, and leaving `[7]` visible in
    the prose while dropping it from the citation list is the more honest
    failure - the reader can see something is off.
    """
    indices: list[int] = []
    dropped: list[int] = []
    seen: set[int] = set()

    for group in _CITATION_GROUP.finditer(raw):
        for part in group.group(1).split(","):
            part = part.strip()
            if not part.isdigit():
                continue
            number = int(part)
            if not 1 <= number <= context_count:
                if number not in dropped:
                    dropped.append(number)
                continue
            index = number - 1
            if index not in seen:
                seen.add(index)
                indices.append(index)

    return ParsedAnswer(text=raw.strip(), indices=indices, dropped=dropped)
```

- [ ] **Step 4: Run the tests**

Run: `cd "/mnt/c/Users/joshi/Notes Rag" && uv run pytest tests/query/test_prompt.py -v && uv run ruff check .`
Expected: PASS (13 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
cd "/mnt/c/Users/joshi/Notes Rag"
git add src/notes_rag/query/prompt.py tests/query/test_prompt.py
git commit -m "feat: assemble prompts and parse citation indices

The model's entire citation vocabulary is [n] against a numbered context
block. It never sees a URL, timestamp, or file path it could echo, which is
what makes the no-fabricated-links guarantee structural rather than a matter
of the model cooperating.

The residual risk is a number referring to nothing, and that is contained
rather than prevented: invalid indices are dropped from the citation list
and reported, while the answer text is left verbatim. Rewriting prose to
hide a bad marker risks corrupting the sentence, and a visible [7] with no
citation attached is the more honest failure.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: The Bedrock generation call

Decisions 27 and 29 in code, plus the §6.4 error classification. This is the only task that constructs an Anthropic client.

**Files:**
- Create: `src/notes_rag/query/generate.py`
- Modify: `pyproject.toml` (add the `anthropic` dependency)
- Test: `tests/query/test_generate.py`

**Interfaces:**
- Consumes: `prompt.SYSTEM_PROMPT`, `prompt.build_user_message`, `prompt.parse_answer`, `prompt.ParsedAnswer`, `errors.UpstreamThrottled`, `errors.GenerationFailed`, `store.base.SearchHit`.
- Produces:
  - `generate.MODEL_ID: str`
  - `generate.DEFAULT_REGION: str`
  - `generate.Generation(answer: ParsedAnswer, model: str, input_tokens: int, output_tokens: int)` — frozen dataclass
  - `generate.HaikuGenerator(*, region: str = DEFAULT_REGION, model_id: str = MODEL_ID, max_tokens: int = 1024, client=None)` with `.generate(question: str, hits: Sequence[SearchHit]) -> Generation`

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add to `dependencies`:

```toml
    # The query Lambda's generation client (decision 29). The bedrock extra
    # pulls boto3/botocore, which the Lambda runtime already provides - see
    # scripts/build_lambda.sh, which installs this without them.
    "anthropic[bedrock]>=0.121",
```

Run: `cd "/mnt/c/Users/joshi/Notes Rag" && uv sync --extra dev`

- [ ] **Step 2: Write the failing tests**

Create `tests/query/test_generate.py`:

```python
import pytest

from notes_rag.models import Chunk
from notes_rag.query.errors import GenerationFailed, UpstreamThrottled
from notes_rag.query.generate import MODEL_ID, HaikuGenerator
from notes_rag.store.base import SearchHit


def hit(text: str = "body") -> SearchHit:
    return SearchHit(
        chunk=Chunk(
            id="c1",
            corpus="video",
            vault_id=None,
            source_path="summaries/x.json",
            chunk_type="summary",
            title="T",
            heading=None,
            context="ctx",
            text=text,
            content_hash="h",
        ),
        distance=0.2,
    )


class _FakeHttpResponse:
    """Minimal httpx.Response stand-in for constructing a real APIStatusError."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.headers = {}
        self.request = None


class _Block:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _Response:
    def __init__(self, text: str, model: str = "claude-haiku-4-5-20251001") -> None:
        self.content = [_Block(text)]
        self.model = model
        self.usage = _Usage(120, 40)


class StubMessages:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.kwargs: dict = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.response


class StubClient:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.messages = StubMessages(response, error)


def test_sends_the_geo_inference_profile_model_id():
    client = StubClient(_Response("A [1]."))
    HaikuGenerator(client=client).generate("q", [hit()])
    assert client.messages.kwargs["model"] == MODEL_ID
    assert MODEL_ID.startswith("us.anthropic.")


def test_omits_thinking_effort_and_sampling_parameters():
    """Decision 27. `effort` errors on Haiku 4.5 and the others buy nothing."""
    client = StubClient(_Response("A [1]."))
    HaikuGenerator(client=client).generate("q", [hit()])
    sent = client.messages.kwargs
    for forbidden in ("thinking", "output_config", "temperature", "top_p", "top_k"):
        assert forbidden not in sent


def test_sends_a_single_user_turn_with_a_system_prompt():
    client = StubClient(_Response("A [1]."))
    HaikuGenerator(client=client).generate("what is X?", [hit("relevant body")])
    sent = client.messages.kwargs
    assert [message["role"] for message in sent["messages"]] == ["user"]
    assert "what is X?" in sent["messages"][0]["content"]
    assert "relevant body" in sent["messages"][0]["content"]
    assert sent["system"]


def test_returns_parsed_answer_model_and_usage():
    client = StubClient(_Response("Answer [1]."))
    result = HaikuGenerator(client=client).generate("q", [hit()])
    assert result.answer.text == "Answer [1]."
    assert result.answer.indices == [0]
    assert result.model == "claude-haiku-4-5-20251001"
    assert result.input_tokens == 120
    assert result.output_tokens == 40


def test_invalid_citation_index_is_reported_not_raised():
    client = StubClient(_Response("Answer [4]."))
    result = HaikuGenerator(client=client).generate("q", [hit()])
    assert result.answer.indices == []
    assert result.answer.dropped == [4]


def test_concatenates_multiple_text_blocks():
    response = _Response("first ")
    response.content.append(_Block("second [1]."))
    result = HaikuGenerator(client=StubClient(response)).generate("q", [hit()])
    assert result.answer.text == "first second [1]."


def test_ignores_non_text_blocks():
    response = _Response("visible [1].")

    class _Other:
        type = "thinking"

    response.content.insert(0, _Other())
    result = HaikuGenerator(client=StubClient(response)).generate("q", [hit()])
    assert result.answer.text == "visible [1]."


def test_throttling_becomes_upstream_throttled():
    import anthropic

    error = anthropic.RateLimitError(
        message="slow down",
        response=_FakeHttpResponse(429),
        body=None,
    )
    with pytest.raises(UpstreamThrottled) as caught:
        HaikuGenerator(client=StubClient(error=error)).generate("q", [hit()])
    assert caught.value.http_status == 503
    assert caught.value.retry_after is not None


def test_other_failures_become_generation_failed():
    with pytest.raises(GenerationFailed) as caught:
        HaikuGenerator(client=StubClient(error=RuntimeError("boom"))).generate("q", [hit()])
    assert caught.value.http_status == 502
    assert isinstance(caught.value.__cause__, RuntimeError)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd "/mnt/c/Users/joshi/Notes Rag" && uv run pytest tests/query/test_generate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'notes_rag.query.generate'`

- [ ] **Step 4: Write `src/notes_rag/query/generate.py`**

```python
"""The Haiku 4.5 call, and nothing else.

Model id and client class are both load-bearing and both non-obvious
(decision 29, verified against the live account rather than inferred from
docs):

  * Haiku 4.5 has no in-region availability in us-east-2, so the bare model id
    `anthropic.claude-haiku-4-5-20251001-v1:0` is rejected outright with
    "on-demand throughput isn't supported". A geo inference profile is
    mandatory, not a preference.
  * AnthropicBedrockMantle is the forward-looking client and its endpoint does
    resolve in us-east-2, but every call 403s on bedrock-mantle:CreateInference,
    which this account has never been granted.

Neither is a style choice. Changing either without re-running the probes in
docs/superpowers/specs/2026-08-07-rag-query-design.md §12 will break the
request path.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from notes_rag.query.errors import GenerationFailed, UpstreamThrottled
from notes_rag.query.prompt import (
    SYSTEM_PROMPT,
    ParsedAnswer,
    build_user_message,
    parse_answer,
)
from notes_rag.store.base import SearchHit

logger = logging.getLogger(__name__)

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_REGION = "us-east-2"


@dataclass(frozen=True)
class Generation:
    answer: ParsedAnswer
    model: str
    input_tokens: int
    output_tokens: int


class HaikuGenerator:
    def __init__(
        self,
        *,
        region: str = DEFAULT_REGION,
        model_id: str = MODEL_ID,
        max_tokens: int = 1024,
        client=None,
    ) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        if client is None:
            # Imported lazily and constructed only when no client is injected,
            # so importing this module never reaches for AWS credentials -
            # the same seam TitanEmbedder uses.
            from anthropic import AnthropicBedrock

            client = AnthropicBedrock(aws_region=region)
        self._client = client

    def generate(self, question: str, hits: Sequence[SearchHit]) -> Generation:
        """One non-streaming turn. No thinking, no effort, no sampling params."""
        try:
            response = self._client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_message(question, hits)}],
            )
        except Exception as error:
            raise _classify(error) from error

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        answer = parse_answer(text, context_count=len(hits))
        if answer.dropped:
            # Decision 24's residual risk, realised. Worth a warning rather
            # than silence: a model citing blocks that do not exist is a
            # prompt problem, and the only place it is visible is here.
            logger.warning(
                "model cited %d context block(s) that do not exist: %s",
                len(answer.dropped),
                answer.dropped,
            )

        return Generation(
            answer=answer,
            model=getattr(response, "model", self.model_id),
            input_tokens=getattr(response.usage, "input_tokens", 0),
            output_tokens=getattr(response.usage, "output_tokens", 0),
        )


def _classify(error: Exception) -> Exception:
    """Throttling is retryable and everything else here is not (§6.4).

    anthropic is imported inside the function so that a unit test injecting a
    stub client never needs the package's exception hierarchy loaded, and so
    an import failure surfaces as a generation error rather than at module
    import time.
    """
    try:
        import anthropic
    except ImportError:  # pragma: no cover - the package is a hard dependency
        return GenerationFailed("anthropic client unavailable")

    if isinstance(error, anthropic.RateLimitError):
        return UpstreamThrottled("bedrock throttled the generation request")
    if isinstance(error, anthropic.APIStatusError) and error.status_code == 429:
        return UpstreamThrottled("bedrock throttled the generation request")
    return GenerationFailed(f"generation failed: {type(error).__name__}")
```

- [ ] **Step 5: Run the tests**

Run: `cd "/mnt/c/Users/joshi/Notes Rag" && uv run pytest tests/query/test_generate.py -v && uv run ruff check .`
Expected: PASS (9 tests), ruff clean

If `anthropic.RateLimitError(...)` cannot be constructed with those arguments in the installed version, adjust the test to construct whatever the installed SDK requires — inspect with `uv run python -c "import anthropic, inspect; print(inspect.signature(anthropic.RateLimitError.__init__))"`. Do not weaken the test to catching a bare `Exception`; the point is that a real throttle maps to a retryable status.

- [ ] **Step 6: Commit**

```bash
cd "/mnt/c/Users/joshi/Notes Rag"
git add src/notes_rag/query/generate.py tests/query/test_generate.py pyproject.toml uv.lock
git commit -m "feat: call Haiku 4.5 through the US geo inference profile

Both the client class and the model id are load-bearing and were settled by
real calls rather than documentation. Haiku 4.5 has no in-region
availability in us-east-2, so the bare model id is rejected outright and an
inference profile is mandatory; AnthropicBedrockMantle resolves but 403s on
an IAM action this account has never been granted.

The module docstring says so, because the next reader's instinct on seeing
a us.-prefixed id and a legacy client will be to simplify both.

Throttling maps to a retryable status and everything else does not, so a
caller can tell 'try again' from 'this will keep failing'.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Query handler — config, orchestration, response contract

**Files:**
- Create: `src/notes_rag/query/handler.py`
- Test: `tests/query/test_handler.py`

**Interfaces:**
- Consumes: everything from Tasks 1–6.
- Produces:
  - `handler.QueryConfig` — frozen dataclass with `index_bucket`, `db_key`, `corpus_scope`, `dimensions=1024`, `bedrock_region="us-east-2"`, `work_dir="/tmp"`, `k=6`, `distance_threshold`, `max_tokens=1024`; classmethod `from_env(env, *, db_key_var, corpus_scope)`
  - `handler.answer_question(question: str, config: QueryConfig, *, s3, embedder, generator, cache: ArtifactCache) -> dict` — the §6.3 payload
  - `handler.build_cache(config: QueryConfig) -> ArtifactCache`
  - `handler.lambda_handler(event, context) -> dict`

- [ ] **Step 1: Write the failing tests**

Create `tests/query/test_handler.py`:

```python
import dataclasses
import json
from pathlib import Path

import pytest

from notes_rag.embed.fake import FakeEmbedder
from notes_rag.models import Chunk
from notes_rag.query.artifact import ArtifactCache
from notes_rag.query.errors import ArtifactMissing
from notes_rag.query.generate import Generation
from notes_rag.query.handler import QueryConfig, answer_question
from notes_rag.query.prompt import REFUSAL_TEXT, ParsedAnswer
from notes_rag.store.sqlite_vec import SqliteVecStore

DIMS = 8


def video_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        corpus="video",
        vault_id=None,
        source_path="summaries/x.json",
        chunk_type="summary",
        title="Compilers",
        heading="Registers",
        context="Compilers / Lecture 3 / Registers",
        text=text,
        content_hash=chunk_id,
        video_id="abc",
        start_seconds=1120,
        url="https://www.youtube.com/watch?v=abc",
    )


@pytest.fixture
def index_bytes(tmp_path):
    """A real sqlite-vec index, serialised to bytes for the S3 stub."""
    path = tmp_path / "seed.db"
    store = SqliteVecStore(path, dimensions=DIMS)
    embedder = FakeEmbedder(dimensions=DIMS)
    chunks = [video_chunk("c1", "register allocation is greedy")]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    store.close()
    raw = path.read_bytes()
    path.unlink()
    return raw


@pytest.fixture
def config(tmp_path):
    return QueryConfig(
        index_bucket="index-bucket",
        db_key="index/full.db",
        corpus_scope="full",
        dimensions=DIMS,
        work_dir=str(tmp_path),
        distance_threshold=2.0,  # permissive: these tests are not about the gate
    )


class StubGenerator:
    def __init__(self, generation: Generation) -> None:
        self.generation = generation
        self.calls = 0

    def generate(self, question, hits):
        self.calls += 1
        self.hits = hits
        return self.generation


def generation(text: str, indices: list[int], dropped: list[int] | None = None) -> Generation:
    return Generation(
        answer=ParsedAnswer(text=text, indices=indices, dropped=dropped or []),
        model="claude-haiku-4-5-20251001",
        input_tokens=120,
        output_tokens=40,
    )


def run(config, s3, generator, question="how are registers allocated?"):
    cache = ArtifactCache(
        bucket=config.index_bucket,
        key=config.db_key,
        dest=Path(config.work_dir) / "index.db",
    )
    return answer_question(
        question,
        config,
        s3=s3,
        embedder=FakeEmbedder(dimensions=DIMS),
        generator=generator,
        cache=cache,
    )


def test_returns_the_spec_response_contract(config, index_bytes, make_s3):
    s3 = make_s3({"index/full.db": index_bytes})
    result = run(config, s3, StubGenerator(generation("Greedily [1].", [0])))

    assert set(result) == {"answer", "citations", "corpus_scope", "model", "usage"}
    assert result["answer"] == "Greedily [1]."
    assert result["corpus_scope"] == "full"
    assert result["model"] == "claude-haiku-4-5-20251001"
    assert result["usage"] == {"input_tokens": 120, "output_tokens": 40}


def test_maps_cited_indices_to_citations(config, index_bytes, make_s3):
    s3 = make_s3({"index/full.db": index_bytes})
    result = run(config, s3, StubGenerator(generation("Greedily [1].", [0])))

    assert len(result["citations"]) == 1
    citation = result["citations"][0]
    assert citation["kind"] == "video"
    assert citation["url"] == "https://www.youtube.com/watch?v=abc&t=1120"
    assert citation["chunk_id"] == "c1"


def test_uncited_hits_do_not_become_citations(config, index_bytes, make_s3):
    """Citations list what the answer used, not what retrieval found."""
    s3 = make_s3({"index/full.db": index_bytes})
    result = run(config, s3, StubGenerator(generation("No sources needed.", [])))
    assert result["citations"] == []


def test_refuses_without_calling_the_model(config, index_bytes, make_s3):
    s3 = make_s3({"index/full.db": index_bytes})
    strict = dataclasses.replace(config, distance_threshold=0.0001)
    generator = StubGenerator(generation("unreachable", []))
    result = run(strict, s3, generator)

    assert result["answer"] == REFUSAL_TEXT
    assert result["citations"] == []
    assert result["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert generator.calls == 0


def test_missing_artifact_raises_artifact_missing(config, make_s3):
    with pytest.raises(ArtifactMissing):
        run(config, make_s3({}), StubGenerator(generation("x", [])))


def test_demo_scope_is_echoed(config, index_bytes, make_s3):
    s3 = make_s3({"index/public.db": index_bytes})
    demo = dataclasses.replace(config, db_key="index/public.db", corpus_scope="public")
    result = run(demo, s3, StubGenerator(generation("Greedily [1].", [0])))
    assert result["corpus_scope"] == "public"


def test_config_from_env_reads_the_named_key_variable():
    env = {"INDEX_BUCKET": "b", "FULL_DB_KEY": "index/full.db", "EMBED_DIMENSIONS": "1024"}
    config = QueryConfig.from_env(env, db_key_var="FULL_DB_KEY", corpus_scope="full")
    assert config.index_bucket == "b"
    assert config.db_key == "index/full.db"
    assert config.corpus_scope == "full"


def test_config_from_env_defaults_the_key():
    config = QueryConfig.from_env({"INDEX_BUCKET": "b"}, db_key_var="PUBLIC_DB_KEY",
                                  corpus_scope="public")
    assert config.db_key == "index/public.db"


def test_config_from_env_requires_the_bucket():
    with pytest.raises(KeyError):
        QueryConfig.from_env({}, db_key_var="FULL_DB_KEY", corpus_scope="full")


def test_config_rejects_an_unknown_corpus_scope():
    with pytest.raises(ValueError):
        QueryConfig.from_env({"INDEX_BUCKET": "b"}, db_key_var="FULL_DB_KEY", corpus_scope="all")


def test_response_is_json_serialisable(config, index_bytes, make_s3):
    s3 = make_s3({"index/full.db": index_bytes})
    result = run(config, s3, StubGenerator(generation("Greedily [1].", [0])))
    json.dumps(result)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/mnt/c/Users/joshi/Notes Rag" && uv run pytest tests/query/test_handler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'notes_rag.query.handler'`

- [ ] **Step 3: Write `src/notes_rag/query/handler.py`**

```python
"""The query Lambda: question in, grounded and cited answer out.

Both entrypoints - this one over full.db and demo/handler.py over public.db -
run the same orchestration and differ only in which artifact key they read and
which scope they echo. The isolation between them is enforced by their IAM
roles (decision 5), not by this code: a bug here cannot leak full.db because
the demo role cannot read it.

Module-level state is deliberate. `_CACHE` survives between invocations on a
warm container, which is the entire mechanism that makes the ETag check cheap
enough to run every time.
"""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from notes_rag.embed.base import Embedder
from notes_rag.query.artifact import ArtifactCache
from notes_rag.query.citations import build_citation
from notes_rag.query.errors import QueryError
from notes_rag.query.prompt import REFUSAL_TEXT
from notes_rag.query.retrieve import DEFAULT_DISTANCE_THRESHOLD, DEFAULT_K, retrieve
from notes_rag.store.sqlite_vec import SqliteVecStore

logger = logging.getLogger(__name__)

VALID_SCOPES = ("full", "public")


@dataclass(frozen=True)
class QueryConfig:
    index_bucket: str
    db_key: str
    corpus_scope: str
    dimensions: int = 1024
    bedrock_region: str = "us-east-2"
    work_dir: str = "/tmp"
    k: int = DEFAULT_K
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD
    max_tokens: int = 1024

    @classmethod
    def from_env(
        cls, env: Mapping[str, str], *, db_key_var: str, corpus_scope: str
    ) -> "QueryConfig":
        """Build config from environment variables.

        `db_key_var` is a parameter rather than a constant so the two
        entrypoints share this method: the demo reads PUBLIC_DB_KEY, the
        authenticated path reads FULL_DB_KEY, and neither can accidentally
        read the other's default.
        """
        if corpus_scope not in VALID_SCOPES:
            raise ValueError(f"corpus_scope must be one of {VALID_SCOPES}: {corpus_scope!r}")
        default_key = "index/public.db" if corpus_scope == "public" else "index/full.db"
        return cls(
            index_bucket=env["INDEX_BUCKET"],
            db_key=env.get(db_key_var, default_key),
            corpus_scope=corpus_scope,
            dimensions=int(env.get("EMBED_DIMENSIONS", "1024")),
            bedrock_region=env.get("BEDROCK_REGION", "us-east-2"),
            work_dir=env.get("WORK_DIR", "/tmp"),
            k=int(env.get("RETRIEVAL_K", str(DEFAULT_K))),
            distance_threshold=float(
                env.get("DISTANCE_THRESHOLD", str(DEFAULT_DISTANCE_THRESHOLD))
            ),
            max_tokens=int(env.get("MAX_TOKENS", "1024")),
        )


def build_cache(config: QueryConfig) -> ArtifactCache:
    # One local filename per artifact key, so a demo container and a full
    # container could not collide even if they somehow shared /tmp.
    name = Path(config.db_key).name
    return ArtifactCache(
        bucket=config.index_bucket,
        key=config.db_key,
        dest=Path(config.work_dir) / name,
    )


def answer_question(
    question: str,
    config: QueryConfig,
    *,
    s3,
    embedder: Embedder,
    generator,
    cache: ArtifactCache,
) -> dict:
    """The §6.3 response payload. Raises QueryError subclasses on failure."""
    path = cache.ensure_current(s3)

    store = SqliteVecStore(path, dimensions=config.dimensions)
    try:
        result = retrieve(
            question,
            store,
            embedder,
            k=config.k,
            threshold=config.distance_threshold,
        )
        if result.refused:
            return {
                "answer": REFUSAL_TEXT,
                "citations": [],
                "corpus_scope": config.corpus_scope,
                "model": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }

        generated = generator.generate(question, result.hits)

        # Citations list what the answer actually used, not what retrieval
        # found: six hits behind a one-sentence answer would be six links the
        # reader has no reason to follow.
        citations = [
            build_citation(result.hits[index].chunk).to_dict()
            for index in generated.answer.indices
        ]
    finally:
        store.close()

    return {
        "answer": generated.answer.text,
        "citations": citations,
        "corpus_scope": config.corpus_scope,
        "model": generated.model,
        "usage": {
            "input_tokens": generated.input_tokens,
            "output_tokens": generated.output_tokens,
        },
    }


# Warm-container state: the point of the ETag cache is that it persists.
_CACHE: ArtifactCache | None = None


def _extract_question(event) -> str:
    """Accept a bare {"question": ...} or an API Gateway proxy body.

    Plan 5 puts API Gateway in front of this; accepting both shapes now means
    the `aws lambda invoke` verification path stays usable afterwards.
    """
    import json

    if isinstance(event, Mapping) and "body" in event and "question" not in event:
        body = event["body"]
        event = json.loads(body) if isinstance(body, str) else (body or {})

    question = (event or {}).get("question") if isinstance(event, Mapping) else None
    if not isinstance(question, str) or not question.strip():
        raise ValueError("event must carry a non-empty 'question'")
    return question.strip()


def _run(event, *, db_key_var: str, corpus_scope: str) -> dict:
    """Shared entrypoint body. `demo/handler.py` calls this with its own key."""
    global _CACHE

    logging.getLogger().setLevel(logging.INFO)

    import boto3

    from notes_rag.embed.bedrock import TitanEmbedder
    from notes_rag.query.generate import HaikuGenerator

    config = QueryConfig.from_env(os.environ, db_key_var=db_key_var, corpus_scope=corpus_scope)
    if _CACHE is None:
        _CACHE = build_cache(config)

    try:
        question = _extract_question(event)
    except ValueError as error:
        return {"error": str(error), "status": 400}

    try:
        return answer_question(
            question,
            config,
            s3=boto3.client("s3"),
            embedder=TitanEmbedder(region=config.bedrock_region, dimensions=config.dimensions),
            generator=HaikuGenerator(region=config.bedrock_region, max_tokens=config.max_tokens),
            cache=_CACHE,
        )
    except QueryError as error:
        # Plan 5's API Gateway adapter maps http_status/retry_after properly.
        # Until then this shape keeps `aws lambda invoke` legible.
        logger.warning("query failed: %s", error)
        payload = {"error": str(error), "status": error.http_status}
        if error.retry_after is not None:
            payload["retry_after"] = error.retry_after
        return payload


def lambda_handler(event, context) -> dict:
    return _run(event, db_key_var="FULL_DB_KEY", corpus_scope="full")
```

- [ ] **Step 4: Run the tests**

Run: `cd "/mnt/c/Users/joshi/Notes Rag" && uv run pytest tests/query/test_handler.py -v && uv run ruff check .`
Expected: PASS (11 tests), ruff clean

- [ ] **Step 5: Run the whole suite**

Run: `cd "/mnt/c/Users/joshi/Notes Rag" && uv run pytest -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
cd "/mnt/c/Users/joshi/Notes Rag"
git add src/notes_rag/query/handler.py tests/query/test_handler.py
git commit -m "feat: wire the query path into a Lambda entrypoint

The response echoes corpus_scope so the frontend can say plainly which index
answered, rather than leaving a demo user to wonder why their question about
notes found nothing.

Citations list what the answer used rather than what retrieval found: six
hits behind a one-sentence answer would be six links with no reason to
follow them.

The artifact cache is module-level state on purpose - persisting across
invocations on a warm container is the entire mechanism that makes a
per-invocation ETag check affordable.

The entrypoint accepts both a bare {question} and an API Gateway proxy body,
so the aws lambda invoke verification path still works once Plan 5 puts a
gateway in front of it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Demo handler

**Files:**
- Create: `src/notes_rag/demo/__init__.py`
- Create: `src/notes_rag/demo/handler.py`
- Test: `tests/demo/__init__.py`, `tests/demo/test_handler.py`

**Interfaces:**
- Consumes: `notes_rag.query.handler._run`.
- Produces: `demo.handler.lambda_handler(event, context) -> dict`

- [ ] **Step 1: Write the failing test**

Create `tests/demo/__init__.py` empty, then `tests/demo/test_handler.py`:

```python
import notes_rag.demo.handler as demo_handler
import notes_rag.query.handler as query_handler


def test_demo_reads_the_public_key_and_public_scope(monkeypatch):
    captured = {}

    def fake_run(event, *, db_key_var, corpus_scope):
        captured.update(db_key_var=db_key_var, corpus_scope=corpus_scope, event=event)
        return {"answer": "ok"}

    monkeypatch.setattr(demo_handler, "_run", fake_run)
    result = demo_handler.lambda_handler({"question": "q"}, None)

    assert captured["db_key_var"] == "PUBLIC_DB_KEY"
    assert captured["corpus_scope"] == "public"
    assert result == {"answer": "ok"}


def test_query_handler_reads_the_full_key_and_full_scope(monkeypatch):
    captured = {}

    def fake_run(event, *, db_key_var, corpus_scope):
        captured.update(db_key_var=db_key_var, corpus_scope=corpus_scope)
        return {}

    monkeypatch.setattr(query_handler, "_run", fake_run)
    query_handler.lambda_handler({"question": "q"}, None)

    assert captured["db_key_var"] == "FULL_DB_KEY"
    assert captured["corpus_scope"] == "full"


def test_demo_never_names_the_full_artifact():
    """The IAM role is the real boundary, but the demo module should not even
    mention the private key - a reviewer scanning it should see nothing that
    could read full.db."""
    source = (
        __import__("pathlib").Path(demo_handler.__file__).read_text()
    )
    assert "full.db" not in source
    assert "FULL_DB_KEY" not in source
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/mnt/c/Users/joshi/Notes Rag" && uv run pytest tests/demo/test_handler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'notes_rag.demo'`

- [ ] **Step 3: Write the demo handler**

Create `src/notes_rag/demo/__init__.py` as an empty file, then `src/notes_rag/demo/handler.py`:

```python
"""The public demo entrypoint: the query path, over public.db only.

This module is deliberately almost empty. Every behavioural difference from
the authenticated path is one of two arguments, and the security boundary is
not here at all - it is the IAM role attached to this function, which has
s3:GetObject on index/public.db and nothing else in that bucket (decision 5).

A bug in this file cannot leak private notes, because the credentials it runs
under cannot read them. That is the property worth preserving: keep the
divergence to configuration, and never reach for the full artifact by name.
"""

from notes_rag.query.handler import _run


def lambda_handler(event, context) -> dict:
    return _run(event, db_key_var="PUBLIC_DB_KEY", corpus_scope="public")
```

- [ ] **Step 4: Run the tests**

Run: `cd "/mnt/c/Users/joshi/Notes Rag" && uv run pytest tests/demo -v && uv run ruff check .`
Expected: PASS (3 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
cd "/mnt/c/Users/joshi/Notes Rag"
git add src/notes_rag/demo tests/demo
git commit -m "feat: add the public demo entrypoint over public.db

Deliberately almost empty. Every difference from the authenticated path is
one of two arguments, and the security boundary is not in this file at all -
it is the IAM role, which can read index/public.db and nothing else.

A test asserts the module never mentions full.db or FULL_DB_KEY. The role is
the real control, but a reviewer scanning this file should find nothing that
could even name the private artifact.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Package, deploy, and verify against the live account

Plan 4 is done when `aws lambda invoke` returns a grounded, cited answer — not when it merges.

**Files:**
- Modify: `scripts/build_lambda.sh`
- Create: `infra/query.tf`
- Modify: `infra/iam.tf`
- Modify: `infra/variables.tf` (add `haiku_model_id`, `haiku_geo_regions`)
- Test: `tests/query/test_handler_integration.py`

**Interfaces:**
- Consumes: Tasks 1–8, existing `aws_s3_bucket.index`, `var.region`, `var.lambda_zip`.
- Produces: deployed `notes-rag-query` and `notes-rag-demo` Lambdas.

- [ ] **Step 1: Add `anthropic` to the Lambda bundle**

In `scripts/build_lambda.sh`, add to the `uv pip install` list (after `"PyYAML>=6.0"`):

```bash
  "anthropic>=0.121"
```

Note it is `anthropic`, **not** `anthropic[bedrock]`: the bedrock extra pulls boto3/botocore, which the runtime already provides and which would add tens of megabytes. `AnthropicBedrock` signs with botocore at runtime, so the runtime's copy is what it uses.

Also update the header comment, which currently claims boto3 is the only exclusion:

```bash
# boto3 is deliberately excluded - the python3.12 runtime provides a current one.
# `anthropic` is installed WITHOUT its [bedrock] extra for the same reason: the
# extra exists only to pull boto3/botocore, and AnthropicBedrock signs with
# whatever botocore is importable at runtime.
```

- [ ] **Step 2: Build and check the bundle size**

```bash
cd "/mnt/c/Users/joshi/Notes Rag"
./scripts/build_lambda.sh
du -sh build/lambda build/lambda.zip
```

Expected: `build/lambda.zip` well under 50MB and `build/lambda` under 250MB (the Lambda hard limits). **If either limit is exceeded, stop and report it** — the fallback is calling `bedrock-runtime:InvokeModel` through boto3 with the Anthropic Messages body (the shape `TitanEmbedder` already uses), which costs zero bundle bytes but discards decision 29's client choice. That is a spec change, not an implementation detail.

- [ ] **Step 3: Verify the import works from the bundle**

```bash
cd "/mnt/c/Users/joshi/Notes Rag"
PYTHONPATH=build/lambda uv run --no-project python -c "
from anthropic import AnthropicBedrock
from notes_rag.query.handler import lambda_handler
from notes_rag.demo.handler import lambda_handler as demo
print('imports ok')
"
```
Expected: `imports ok`

- [ ] **Step 4: Add the Terraform variables**

Append to `infra/variables.tf`:

```hcl
variable "haiku_model_id" {
  description = "Generation model id. Must be an inference profile: Haiku 4.5 has no in-region availability in us-east-2, so a bare model id is rejected."
  type        = string
  default     = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
}

variable "haiku_geo_regions" {
  description = "Destination regions of the us. geo inference profile. Invoking through a profile authorizes against the profile ARN AND the foundation-model ARN in every region it can route to, so a policy naming only the profile fails closed."
  type        = list(string)
  default     = ["us-east-1", "us-east-2", "us-west-2"]
}

variable "haiku_foundation_model" {
  description = "The bare foundation-model id behind the inference profile, used to build the per-region resource ARNs."
  type        = string
  default     = "anthropic.claude-haiku-4-5-20251001-v1:0"
}

variable "distance_threshold" {
  description = "Retrieval distances above this refuse without calling Bedrock. Measured, not guessed - see scripts/measure_threshold.py."
  type        = string
  default     = "<the value measured in Task 3>"
}
```

- [ ] **Step 5: Add the query IAM document to `infra/iam.tf`**

Append (do not modify the existing indexer document):

```hcl
locals {
  # Invoking through an inference profile authorizes against two resource
  # types at once: the profile itself, and the underlying foundation model in
  # every region the profile can route to. A policy naming only the profile
  # gets AccessDenied on the first cross-region hop - and because routing is
  # opportunistic, it would pass in testing and fail later under load.
  haiku_invoke_resources = concat(
    ["arn:aws:bedrock:${var.region}:${data.aws_caller_identity.current.account_id}:inference-profile/${var.haiku_model_id}"],
    [for r in var.haiku_geo_regions : "arn:aws:bedrock:${r}::foundation-model/${var.haiku_foundation_model}"],
  )
}

data "aws_caller_identity" "current" {}

data "aws_iam_policy_document" "query" {
  statement {
    sid       = "ReadFullIndex"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.index.arn}/${var.full_db_key}"]
  }

  statement {
    sid     = "InvokeTitanEmbeddings"
    actions = ["bedrock:InvokeModel"]
    resources = [
      "arn:aws:bedrock:${var.region}::foundation-model/${var.embed_model_id}",
    ]
  }

  statement {
    sid       = "InvokeHaiku"
    actions   = ["bedrock:InvokeModel"]
    resources = local.haiku_invoke_resources
  }
}

data "aws_iam_policy_document" "demo" {
  # Decision 5's boundary. This role can read public.db and cannot name
  # full.db - which is what makes the demo handler's correctness a
  # convenience rather than a security control.
  statement {
    sid       = "ReadPublicIndex"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.index.arn}/${var.public_db_key}"]
  }

  statement {
    sid     = "InvokeTitanEmbeddings"
    actions = ["bedrock:InvokeModel"]
    resources = [
      "arn:aws:bedrock:${var.region}::foundation-model/${var.embed_model_id}",
    ]
  }

  statement {
    sid       = "InvokeHaiku"
    actions   = ["bedrock:InvokeModel"]
    resources = local.haiku_invoke_resources
  }
}
```

If `var.full_db_key` / `var.public_db_key` do not exist, add them to `infra/variables.tf` with defaults `index/full.db` and `index/public.db`, and reference them from `infra/indexer.tf`'s environment block so the two cannot drift. If `data "aws_caller_identity" "current"` already exists elsewhere in `infra/`, do not declare it twice.

- [ ] **Step 6: Write `infra/query.tf`**

```hcl
locals {
  query_env = {
    INDEX_BUCKET       = aws_s3_bucket.index.id
    EMBED_DIMENSIONS   = tostring(var.embed_dimensions)
    BEDROCK_REGION     = var.region
    DISTANCE_THRESHOLD = var.distance_threshold
  }
}

resource "aws_iam_role" "query" {
  name               = "notes-rag-query"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role" "demo" {
  name               = "notes-rag-demo"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy" "query" {
  name   = "notes-rag-query"
  role   = aws_iam_role.query.id
  policy = data.aws_iam_policy_document.query.json
}

resource "aws_iam_role_policy" "demo" {
  name   = "notes-rag-demo"
  role   = aws_iam_role.demo.id
  policy = data.aws_iam_policy_document.demo.json
}

resource "aws_iam_role_policy_attachment" "query_logs" {
  role       = aws_iam_role.query.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy_attachment" "demo_logs" {
  role       = aws_iam_role.demo.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_lambda_function" "query" {
  function_name = "notes-rag-query"
  role          = aws_iam_role.query.arn
  handler       = "notes_rag.query.handler.lambda_handler"
  runtime       = "python3.12"
  architectures = ["x86_64"]

  filename         = var.lambda_zip
  source_code_hash = filebase64sha256(var.lambda_zip)

  # A generation turn is seconds, not minutes, but a cold start downloads the
  # index first. Well short of API Gateway's own 29s ceiling (Plan 5), so the
  # gateway times out before the function does rather than after.
  timeout     = 25
  memory_size = 1024

  ephemeral_storage {
    size = 2048
  }

  environment {
    variables = merge(local.query_env, { FULL_DB_KEY = var.full_db_key })
  }
}

resource "aws_lambda_function" "demo" {
  function_name = "notes-rag-demo"
  role          = aws_iam_role.demo.arn
  handler       = "notes_rag.demo.handler.lambda_handler"
  runtime       = "python3.12"
  architectures = ["x86_64"]

  filename         = var.lambda_zip
  source_code_hash = filebase64sha256(var.lambda_zip)

  timeout     = 25
  memory_size = 1024

  ephemeral_storage {
    size = 2048
  }

  environment {
    variables = merge(local.query_env, { PUBLIC_DB_KEY = var.public_db_key })
  }
}

resource "aws_cloudwatch_log_group" "query" {
  name              = "/aws/lambda/${aws_lambda_function.query.function_name}"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "demo" {
  name              = "/aws/lambda/${aws_lambda_function.demo.function_name}"
  retention_in_days = 14
}
```

If `data "aws_iam_policy_document" "lambda_assume"` does not already exist in `infra/`, add it:

```hcl
data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}
```

Check first — the indexer role already assumes a Lambda principal, so a suitable document is likely present under another name. Reuse it rather than duplicating.

- [ ] **Step 7: Plan the Terraform**

```bash
cd "/mnt/c/Users/joshi/Notes Rag/infra"
terraform init -upgrade > /tmp/tf-init.txt 2>&1; tail -20 /tmp/tf-init.txt
terraform plan -out=/tmp/query.tfplan > /tmp/tf-plan.txt 2>&1; tail -60 /tmp/tf-plan.txt
```

**Never pipe terraform through `head`** — a SIGPIPE strands the S3 state lock and `force-unlock` is blocked by the safety classifier, so Josh has to clear it manually from a `!` prompt.

Expected: creates two roles, two role policies, two attachments, two functions, two log groups. Note that anything referencing a resource created in the same apply shows `(known after apply)` — including env vars — so those get checked after the apply, not in the plan.

- [ ] **Step 8: Apply**

```bash
cd "/mnt/c/Users/joshi/Notes Rag/infra"
terraform apply /tmp/query.tfplan > /tmp/tf-apply.txt 2>&1; tail -30 /tmp/tf-apply.txt
```

If this fails on missing deployer IAM permissions, that is spec §12 item 4 arriving early — report the exact denied action rather than widening a policy on the spot.

- [ ] **Step 9: Verify the deployed environment variables**

```bash
cd "/mnt/c/Users/joshi/Notes Rag"
aws lambda get-function-configuration --function-name notes-rag-query \
  --query 'Environment.Variables' --output json
aws lambda get-function-configuration --function-name notes-rag-demo \
  --query 'Environment.Variables' --output json
```

Expected: query carries `FULL_DB_KEY` and no `PUBLIC_DB_KEY`; demo carries `PUBLIC_DB_KEY` and no `FULL_DB_KEY`. Both carry a `DISTANCE_THRESHOLD` matching Task 3.

- [ ] **Step 10: Invoke the authenticated path — the plan's done-when**

```bash
cd "/mnt/c/Users/joshi/Notes Rag"
aws lambda invoke --function-name notes-rag-query \
  --cli-binary-format raw-in-base64-out \
  --payload '{"question":"how does register allocation work?"}' \
  /tmp/query-out.json > /dev/null
python3 -m json.tool /tmp/query-out.json
```

Expected: an answer grounded in the corpus, at least one citation with a real URL, `"corpus_scope": "full"`, and non-zero `usage`. Replace the question with one you know the corpus covers — reuse a question from `eval/questions.yaml`.

- [ ] **Step 11: Invoke the demo path and confirm the corpus split**

```bash
cd "/mnt/c/Users/joshi/Notes Rag"
aws lambda invoke --function-name notes-rag-demo \
  --cli-binary-format raw-in-base64-out \
  --payload '{"question":"how does register allocation work?"}' \
  /tmp/demo-out.json > /dev/null
python3 -m json.tool /tmp/demo-out.json
```

Expected: `"corpus_scope": "public"` and every citation `"kind": "video"`. **A note citation here is a decision 5 failure** — stop and report it.

- [ ] **Step 12: Confirm the refusal path live**

```bash
cd "/mnt/c/Users/joshi/Notes Rag"
aws lambda invoke --function-name notes-rag-demo \
  --cli-binary-format raw-in-base64-out \
  --payload '{"question":"When should I prune wisteria in a temperate climate?"}' \
  /tmp/refuse-out.json > /dev/null
python3 -m json.tool /tmp/refuse-out.json
```

Expected: the refusal text, `"citations": []`, `"model": null`, and zero usage — proving the threshold gate ran ahead of Bedrock rather than after it.

- [ ] **Step 13: Record the live results as an integration test**

Create `tests/query/test_handler_integration.py`:

```python
"""Live checks against the deployed query and demo Lambdas.

Marked integration and deselected by default (see pyproject addopts). These
are the assertions the Plan 4 verification made by hand, kept runnable so a
later change has something to fail against.

Run with: uv run pytest -m integration tests/query/test_handler_integration.py
"""

import json

import pytest

pytestmark = pytest.mark.integration

IN_CORPUS = "how does register allocation work?"
OUT_OF_CORPUS = "When should I prune wisteria in a temperate climate?"


def invoke(function_name: str, question: str) -> dict:
    import boto3

    client = boto3.client("lambda", region_name="us-east-2")
    response = client.invoke(
        FunctionName=function_name,
        Payload=json.dumps({"question": question}).encode(),
    )
    assert "FunctionError" not in response, response.get("FunctionError")
    return json.loads(response["Payload"].read())


def test_query_returns_a_cited_answer():
    result = invoke("notes-rag-query", IN_CORPUS)
    assert result["corpus_scope"] == "full"
    assert result["answer"]
    assert result["citations"], "expected at least one citation"
    assert result["usage"]["output_tokens"] > 0
    for citation in result["citations"]:
        assert citation["url"], f"citation without a url: {citation}"


def test_demo_never_cites_a_note():
    """Decision 5: the demo role cannot read full.db, so a note here would
    mean the corpus split itself broke."""
    result = invoke("notes-rag-demo", IN_CORPUS)
    assert result["corpus_scope"] == "public"
    assert {citation["kind"] for citation in result["citations"]} <= {"video"}


def test_out_of_corpus_refuses_without_paying_for_generation():
    result = invoke("notes-rag-demo", OUT_OF_CORPUS)
    assert result["citations"] == []
    assert result["model"] is None
    assert result["usage"] == {"input_tokens": 0, "output_tokens": 0}
```

Run: `cd "/mnt/c/Users/joshi/Notes Rag" && uv run pytest -m integration tests/query/test_handler_integration.py -v`
Expected: PASS (3 tests). Adjust `IN_CORPUS` if the question you verified with in Step 10 differed.

- [ ] **Step 14: Confirm the default suite still deselects them**

Run: `cd "/mnt/c/Users/joshi/Notes Rag" && uv run pytest -q && uv run ruff check .`
Expected: all unit tests pass, the three integration tests are deselected, ruff clean.

- [ ] **Step 15: Commit**

```bash
cd "/mnt/c/Users/joshi/Notes Rag"
git add scripts/build_lambda.sh infra/ tests/query/test_handler_integration.py
git commit -m "feat: deploy the query and demo Lambdas

The generation grant is not the single-ARN shape the indexer uses for Titan.
Invoking through an inference profile authorizes against the profile ARN and
the foundation-model ARN in every region the profile can route to, so a
policy naming only the profile passes in testing and then fails later under
load, when routing first crosses a region. The region list is a variable, so
switching profiles is a variable change rather than an archaeology exercise.

anthropic is bundled without its [bedrock] extra: the extra exists only to
pull boto3/botocore, which the runtime already provides and which would cost
tens of megabytes of bundle for nothing.

The two functions get separate roles, and the demo role cannot name full.db.
That is decision 5's boundary, and it is why the demo handler being correct
is a convenience rather than a security control.

Timeout is 25s, under API Gateway's 29s ceiling, so once Plan 5 lands the
gateway times out after the function rather than before it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 16: Push**

```bash
cd "/mnt/c/Users/joshi/Notes Rag" && git push origin main
```

---

## Done when

- `aws lambda invoke --function-name notes-rag-query` returns a grounded answer with at least one real citation URL.
- `aws lambda invoke --function-name notes-rag-demo` returns `corpus_scope: "public"` and cites no notes.
- An out-of-corpus question refuses with zero citations and zero token usage.
- `uv run pytest -q` and `ruff check .` are clean; the three integration tests are deselected by default.

## Deliberately not in this plan

| Deferred | Where it lands |
|---|---|
| CloudFront, API Gateway, Cognito | Plan 5 |
| The DynamoDB spend guard and its counters | Plan 5 |
| Mapping `QueryError.http_status` to real HTTP responses | Plan 5 — the attribute exists now so the adapter is a translation, not a redesign |
| Frontend | Plan 6 |
| Groundedness / citation-accuracy scoring with an LLM judge | Not planned; `eval/run.py` remains retrieval-only |
| Prompt caching | Spec §12: Haiku 4.5's minimum cacheable prefix is 4096 tokens and the system prompt does not approach it, so a `cache_control` marker would silently cache nothing |
