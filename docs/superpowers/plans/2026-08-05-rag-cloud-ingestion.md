# RAG Cloud Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the existing local indexer in AWS on a schedule — an indexer Lambda that watches the Video Vault S3 bucket, rebuilds the index when its contents change, and uploads `full.db` and `public.db`, with all infrastructure in Terraform.

**Architecture:** A `sources/s3.py` adapter lists objects with ETags and moves bytes; `indexer/manifest.py` diffs those ETags against the manifest from the previous run; `indexer/handler.py` orchestrates — on an empty diff it exits in ~200ms without downloading anything, otherwise it re-chunks every source, calls the existing `build_index` (which re-embeds only chunks whose `content_hash` changed), writes `public.db` via `copy_filtered`, and uploads both artifacts plus the new manifest. Terraform provisions the index bucket, the Lambda, its IAM role, and an EventBridge Scheduler firing every 5 minutes.

**Tech Stack:** Python 3.12, pytest, Terraform >= 1.10, AWS Lambda, S3, EventBridge Scheduler, Amazon Bedrock (Titan Text Embeddings v2).

## Global Constraints

- Python 3.12. Target runtime is AWS Lambda `python3.12`. No dependency may require a compile step at install time — wheels only.
- **The Lambda runtime's stdlib `sqlite3` is built WITHOUT loadable-extension support.** `sqlite-vec` therefore cannot load through it. `src/notes_rag/store/sqlite_vec.py` already prefers `pysqlite3`, which bundles its own SQLite with extensions enabled; the deployment bundle MUST include `pysqlite3-binary` or the store fails on its first connection. Verified against `public.ecr.aws/lambda/python:3.12`.
- Embedding model is `amazon.titan-embed-text-v2:0` at **1024 dimensions**, in **`us-east-2`**. Entitlement verified 2026-08-04.
- Everything lives in **`us-east-2`**: the Video Vault bucket, the index bucket, the Lambda, Bedrock.
- **No AWS calls in unit tests.** Every test in `tests/` must pass with no credentials. AWS-touching tests are marked `@pytest.mark.integration` and are deselected by default.
- Terraform state uses a **dedicated bucket created by a run-once bootstrap stack with local state**, mirroring the pattern already in `SASE-UARK-Website/infra/bootstrap/`. Native S3 locking (`use_lockfile = true`, Terraform >= 1.10) — no DynamoDB lock table.
- `reserved_concurrent_executions = 1` on the indexer. Two concurrent runs would race on the index artifact.
- Conventional commit prefixes (`feat:`, `test:`, `chore:`, `fix:`). Lint (`ruff check` and `ruff format --check` over `src tests eval`) and tests green before every commit.
- Existing code is **not** to be rewritten. `build_index`, `derive_backlinks`, `SqliteVecStore`, `TitanEmbedder`, and the three chunkers ship as-is. Task 3 is the one refactor, and it is behaviour-preserving.

---

## Existing interfaces this plan builds on

All of these already exist on `main` and must be used exactly as written — do not redefine them.

```python
# notes_rag.models
@dataclass(frozen=True)
class Chunk:
    id: str; corpus: str; vault_id: str | None; source_path: str; chunk_type: str
    title: str; heading: str | None; context: str; text: str; content_hash: str
    video_id: str | None = None; start_seconds: int | None = None; url: str | None = None
    links_to: tuple[str, ...] = (); backlinks: tuple[str, ...] = ()

# notes_rag.indexer.build
@dataclass(frozen=True)
class BuildStats:
    chunks_written: int; vectors_embedded: int; vectors_reused: int; paths_deleted: int

def derive_backlinks(chunks: Sequence[Chunk]) -> list[Chunk]: ...
def build_index(chunks: Sequence[Chunk], store: VectorStore, embedder: Embedder) -> BuildStats: ...

# notes_rag.store.sqlite_vec
class SqliteVecStore:
    def __init__(self, path: str | Path, *, dimensions: int = 1024) -> None: ...
    def copy_filtered(self, dest: str | Path, *, corpus: str) -> None: ...
    def close(self) -> None: ...

# notes_rag.embed.bedrock
class TitanEmbedder:
    def __init__(self, *, region: str = "us-east-2", dimensions: int = 1024, client=None) -> None: ...

# notes_rag.embed.fake
class FakeEmbedder:
    def __init__(self, dimensions: int = 1024) -> None: ...

# notes_rag.chunkers
def chunk_video_summary(summary: dict, *, source_path: str) -> list[Chunk]: ...
def chunk_video_transcript(transcript: dict, summary: dict, *, source_path: str) -> list[Chunk]: ...
def chunk_markdown(text: str, *, source_path: str, vault_id: str) -> list[Chunk]: ...
```

---

## Design note: why every run re-chunks everything

`build_index` deletes any source path present in the store but absent from the `chunks` argument — that is how it handles deletions and renames. Passing only the *changed* chunks would therefore delete every unchanged source from the index.

So the ETag diff is **not** used to decide what to chunk. It is used for exactly one thing: the **no-op early exit**. This matches spec §3's split:

- **Embedding is incremental** — `build_index` reuses any vector whose `content_hash` is already in the store, so an unchanged chunk costs one index lookup instead of a Bedrock call.
- **Artifact assembly is a full rebuild** — chunking is local, free, and takes milliseconds. Re-chunking everything every run is simpler than a partial-update path and cannot drift.

The no-op path is the load-bearing optimisation: roughly 8,639 of ~8,640 monthly runs find nothing changed and must return **before** downloading `full.db`.

---

## File Structure

```
src/notes_rag/sources/__init__.py
src/notes_rag/sources/s3.py            S3Object, list_objects, get/put/download/upload
src/notes_rag/indexer/manifest.py      Manifest, ManifestDiff — pure ETag bookkeeping
src/notes_rag/indexer/collect.py       SourceDocument -> classified docs -> chunks (shared by CLI and Lambda)
src/notes_rag/indexer/handler.py       IndexerConfig, run_index, lambda_handler
src/notes_rag/indexer/cli.py           MODIFIED: delegates classification to collect.py

tests/sources/__init__.py
tests/sources/test_s3.py
tests/indexer/test_manifest.py
tests/indexer/test_collect.py
tests/indexer/test_handler.py
tests/indexer/test_handler_integration.py   integration-marked, real AWS

scripts/build_lambda.sh                cross-platform wheel bundle, no Docker

infra/bootstrap/main.tf                run-once: creates the Terraform state bucket
infra/bootstrap/.gitignore
infra/main.tf                          backend, provider, default_tags
infra/variables.tf
infra/storage.tf                       index bucket
infra/iam.tf                           indexer role + policies
infra/indexer.tf                       Lambda + EventBridge Scheduler
infra/outputs.tf
infra/.gitignore
```

`sources/` is its own package because a GitHub source lands there in a later plan; keeping S3 access out of `indexer/` means that addition is a new file rather than an edit.

---

### Task 1: S3 source adapter

**Files:**
- Create: `src/notes_rag/sources/__init__.py`, `src/notes_rag/sources/s3.py`
- Test: `tests/sources/__init__.py`, `tests/sources/test_s3.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `S3Object` frozen dataclass with `key: str`, `etag: str`. Functions `list_objects(client, bucket, prefixes) -> list[S3Object]`, `get_bytes(client, bucket, key) -> bytes`, `get_json(client, bucket, key) -> dict | None`, `put_json(client, bucket, key, payload) -> None`, `download_file(client, bucket, key, dest) -> bool`, `upload_file(client, bucket, key, src) -> None`.

**Design note:** every function takes an explicit client as its first argument, so tests stub it directly and never need `moto` or credentials. ETags come back from S3 wrapped in literal double quotes; they are stripped on the way in so the manifest stores a stable value.

- [ ] **Step 1: Add the shared S3 stub to the root conftest**

`tests/` is not a package (it has no `__init__.py`), so a test module cannot reliably import a helper from a sibling test module. Task 4 needs this same stub, so it goes in `tests/conftest.py`, which pytest loads automatically and which needs no import at all.

Append to `tests/conftest.py`:

```python
class _NoSuchKey(Exception):
    pass


class StubS3:
    """Minimal stand-in for a boto3 S3 client.

    Only the three calls the adapter makes are implemented. `exceptions.NoSuchKey`
    mirrors botocore's generated exception class, which the adapter catches by
    attribute off the client rather than by importing botocore.
    """

    class exceptions:  # noqa: N801 - mirrors the boto3 client attribute name
        NoSuchKey = _NoSuchKey

    def __init__(self, objects: dict[str, bytes] | None = None, page_size: int = 1000) -> None:
        self.objects = dict(objects or {})
        self.page_size = page_size

    def list_objects_v2(self, **kwargs):
        prefix = kwargs["Prefix"]
        token = kwargs.get("ContinuationToken")
        keys = sorted(k for k in self.objects if k.startswith(prefix))
        start = int(token) if token is not None else 0
        page = keys[start : start + self.page_size]
        contents = [{"Key": key, "ETag": f'"etag-of-{key}"'} for key in page]
        truncated = start + self.page_size < len(keys)
        response = {"Contents": contents, "IsTruncated": truncated}
        if truncated:
            response["NextContinuationToken"] = str(start + self.page_size)
        return response

    def get_object(self, **kwargs):
        key = kwargs["Key"]
        if key not in self.objects:
            raise _NoSuchKey(key)

        class _Body:
            def __init__(self, raw: bytes) -> None:
                self._raw = raw

            def read(self) -> bytes:
                return self._raw

        return {"Body": _Body(self.objects[key])}

    def put_object(self, **kwargs):
        self.objects[kwargs["Key"]] = kwargs["Body"]
        return {}


@pytest.fixture
def make_s3():
    """Factory for StubS3, so tests never import across test modules."""

    def _make(objects: dict[str, bytes] | None = None, page_size: int = 1000) -> StubS3:
        return StubS3(objects, page_size)

    return _make
```

- [ ] **Step 2: Write the failing test**

Create `tests/sources/__init__.py` (empty) and `tests/sources/test_s3.py`:

```python
import json

from notes_rag.sources.s3 import (
    S3Object,
    download_file,
    get_bytes,
    get_json,
    list_objects,
    put_json,
    upload_file,
)


def test_list_objects_returns_keys_and_etags(make_s3):
    client = make_s3({"summaries/a.json": b"{}", "summaries/b.json": b"{}"})
    assert list_objects(client, "bucket", ["summaries/"]) == [
        S3Object(key="summaries/a.json", etag="etag-of-summaries/a.json"),
        S3Object(key="summaries/b.json", etag="etag-of-summaries/b.json"),
    ]


def test_list_objects_strips_the_quotes_s3_wraps_around_etags(make_s3):
    client = make_s3({"summaries/a.json": b"{}"})
    assert not list_objects(client, "bucket", ["summaries/"])[0].etag.startswith('"')


def test_list_objects_spans_every_prefix(make_s3):
    client = make_s3({"summaries/a.json": b"{}", "transcripts/a.json": b"{}", "other/a.json": b"{}"})
    keys = [o.key for o in list_objects(client, "bucket", ["summaries/", "transcripts/"])]
    assert keys == ["summaries/a.json", "transcripts/a.json"]


def test_list_objects_follows_pagination(make_s3):
    client = make_s3({f"summaries/{i:03d}.json": b"{}" for i in range(2500)}, page_size=1000)
    assert len(list_objects(client, "bucket", ["summaries/"])) == 2500


def test_list_objects_skips_directory_placeholder_keys(make_s3):
    client = make_s3({"summaries/": b"", "summaries/a.json": b"{}"})
    assert [o.key for o in list_objects(client, "bucket", ["summaries/"])] == ["summaries/a.json"]


def test_list_objects_on_an_empty_prefix_returns_empty(make_s3):
    assert list_objects(make_s3(), "bucket", ["summaries/"]) == []


def test_get_bytes_returns_the_body(make_s3):
    client = make_s3({"k": b"payload"})
    assert get_bytes(client, "bucket", "k") == b"payload"


def test_get_json_parses_the_object(make_s3):
    client = make_s3({"m.json": json.dumps({"a": 1}).encode()})
    assert get_json(client, "bucket", "m.json") == {"a": 1}


def test_get_json_returns_none_for_a_missing_key(make_s3):
    # The first ever run has no manifest and no index; missing is expected.
    assert get_json(make_s3(), "bucket", "m.json") is None


def test_put_json_round_trips_through_get_json(make_s3):
    client = make_s3()
    put_json(client, "bucket", "m.json", {"b": 2})
    assert get_json(client, "bucket", "m.json") == {"b": 2}


def test_download_file_writes_the_bytes_and_reports_true(tmp_path, make_s3):
    client = make_s3({"index/full.db": b"sqlite-bytes"})
    dest = tmp_path / "nested" / "full.db"
    assert download_file(client, "bucket", "index/full.db", dest) is True
    assert dest.read_bytes() == b"sqlite-bytes"


def test_download_file_reports_false_for_a_missing_key(tmp_path, make_s3):
    dest = tmp_path / "full.db"
    assert download_file(make_s3(), "bucket", "index/full.db", dest) is False
    assert not dest.exists()


def test_upload_file_sends_the_file_bytes(tmp_path, make_s3):
    src = tmp_path / "public.db"
    src.write_bytes(b"db-bytes")
    client = make_s3()
    upload_file(client, "bucket", "index/public.db", src)
    assert client.objects["index/public.db"] == b"db-bytes"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/sources/test_s3.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'notes_rag.sources'`

- [ ] **Step 4: Write the implementation**

Create `src/notes_rag/sources/__init__.py` (empty file).

Create `src/notes_rag/sources/s3.py`:

```python
"""S3 access for the indexer: listing with ETags, and moving bytes.

Deliberately free of indexing logic. Every function takes an explicit
boto3-style client as its first argument so tests can stub it without moto and
without credentials.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class S3Object:
    key: str
    etag: str


def _strip_quotes(etag: str) -> str:
    """S3 returns ETags wrapped in literal double quotes.

    Stripping them here means the manifest stores one stable representation,
    so a round trip through JSON never produces a spurious diff.
    """
    return etag.strip('"')


def list_objects(client, bucket: str, prefixes: Sequence[str]) -> list[S3Object]:
    """Every object under each prefix, sorted by key, with quote-free ETags.

    Directory placeholder keys (those ending in "/") are skipped: the console
    creates them and they carry no content.
    """
    found: list[S3Object] = []
    for prefix in prefixes:
        token: str | None = None
        while True:
            kwargs: dict = {"Bucket": bucket, "Prefix": prefix}
            if token is not None:
                kwargs["ContinuationToken"] = token
            response = client.list_objects_v2(**kwargs)
            for item in response.get("Contents") or []:
                if item["Key"].endswith("/"):
                    continue
                found.append(S3Object(key=item["Key"], etag=_strip_quotes(item["ETag"])))
            if not response.get("IsTruncated"):
                break
            token = response["NextContinuationToken"]
    return sorted(found, key=lambda obj: obj.key)


def get_bytes(client, bucket: str, key: str) -> bytes:
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


def get_json(client, bucket: str, key: str) -> dict | None:
    """Parsed JSON at `key`, or None when the key does not exist.

    The first ever run has no manifest and no index, so a missing key is an
    expected state rather than an error.
    """
    try:
        raw = get_bytes(client, bucket, key)
    except client.exceptions.NoSuchKey:
        return None
    return json.loads(raw)


def put_json(client, bucket: str, key: str, payload: dict) -> None:
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2, sort_keys=True).encode(),
        ContentType="application/json",
    )


def download_file(client, bucket: str, key: str, dest: Path) -> bool:
    """Write `key` to `dest`. Returns False when the key does not exist."""
    try:
        raw = get_bytes(client, bucket, key)
    except client.exceptions.NoSuchKey:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    return True


def upload_file(client, bucket: str, key: str, src: Path) -> None:
    client.put_object(Bucket=bucket, Key=key, Body=src.read_bytes())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/sources/test_s3.py -v`
Expected: 13 passed

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff check src tests eval && .venv/bin/ruff format src tests eval
git add src/notes_rag/sources tests/conftest.py tests/sources
git commit -m "feat: add S3 source adapter with ETag listing"
```

---

### Task 2: ETag manifest and diff

**Files:**
- Create: `src/notes_rag/indexer/manifest.py`
- Test: `tests/indexer/test_manifest.py`

**Interfaces:**
- Consumes: `S3Object` from `notes_rag.sources.s3`.
- Produces: `MANIFEST_VERSION: int`; `ManifestDiff` frozen dataclass with `changed: tuple[str, ...]`, `removed: tuple[str, ...]`, property `is_empty: bool`; `Manifest` frozen dataclass with `etags: dict[str, str]`, classmethods `empty()`, `from_dict(payload)`, `of(objects)`, methods `to_dict()`, `diff(objects) -> ManifestDiff`.

**Design note:** pure data, no IO. This is the component the no-op path depends on, so it is worth testing exhaustively — a diff that wrongly reports "changed" turns 8,639 free runs into 8,639 paid rebuilds.

- [ ] **Step 1: Write the failing test**

Create `tests/indexer/test_manifest.py`:

```python
from notes_rag.indexer.manifest import MANIFEST_VERSION, Manifest, ManifestDiff
from notes_rag.sources.s3 import S3Object


def obj(key: str, etag: str) -> S3Object:
    return S3Object(key=key, etag=etag)


def test_empty_manifest_reports_everything_as_changed():
    diff = Manifest.empty().diff([obj("a", "1"), obj("b", "2")])
    assert diff.changed == ("a", "b")
    assert diff.removed == ()


def test_identical_etags_produce_an_empty_diff():
    current = [obj("a", "1"), obj("b", "2")]
    diff = Manifest.of(current).diff(current)
    assert diff.changed == ()
    assert diff.removed == ()
    assert diff.is_empty is True


def test_a_changed_etag_is_reported_as_changed():
    diff = Manifest.of([obj("a", "1")]).diff([obj("a", "2")])
    assert diff.changed == ("a",)
    assert diff.is_empty is False


def test_a_new_key_is_reported_as_changed():
    diff = Manifest.of([obj("a", "1")]).diff([obj("a", "1"), obj("b", "2")])
    assert diff.changed == ("b",)


def test_a_vanished_key_is_reported_as_removed():
    diff = Manifest.of([obj("a", "1"), obj("b", "2")]).diff([obj("a", "1")])
    assert diff.changed == ()
    assert diff.removed == ("b",)
    assert diff.is_empty is False


def test_changed_and_removed_are_both_reported_in_one_diff():
    diff = Manifest.of([obj("a", "1"), obj("b", "2")]).diff([obj("a", "9"), obj("c", "3")])
    assert diff.changed == ("a", "c")
    assert diff.removed == ("b",)


def test_diff_against_nothing_reports_every_known_key_removed():
    diff = Manifest.of([obj("a", "1"), obj("b", "2")]).diff([])
    assert diff.removed == ("a", "b")


def test_empty_manifest_against_empty_listing_is_empty():
    assert Manifest.empty().diff([]).is_empty is True


def test_to_dict_round_trips_through_from_dict():
    manifest = Manifest.of([obj("a", "1"), obj("b", "2")])
    assert Manifest.from_dict(manifest.to_dict()) == manifest


def test_to_dict_records_the_schema_version():
    assert Manifest.of([obj("a", "1")]).to_dict()["version"] == MANIFEST_VERSION


def test_from_dict_treats_none_as_an_empty_manifest():
    # get_json returns None when the manifest key does not exist yet.
    assert Manifest.from_dict(None) == Manifest.empty()


def test_from_dict_tolerates_a_payload_with_no_etags_key():
    assert Manifest.from_dict({"version": MANIFEST_VERSION}) == Manifest.empty()


def test_manifest_diff_is_empty_only_when_both_lists_are_empty():
    assert ManifestDiff(changed=(), removed=()).is_empty is True
    assert ManifestDiff(changed=("a",), removed=()).is_empty is False
    assert ManifestDiff(changed=(), removed=("a",)).is_empty is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/indexer/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'notes_rag.indexer.manifest'`

- [ ] **Step 3: Write the implementation**

Create `src/notes_rag/indexer/manifest.py`:

```python
"""What the previous run saw, so this run can tell whether anything changed.

Pure data - no IO. The manifest maps every source object key to the ETag it had
when the index was last built. Comparing it to a fresh listing is what lets the
overwhelmingly common no-op run exit in milliseconds, before downloading the
index or calling Bedrock.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from notes_rag.sources.s3 import S3Object

MANIFEST_VERSION = 1


@dataclass(frozen=True)
class ManifestDiff:
    changed: tuple[str, ...]
    removed: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return not self.changed and not self.removed


@dataclass(frozen=True)
class Manifest:
    etags: dict[str, str]

    @classmethod
    def empty(cls) -> "Manifest":
        return cls(etags={})

    @classmethod
    def of(cls, objects: Sequence[S3Object]) -> "Manifest":
        """The manifest describing a listing - what gets written after a build."""
        return cls(etags={obj.key: obj.etag for obj in objects})

    @classmethod
    def from_dict(cls, payload: dict | None) -> "Manifest":
        """Parse a stored manifest. `None` (missing key) means the first run."""
        if not payload:
            return cls.empty()
        return cls(etags=dict(payload.get("etags") or {}))

    def to_dict(self) -> dict:
        return {"version": MANIFEST_VERSION, "etags": dict(self.etags)}

    def diff(self, objects: Sequence[S3Object]) -> ManifestDiff:
        """What differs between this manifest and a fresh listing.

        `changed` covers both new keys and keys whose ETag moved - the indexer
        treats them identically. `removed` is what the listing no longer has.
        """
        current = {obj.key: obj.etag for obj in objects}
        changed = tuple(sorted(k for k, etag in current.items() if self.etags.get(k) != etag))
        removed = tuple(sorted(set(self.etags) - set(current)))
        return ManifestDiff(changed=changed, removed=removed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/indexer/test_manifest.py -v`
Expected: 13 passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src tests eval && .venv/bin/ruff format src tests eval
git add src/notes_rag/indexer/manifest.py tests/indexer/test_manifest.py
git commit -m "feat: add ETag manifest with change detection"
```

---

### Task 3: Extract shared document collection

**Files:**
- Create: `src/notes_rag/indexer/collect.py`
- Modify: `src/notes_rag/indexer/cli.py` (replace `_collect` and `_build_chunks` with calls into `collect.py`)
- Test: `tests/indexer/test_collect.py`

**Interfaces:**
- Consumes: `Chunk` from `notes_rag.models`; `chunk_video_summary`, `chunk_video_transcript`, `chunk_markdown` from `notes_rag.chunkers.*`.
- Produces: `SourceDocument` frozen dataclass with `source_path: str`, `raw: bytes`; `CollectedDocuments` frozen dataclass with `summaries: tuple[tuple[dict, str], ...]`, `transcripts: tuple[tuple[dict, str], ...]`, `markdown_notes: tuple[tuple[str, str], ...]`, `skipped: tuple[tuple[str, str], ...]`; `classify(documents) -> CollectedDocuments`; `build_chunks(collected, *, vault_id) -> tuple[list[Chunk], tuple[tuple[str, str], ...]]`.

**Why this refactor:** the Lambda needs exactly the shape-dispatch and transcript/summary pairing that `cli.py` already has, but over bytes fetched from S3 rather than files on disk. Copying it would be two implementations of the dedupe and pairing rules. Moving it behind `SourceDocument` lets the CLI read files and the Lambda read S3 objects into the same type.

**Behaviour change, deliberate:** the current code prints skips to stderr from inside the classification logic. The extracted version *returns* skips as `(source_path, reason)` pairs, and the CLI prints them. A library that writes to stderr cannot be tested on what it skipped, and the Lambda needs to log skips rather than print them.

- [ ] **Step 1: Write the failing test**

Create `tests/indexer/test_collect.py`:

```python
import json

from notes_rag.indexer.collect import CollectedDocuments, SourceDocument, build_chunks, classify

SUMMARY = {
    "video_id": "vid1",
    "title": "T",
    "channel": "C",
    "url": "https://example.com/watch?v=vid1",
    "summary": {
        "verdict": "v",
        "tldr": "t",
        "takeaways": ["a"],
        "sections": [{"start_seconds": 0, "title": "Intro", "summary": "s"}],
    },
}
TRANSCRIPT = {
    "video_id": "vid1",
    "language": "en",
    "segments": [{"start_seconds": 0, "text": "hello there"}],
}


def doc(path: str, payload) -> SourceDocument:
    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return SourceDocument(source_path=path, raw=raw)


def test_classifies_a_summary_by_its_summary_object():
    collected = classify([doc("summaries/vid1.json", SUMMARY)])
    assert [path for _, path in collected.summaries] == ["summaries/vid1.json"]
    assert collected.transcripts == ()


def test_classifies_a_transcript_by_its_segments_list():
    collected = classify([doc("transcripts/vid1.json", TRANSCRIPT)])
    assert [path for _, path in collected.transcripts] == ["transcripts/vid1.json"]
    assert collected.summaries == ()


def test_classifies_markdown_by_suffix():
    collected = classify([SourceDocument(source_path="notes/a.md", raw=b"# hi")])
    assert [path for _, path in collected.markdown_notes] == ["notes/a.md"]


def test_skips_json_whose_top_level_is_not_an_object():
    collected = classify([doc("summaries/bad.json", [1, 2, 3])])
    assert collected.summaries == ()
    assert [path for path, _ in collected.skipped] == ["summaries/bad.json"]


def test_skips_json_of_an_unrecognized_shape():
    collected = classify([doc("summaries/odd.json", {"nothing": "useful"})])
    assert [path for path, _ in collected.skipped] == ["summaries/odd.json"]


def test_skips_malformed_json_rather_than_raising():
    collected = classify([doc("summaries/broken.json", b"{not json")])
    assert [path for path, _ in collected.skipped] == ["summaries/broken.json"]


def test_skips_a_file_with_an_unhandled_suffix():
    collected = classify([SourceDocument(source_path="notes/photo.png", raw=b"\x89PNG")])
    assert [path for path, _ in collected.skipped] == ["notes/photo.png"]


def test_every_skip_carries_a_reason():
    collected = classify([doc("summaries/broken.json", b"{not json")])
    assert all(reason for _, reason in collected.skipped)


def test_classify_of_nothing_is_empty():
    collected = classify([])
    assert collected == CollectedDocuments(
        summaries=(), transcripts=(), markdown_notes=(), skipped=()
    )


def test_build_chunks_emits_summary_and_transcript_chunks():
    collected = classify(
        [doc("summaries/vid1.json", SUMMARY), doc("transcripts/vid1.json", TRANSCRIPT)]
    )
    chunks, skipped = build_chunks(collected, vault_id="V")
    assert skipped == ()
    assert {chunk.chunk_type for chunk in chunks} == {"summary", "transcript"}
    assert all(chunk.video_id == "vid1" for chunk in chunks)


def test_build_chunks_pairs_a_transcript_with_its_summary_by_video_id():
    other_summary = dict(SUMMARY, video_id="vid2", url="https://example.com/watch?v=vid2")
    collected = classify(
        [
            doc("summaries/vid1.json", SUMMARY),
            doc("summaries/vid2.json", other_summary),
            doc("transcripts/vid2.json", dict(TRANSCRIPT, video_id="vid2")),
        ]
    )
    chunks, _ = build_chunks(collected, vault_id="V")
    transcript_chunks = [c for c in chunks if c.chunk_type == "transcript"]
    assert transcript_chunks
    assert all(c.video_id == "vid2" for c in transcript_chunks)


def test_build_chunks_skips_a_transcript_with_no_matching_summary():
    collected = classify([doc("transcripts/orphan.json", dict(TRANSCRIPT, video_id="missing"))])
    chunks, skipped = build_chunks(collected, vault_id="V")
    assert chunks == []
    assert [path for path, _ in skipped] == ["transcripts/orphan.json"]
    assert "summary" in skipped[0][1]


def test_build_chunks_applies_vault_id_to_markdown_chunks():
    collected = classify([SourceDocument(source_path="notes/a.md", raw=b"# hi\n\nbody")])
    chunks, _ = build_chunks(collected, vault_id="Class Notes")
    assert all(chunk.vault_id == "Class Notes" for chunk in chunks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/indexer/test_collect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'notes_rag.indexer.collect'`

- [ ] **Step 3: Write the implementation**

Create `src/notes_rag/indexer/collect.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/indexer/test_collect.py -v`
Expected: 13 passed

- [ ] **Step 5: Rewire the CLI onto the shared module**

In `src/notes_rag/indexer/cli.py`, delete the `_collect` and `_build_chunks` functions and the now-unused imports (`json`, `chunk_markdown`, `chunk_video_summary`, `chunk_video_transcript`, `Chunk`, and the `SummaryDoc` / `TranscriptDoc` / `MarkdownDoc` aliases). Replace them with a file reader, and update `main` to use it.

Add this import block alongside the existing ones:

```python
from notes_rag.indexer.collect import SourceDocument, build_chunks, classify
```

Add this function in place of the deleted pair:

```python
def _read_documents(source: Path) -> list[SourceDocument]:
    """Every file under `source`, as SourceDocuments with source-relative paths.

    `source_path` is posix-separated because it ends up in `Chunk.source_path`,
    which must match the S3 key layout the indexer Lambda produces.
    """
    return [
        SourceDocument(source_path=path.relative_to(source).as_posix(), raw=path.read_bytes())
        for path in sorted(source.rglob("*"))
        if path.is_file()
    ]
```

In `main`, replace these two lines:

```python
    summaries, transcripts, markdown_notes = _collect(Path(args.source))
    chunks = _build_chunks(summaries, transcripts, markdown_notes, vault_id=args.vault_id)
```

with:

```python
    collected = classify(_read_documents(Path(args.source)))
    chunks, unpairable = build_chunks(collected, vault_id=args.vault_id)
    for path, reason in collected.skipped + unpairable:
        print(f"skipping {path}: {reason}", file=sys.stderr)
```

- [ ] **Step 6: Verify the CLI is unchanged in behaviour**

Run: `.venv/bin/pytest tests/indexer/test_cli.py -v`
Expected: every existing CLI test passes untouched — including the orphan-transcript test, which asserts on the `"no summary found"` text now produced by `build_chunks` and printed by `main`.

Then run the full suite: `.venv/bin/pytest`
Expected: all pass, 1 deselected.

- [ ] **Step 7: Lint and commit**

```bash
.venv/bin/ruff check src tests eval && .venv/bin/ruff format src tests eval
git add src/notes_rag/indexer/collect.py src/notes_rag/indexer/cli.py tests/indexer/test_collect.py
git commit -m "refactor: extract document collection shared by CLI and indexer"
```

---

### Task 4: Indexer orchestration

**Files:**
- Create: `src/notes_rag/indexer/handler.py`
- Test: `tests/indexer/test_handler.py`

**Interfaces:**
- Consumes: `list_objects`, `get_json`, `put_json`, `download_file`, `upload_file`, `get_bytes` from `notes_rag.sources.s3`; `Manifest` from `notes_rag.indexer.manifest`; `SourceDocument`, `classify`, `build_chunks` from `notes_rag.indexer.collect`; `build_index`, `derive_backlinks` from `notes_rag.indexer.build`; `SqliteVecStore`; `Embedder`.
- Produces: `IndexerConfig` frozen dataclass with `source_bucket: str`, `source_prefixes: tuple[str, ...]`, `index_bucket: str`, `full_db_key: str`, `public_db_key: str`, `manifest_key: str`, `dimensions: int`, `bedrock_region: str`, `vault_id: str`, `work_dir: str`, plus classmethod `from_env(env)`. `IndexerResult` frozen dataclass with `status: str`, `changed: int`, `removed: int`, `chunks_written: int`, `vectors_embedded: int`, `vectors_reused: int`, plus `to_dict()`. `run_index(config, *, s3, embedder) -> IndexerResult`. `lambda_handler(event, context) -> dict`.

**Design note — the no-op path:** `run_index` must return `status="no-op"` after only the listing and the manifest read. No index download, no Bedrock client construction, no chunking. The test asserts this by giving the stub client an index object and checking it was never fetched.

- [ ] **Step 1: Write the failing test**

Create `tests/indexer/test_handler.py`:

```python
import json

import pytest

from notes_rag.embed.fake import FakeEmbedder
from notes_rag.indexer.handler import IndexerConfig, IndexerResult, run_index
from notes_rag.store.sqlite_vec import SqliteVecStore

DIMS = 8

SUMMARY = {
    "video_id": "vid1",
    "title": "T",
    "channel": "C",
    "url": "https://example.com/watch?v=vid1",
    "summary": {
        "verdict": "v",
        "tldr": "t",
        "takeaways": ["a"],
        "sections": [{"start_seconds": 0, "title": "Intro", "summary": "s"}],
    },
}
TRANSCRIPT = {
    "video_id": "vid1",
    "language": "en",
    "segments": [{"start_seconds": 0, "text": "hello there"}],
}


def config(tmp_path) -> IndexerConfig:
    return IndexerConfig(
        source_bucket="source",
        source_prefixes=("summaries/", "transcripts/"),
        index_bucket="index",
        full_db_key="index/full.db",
        public_db_key="index/public.db",
        manifest_key="index/manifest.json",
        dimensions=DIMS,
        bedrock_region="us-east-2",
        vault_id="V",
        work_dir=str(tmp_path),
    )


def stub_with_sources(make_s3):
    return make_s3(
        {
            "summaries/vid1.json": json.dumps(SUMMARY).encode(),
            "transcripts/vid1.json": json.dumps(TRANSCRIPT).encode(),
        }
    )


def test_first_run_builds_and_uploads_both_artifacts(tmp_path, make_s3):
    client = stub_with_sources(make_s3)
    result = run_index(config(tmp_path), s3=client, embedder=FakeEmbedder(dimensions=DIMS))

    assert result.status == "rebuilt"
    assert result.chunks_written > 0
    assert "index/full.db" in client.objects
    assert "index/public.db" in client.objects
    assert "index/manifest.json" in client.objects


def test_first_run_embeds_everything(tmp_path, make_s3):
    result = run_index(config(tmp_path), s3=stub_with_sources(make_s3), embedder=FakeEmbedder(DIMS))
    assert result.vectors_embedded == result.chunks_written
    assert result.vectors_reused == 0


def test_second_run_with_unchanged_sources_is_a_no_op(tmp_path, make_s3):
    client = stub_with_sources(make_s3)
    run_index(config(tmp_path), s3=client, embedder=FakeEmbedder(dimensions=DIMS))

    result = run_index(config(tmp_path), s3=client, embedder=FakeEmbedder(dimensions=DIMS))
    assert result.status == "no-op"
    assert result.chunks_written == 0
    assert result.vectors_embedded == 0


def test_the_no_op_path_never_downloads_the_index(tmp_path, make_s3):
    """This is the whole point of the manifest: ~8,639 of ~8,640 monthly runs
    must exit before touching the index or Bedrock."""
    client = stub_with_sources(make_s3)
    run_index(config(tmp_path), s3=client, embedder=FakeEmbedder(dimensions=DIMS))

    fetched: list[str] = []
    original_get_object = client.get_object

    def recording_get_object(**kwargs):
        fetched.append(kwargs["Key"])
        return original_get_object(**kwargs)

    client.get_object = recording_get_object
    run_index(config(tmp_path), s3=client, embedder=FakeEmbedder(dimensions=DIMS))

    assert "index/full.db" not in fetched
    assert fetched == ["index/manifest.json"]


def test_a_changed_source_triggers_a_rebuild_that_reuses_cached_vectors(tmp_path, make_s3):
    client = stub_with_sources(make_s3)
    run_index(config(tmp_path), s3=client, embedder=FakeEmbedder(dimensions=DIMS))

    changed = dict(SUMMARY)
    changed["summary"] = dict(SUMMARY["summary"], tldr="a different tldr entirely")
    client.objects["summaries/vid1.json"] = json.dumps(changed).encode()

    result = run_index(config(tmp_path), s3=client, embedder=FakeEmbedder(dimensions=DIMS))
    assert result.status == "rebuilt"
    assert result.vectors_reused > 0, "transcript chunks were unchanged and should be cached"


def test_a_removed_source_drops_its_chunks_from_the_index(tmp_path, make_s3):
    client = stub_with_sources(make_s3)
    run_index(config(tmp_path), s3=client, embedder=FakeEmbedder(dimensions=DIMS))

    del client.objects["transcripts/vid1.json"]
    result = run_index(config(tmp_path), s3=client, embedder=FakeEmbedder(dimensions=DIMS))

    assert result.status == "rebuilt"
    local = tmp_path / "verify.db"
    local.write_bytes(client.objects["index/full.db"])
    store = SqliteVecStore(local, dimensions=DIMS)
    try:
        assert store.all_source_paths() == {"summaries/vid1.json"}
    finally:
        store.close()


def test_public_db_contains_only_video_corpus_chunks(tmp_path, make_s3):
    client = stub_with_sources(make_s3)
    run_index(config(tmp_path), s3=client, embedder=FakeEmbedder(dimensions=DIMS))

    local = tmp_path / "public.db"
    local.write_bytes(client.objects["index/public.db"])
    store = SqliteVecStore(local, dimensions=DIMS)
    try:
        corpora = {row[0] for row in store._db.execute("SELECT DISTINCT corpus FROM chunks")}
        assert corpora == {"video"}
    finally:
        store.close()


def test_manifest_records_every_source_etag(tmp_path, make_s3):
    client = stub_with_sources(make_s3)
    run_index(config(tmp_path), s3=client, embedder=FakeEmbedder(dimensions=DIMS))
    manifest = json.loads(client.objects["index/manifest.json"])
    assert set(manifest["etags"]) == {"summaries/vid1.json", "transcripts/vid1.json"}


def test_run_with_no_sources_at_all_is_a_no_op(tmp_path, make_s3):
    result = run_index(config(tmp_path), s3=make_s3(), embedder=FakeEmbedder(dimensions=DIMS))
    assert result.status == "no-op"


def test_result_serialises_for_the_lambda_response(tmp_path, make_s3):
    result = run_index(config(tmp_path), s3=stub_with_sources(make_s3), embedder=FakeEmbedder(DIMS))
    payload = result.to_dict()
    assert payload["status"] == "rebuilt"
    assert isinstance(payload["chunks_written"], int)


def test_config_reads_every_field_from_the_environment():
    config = IndexerConfig.from_env(
        {
            "SOURCE_BUCKET": "src",
            "SOURCE_PREFIXES": "summaries/,transcripts/",
            "INDEX_BUCKET": "idx",
            "EMBED_DIMENSIONS": "1024",
            "BEDROCK_REGION": "us-east-2",
        }
    )
    assert config.source_bucket == "src"
    assert config.source_prefixes == ("summaries/", "transcripts/")
    assert config.index_bucket == "idx"
    assert config.dimensions == 1024


def test_config_defaults_the_optional_fields():
    config = IndexerConfig.from_env({"SOURCE_BUCKET": "src", "INDEX_BUCKET": "idx"})
    assert config.source_prefixes == ("summaries/", "transcripts/")
    assert config.dimensions == 1024
    assert config.bedrock_region == "us-east-2"
    assert config.full_db_key == "index/full.db"
    assert config.public_db_key == "index/public.db"
    assert config.manifest_key == "index/manifest.json"
    assert config.work_dir == "/tmp"


def test_config_rejects_a_missing_required_variable():
    with pytest.raises(KeyError):
        IndexerConfig.from_env({"INDEX_BUCKET": "idx"})


def test_indexer_result_no_op_helper_is_all_zeros():
    result = IndexerResult.no_op()
    assert result.status == "no-op"
    assert result.chunks_written == 0
    assert result.vectors_embedded == 0
    assert result.vectors_reused == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/indexer/test_handler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'notes_rag.indexer.handler'`

- [ ] **Step 3: Write the implementation**

Create `src/notes_rag/indexer/handler.py`:

```python
"""The indexer Lambda: rebuild the index when the source bucket changes.

Ordering matters. The manifest diff runs first and, when nothing changed,
returns before downloading the index or constructing a Bedrock client - roughly
8,639 of ~8,640 monthly runs take that path and must stay cheap.

When something did change, every source is re-chunked, not just the changed
ones: `build_index` deletes any source path absent from the chunks it is given,
so a partial rebuild would delete the rest of the corpus. Chunking is local and
free; the expensive part is embedding, and that stays incremental because
`build_index` reuses any vector whose content_hash is already stored.
"""

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from notes_rag.embed.base import Embedder
from notes_rag.indexer.build import build_index, derive_backlinks
from notes_rag.indexer.collect import SourceDocument, build_chunks, classify
from notes_rag.indexer.manifest import Manifest
from notes_rag.sources.s3 import (
    download_file,
    get_bytes,
    get_json,
    list_objects,
    put_json,
    upload_file,
)
from notes_rag.store.sqlite_vec import SqliteVecStore

logger = logging.getLogger(__name__)

DEFAULT_PREFIXES = ("summaries/", "transcripts/")


@dataclass(frozen=True)
class IndexerConfig:
    source_bucket: str
    index_bucket: str
    source_prefixes: tuple[str, ...] = DEFAULT_PREFIXES
    full_db_key: str = "index/full.db"
    public_db_key: str = "index/public.db"
    manifest_key: str = "index/manifest.json"
    dimensions: int = 1024
    bedrock_region: str = "us-east-2"
    vault_id: str = "Vault"
    work_dir: str = "/tmp"

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "IndexerConfig":
        """Build config from environment variables.

        SOURCE_BUCKET and INDEX_BUCKET are required; a missing one raises
        KeyError at cold start, which is the right time to find out.
        """
        prefixes = env.get("SOURCE_PREFIXES")
        return cls(
            source_bucket=env["SOURCE_BUCKET"],
            index_bucket=env["INDEX_BUCKET"],
            source_prefixes=(
                tuple(p.strip() for p in prefixes.split(",") if p.strip())
                if prefixes
                else DEFAULT_PREFIXES
            ),
            full_db_key=env.get("FULL_DB_KEY", "index/full.db"),
            public_db_key=env.get("PUBLIC_DB_KEY", "index/public.db"),
            manifest_key=env.get("MANIFEST_KEY", "index/manifest.json"),
            dimensions=int(env.get("EMBED_DIMENSIONS", "1024")),
            bedrock_region=env.get("BEDROCK_REGION", "us-east-2"),
            vault_id=env.get("VAULT_ID", "Vault"),
            work_dir=env.get("WORK_DIR", "/tmp"),
        )


@dataclass(frozen=True)
class IndexerResult:
    status: str
    changed: int = 0
    removed: int = 0
    chunks_written: int = 0
    vectors_embedded: int = 0
    vectors_reused: int = 0

    @classmethod
    def no_op(cls) -> "IndexerResult":
        return cls(status="no-op")

    def to_dict(self) -> dict:
        return asdict(self)


def run_index(config: IndexerConfig, *, s3, embedder: Embedder) -> IndexerResult:
    """List, diff, and rebuild if anything moved. Returns what happened."""
    objects = list_objects(s3, config.source_bucket, config.source_prefixes)
    previous = Manifest.from_dict(get_json(s3, config.index_bucket, config.manifest_key))
    diff = previous.diff(objects)

    if diff.is_empty:
        logger.info("no source changes; skipping rebuild")
        return IndexerResult.no_op()

    logger.info("rebuilding: %d changed, %d removed", len(diff.changed), len(diff.removed))

    work = Path(config.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    full_db = work / "full.db"
    public_db = work / "public.db"
    for stale in (full_db, public_db):
        stale.unlink(missing_ok=True)

    # The previous index is the embedding cache. Its absence is fine - the first
    # run has none, and build_index simply embeds everything.
    download_file(s3, config.index_bucket, config.full_db_key, full_db)

    documents = [
        SourceDocument(source_path=obj.key, raw=get_bytes(s3, config.source_bucket, obj.key))
        for obj in objects
    ]
    collected = classify(documents)
    chunks, unpairable = build_chunks(collected, vault_id=config.vault_id)
    for path, reason in collected.skipped + unpairable:
        logger.warning("skipping %s: %s", path, reason)

    chunks = derive_backlinks(chunks)

    store = SqliteVecStore(full_db, dimensions=config.dimensions)
    try:
        stats = build_index(chunks, store, embedder)
        store.copy_filtered(public_db, corpus="video")
    finally:
        store.close()

    upload_file(s3, config.index_bucket, config.full_db_key, full_db)
    upload_file(s3, config.index_bucket, config.public_db_key, public_db)
    put_json(s3, config.index_bucket, config.manifest_key, Manifest.of(objects).to_dict())

    return IndexerResult(
        status="rebuilt",
        changed=len(diff.changed),
        removed=len(diff.removed),
        chunks_written=stats.chunks_written,
        vectors_embedded=stats.vectors_embedded,
        vectors_reused=stats.vectors_reused,
    )


def lambda_handler(event, context) -> dict:
    """Entry point. `event` is ignored: the scheduler sends nothing useful, and
    an on-demand `aws lambda invoke` should behave identically."""
    logging.getLogger().setLevel(logging.INFO)

    # Imported here rather than at module scope so that importing this module -
    # which the unit tests do - never constructs an AWS client.
    import boto3

    from notes_rag.embed.bedrock import TitanEmbedder

    config = IndexerConfig.from_env(os.environ)
    result = run_index(
        config,
        s3=boto3.client("s3"),
        embedder=TitanEmbedder(region=config.bedrock_region, dimensions=config.dimensions),
    )
    logger.info("indexer result: %s", result.to_dict())
    return result.to_dict()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/indexer/test_handler.py -v`
Expected: 14 passed

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/pytest`
Expected: all pass, 1 deselected.

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff check src tests eval && .venv/bin/ruff format src tests eval
git add src/notes_rag/indexer/handler.py tests/indexer/test_handler.py
git commit -m "feat: add indexer orchestration with no-op fast path"
```

---

### Task 5: Lambda deployment bundle

**Files:**
- Create: `scripts/build_lambda.sh`
- Modify: `.gitignore` (add `build/`)

**Interfaces:**
- Consumes: `src/notes_rag/**` and the runtime dependencies in `pyproject.toml`.
- Produces: `build/lambda.zip`, a deployment package whose root contains `notes_rag/` plus the third-party wheels. Terraform (Task 7) consumes this path.

**Design note — the packaging facts, all verified 2026-08-05:**
- `uv pip install --python-platform x86_64-manylinux2014 --python-version 3.12 --only-binary :all: --target build/lambda` produces a Linux bundle from any host, no Docker. Measured 18 MB unzipped for `sqlite-vec` + `pysqlite3-binary` + `PyYAML`.
- `pysqlite3-binary` is mandatory (see Global Constraints).
- `boto3` is **excluded**: the runtime provides 1.42.97, which is current. Bundling it would add ~10 MB for no gain.
- The built bundle was confirmed to import and run a `vec0` create/insert/search inside `public.ecr.aws/lambda/python:3.12`.

- [ ] **Step 1: Write the build script**

Create `scripts/build_lambda.sh`:

```bash
#!/usr/bin/env bash
# Build the indexer Lambda deployment package.
#
# Cross-compiles for the Lambda runtime from any host: every dependency ships a
# manylinux wheel, so --only-binary makes this a download-and-unpack rather than
# a build, and no Docker is required.
#
# boto3 is deliberately excluded - the python3.12 runtime provides a current one.
# pysqlite3-binary is deliberately INCLUDED and is not optional: the runtime's
# stdlib sqlite3 has no loadable-extension support, so sqlite-vec cannot load
# without it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build/lambda"
ZIP="$ROOT/build/lambda.zip"

rm -rf "$BUILD" "$ZIP"
mkdir -p "$BUILD"

uv pip install \
  --python-platform x86_64-manylinux2014 \
  --python-version 3.12 \
  --only-binary :all: \
  --target "$BUILD" \
  --link-mode=copy \
  --quiet \
  "sqlite-vec>=0.1.6" \
  "pysqlite3-binary>=0.5" \
  "PyYAML>=6.0"

cp -r "$ROOT/src/notes_rag" "$BUILD/notes_rag"
find "$BUILD" -name '__pycache__' -type d -prune -exec rm -rf {} +

( cd "$BUILD" && zip -qr "$ZIP" . )

echo "built $ZIP ($(du -h "$ZIP" | cut -f1), $(unzip -l "$ZIP" | tail -1 | awk '{print $2}') files)"
```

- [ ] **Step 2: Make it executable and run it**

```bash
chmod +x scripts/build_lambda.sh
./scripts/build_lambda.sh
```

Expected: prints a `built .../build/lambda.zip` line. The zip should be roughly 5-7 MB compressed.

- [ ] **Step 3: Verify the bundle imports inside the real Lambda runtime**

Run:

```bash
docker run --rm -v "$PWD/build/lambda":/opt/bundle:ro \
  --entrypoint /bin/sh public.ecr.aws/lambda/python:3.12 -c '
python - <<EOF
import sys; sys.path.insert(0, "/opt/bundle")
from notes_rag.indexer.handler import IndexerConfig, run_index
from notes_rag.store.sqlite_vec import SQLITE_MODULE
import sqlite_vec
print("handler imports OK; sqlite module =", SQLITE_MODULE)
assert SQLITE_MODULE == "pysqlite3", SQLITE_MODULE
from notes_rag.store.sqlite_vec import SqliteVecStore
store = SqliteVecStore("/tmp/probe.db", dimensions=8)
store.close()
print("SqliteVecStore constructed OK in the lambda runtime")
EOF'
```

Expected: prints `sqlite module = pysqlite3` and `SqliteVecStore constructed OK in the lambda runtime`.

This is the check that would have caught the runtime's missing extension support. If `SQLITE_MODULE` reports `sqlite3`, `pysqlite3-binary` did not make it into the bundle and the deploy will fail at first connection.

- [ ] **Step 4: Ignore build output and commit**

Append to `.gitignore`:

```
build/
```

```bash
git add scripts/build_lambda.sh .gitignore
git commit -m "chore: add Lambda deployment bundle build script"
```

---

### Task 6: Terraform bootstrap — state bucket

**Files:**
- Create: `infra/bootstrap/main.tf`, `infra/bootstrap/.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: an S3 bucket holding Terraform state for the main stack, and an output `state_bucket` naming it. Task 7's `terraform init -backend-config="bucket=..."` consumes that name.

**Design note:** this mirrors the pattern already in `SASE-UARK-Website/infra/bootstrap/`. It is a separate root module with **local state**, run once. Terraform cannot store state in a bucket it is also creating, hence the split. Its own state file is trivial to lose — re-creating the bucket is a one-line `terraform import`.

- [ ] **Step 1: Write the bootstrap stack**

Create `infra/bootstrap/main.tf`:

```hcl
# Creates the bucket that holds Terraform state for the main stack.
#
# Run once, from this directory, with local state. Its own tiny state file stays
# on disk and is gitignored; losing it costs nothing, because re-creating this
# bucket is a one-line import.

terraform {
  required_version = ">= 1.10"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

variable "region" {
  type    = string
  default = "us-east-2"
}

variable "state_bucket" {
  type        = string
  description = "Globally unique name for the Terraform state bucket."
}

resource "aws_s3_bucket" "state" {
  bucket = var.state_bucket
}

resource "aws_s3_bucket_public_access_block" "state" {
  bucket                  = aws_s3_bucket.state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Versioned so a corrupted or truncated state file can be rolled back.
resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "state" {
  bucket = aws_s3_bucket.state.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

output "state_bucket" {
  value = aws_s3_bucket.state.id
}
```

Create `infra/bootstrap/.gitignore`:

```
.terraform/
terraform.tfstate
terraform.tfstate.backup
```

- [ ] **Step 2: Apply it**

```bash
cd infra/bootstrap
terraform init
terraform apply -var="state_bucket=notes-rag-tfstate-207423186995"
cd ../..
```

Expected: creates the bucket and prints `state_bucket = "notes-rag-tfstate-207423186995"`. The account ID suffix keeps the name globally unique.

- [ ] **Step 3: Commit**

```bash
git add infra/bootstrap
git commit -m "feat: add Terraform bootstrap stack for state bucket"
```

---

### Task 7: Terraform main stack — index bucket, IAM, Lambda, scheduler

**Files:**
- Create: `infra/main.tf`, `infra/variables.tf`, `infra/storage.tf`, `infra/iam.tf`, `infra/indexer.tf`, `infra/outputs.tf`, `infra/.gitignore`

**Interfaces:**
- Consumes: `build/lambda.zip` from Task 5; the state bucket from Task 6; `notes_rag.indexer.handler.lambda_handler` from Task 4.
- Produces: an index bucket, an indexer Lambda, and a 5-minute EventBridge Scheduler. Outputs `index_bucket` and `indexer_function_name`, which Task 8 uses.

**Design notes:**
- `reserved_concurrent_executions = 1` — two concurrent runs would race on the index artifact.
- The Lambda's Bedrock grant is scoped to the Titan model ARN only, not `bedrock:*`.
- Source-bucket access is read-only: the indexer must never be able to modify Video Vault's content.
- `ephemeral_storage` is raised to 2048 MB. `/tmp` holds `full.db` plus `public.db`; the spec sizes `full.db` at ~130 MB at 20k chunks, and the default 512 MB leaves less headroom than a 5-minute-cadence job deserves.

- [ ] **Step 1: Write the backend and provider**

Create `infra/main.tf`:

```hcl
terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Native S3 state locking. No DynamoDB table: that requirement is obsolete as
  # of Terraform 1.10. Bucket name is passed with -backend-config at init time,
  # because it is created by the bootstrap stack.
  backend "s3" {
    key          = "notes-rag/terraform.tfstate"
    region       = "us-east-2"
    encrypt      = true
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "notes-rag"
      ManagedBy = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}
```

Create `infra/variables.tf`:

```hcl
variable "region" {
  type    = string
  default = "us-east-2"
}

variable "index_bucket" {
  type        = string
  description = "Globally unique name for the bucket holding full.db and public.db."
}

variable "source_bucket" {
  type        = string
  description = "Video Vault content bucket holding summaries/ and transcripts/."
  default     = "videovaultstack-contentbucket52d4b12c-s0o3jpdq69b4"
}

variable "embed_model_id" {
  type    = string
  default = "amazon.titan-embed-text-v2:0"
}

variable "embed_dimensions" {
  type    = number
  default = 1024
}

variable "schedule_expression" {
  type        = string
  description = "EventBridge Scheduler expression for the indexer."
  default     = "rate(5 minutes)"
}

variable "lambda_zip" {
  type        = string
  description = "Path to the deployment package built by scripts/build_lambda.sh."
  default     = "../build/lambda.zip"
}
```

Create `infra/.gitignore`:

```
.terraform/
.terraform.lock.hcl
terraform.tfstate
terraform.tfstate.backup
```

- [ ] **Step 2: Write the index bucket**

Create `infra/storage.tf`:

```hcl
# Holds full.db, public.db, and the ETag manifest. Private: the demo path in a
# later plan reads public.db through a Lambda, never directly from the browser.
resource "aws_s3_bucket" "index" {
  bucket = var.index_bucket
}

resource "aws_s3_bucket_public_access_block" "index" {
  bucket                  = aws_s3_bucket.index.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "index" {
  bucket = aws_s3_bucket.index.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# One noncurrent version is enough to roll back a bad index build; keeping more
# would grow storage without adding recovery value.
resource "aws_s3_bucket_versioning" "index" {
  bucket = aws_s3_bucket.index.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "index" {
  bucket = aws_s3_bucket.index.id

  rule {
    id     = "expire-old-index-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      newer_noncurrent_versions = 1
      noncurrent_days           = 7
    }
  }
}
```

- [ ] **Step 3: Write the IAM role**

Create `infra/iam.tf`:

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

resource "aws_iam_role" "indexer" {
  name               = "notes-rag-indexer"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "indexer_logs" {
  role       = aws_iam_role.indexer.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "indexer" {
  # Read-only on the Video Vault bucket. The indexer must never be able to
  # modify the corpus it is indexing.
  statement {
    sid       = "ReadSourceBucket"
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [
      "arn:aws:s3:::${var.source_bucket}",
      "arn:aws:s3:::${var.source_bucket}/*",
    ]
  }

  statement {
    sid       = "ReadWriteIndexBucket"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [
      aws_s3_bucket.index.arn,
      "${aws_s3_bucket.index.arn}/*",
    ]
  }

  # Scoped to the one embedding model, not bedrock:*.
  statement {
    sid       = "InvokeTitanEmbeddings"
    actions   = ["bedrock:InvokeModel"]
    resources = [
      "arn:aws:bedrock:${var.region}::foundation-model/${var.embed_model_id}",
    ]
  }
}

resource "aws_iam_role_policy" "indexer" {
  name   = "notes-rag-indexer"
  role   = aws_iam_role.indexer.id
  policy = data.aws_iam_policy_document.indexer.json
}
```

- [ ] **Step 4: Write the Lambda and scheduler**

Create `infra/indexer.tf`:

```hcl
resource "aws_lambda_function" "indexer" {
  function_name = "notes-rag-indexer"
  role          = aws_iam_role.indexer.arn
  handler       = "notes_rag.indexer.handler.lambda_handler"
  runtime       = "python3.12"
  architectures = ["x86_64"]

  filename         = var.lambda_zip
  source_code_hash = filebase64sha256(var.lambda_zip)

  timeout     = 300
  memory_size = 1024

  # /tmp holds full.db plus public.db. The spec sizes full.db at ~130MB at 20k
  # chunks; the 512MB default is thinner headroom than a 5-minute job deserves.
  ephemeral_storage {
    size = 2048
  }

  # Two concurrent runs would race on the index artifact.
  reserved_concurrent_executions = 1

  environment {
    variables = {
      SOURCE_BUCKET    = var.source_bucket
      SOURCE_PREFIXES  = "summaries/,transcripts/"
      INDEX_BUCKET     = aws_s3_bucket.index.id
      EMBED_DIMENSIONS = tostring(var.embed_dimensions)
      BEDROCK_REGION   = var.region
    }
  }
}

resource "aws_cloudwatch_log_group" "indexer" {
  name              = "/aws/lambda/${aws_lambda_function.indexer.function_name}"
  retention_in_days = 14
}

data "aws_iam_policy_document" "scheduler_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["scheduler.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "scheduler" {
  name               = "notes-rag-indexer-scheduler"
  assume_role_policy = data.aws_iam_policy_document.scheduler_assume.json
}

data "aws_iam_policy_document" "scheduler_invoke" {
  statement {
    actions   = ["lambda:InvokeFunction"]
    resources = [aws_lambda_function.indexer.arn]
  }
}

resource "aws_iam_role_policy" "scheduler_invoke" {
  name   = "invoke-indexer"
  role   = aws_iam_role.scheduler.id
  policy = data.aws_iam_policy_document.scheduler_invoke.json
}

resource "aws_scheduler_schedule" "indexer" {
  name = "notes-rag-indexer"

  flexible_time_window {
    mode = "OFF"
  }

  schedule_expression          = var.schedule_expression
  schedule_expression_timezone = "UTC"

  target {
    arn      = aws_lambda_function.indexer.arn
    role_arn = aws_iam_role.scheduler.arn

    # The handler ignores the event; an empty payload keeps the scheduled path
    # and an on-demand `aws lambda invoke` identical.
    input = jsonencode({})

    retry_policy {
      # The next tick is 5 minutes away and does the same work, so retrying a
      # failed run buys nothing.
      maximum_retry_attempts = 0
    }
  }
}
```

Create `infra/outputs.tf`:

```hcl
output "index_bucket" {
  value = aws_s3_bucket.index.id
}

output "indexer_function_name" {
  value = aws_lambda_function.indexer.function_name
}

output "indexer_role_arn" {
  value = aws_iam_role.indexer.arn
}
```

- [ ] **Step 5: Validate without applying**

```bash
./scripts/build_lambda.sh
cd infra
terraform init -backend-config="bucket=notes-rag-tfstate-207423186995"
terraform validate
terraform plan -var="index_bucket=notes-rag-index-207423186995"
cd ..
```

Expected: `Success! The configuration is valid.` and a plan creating roughly 14 resources. Review the plan; do not apply yet.

- [ ] **Step 6: Commit**

```bash
git add infra/main.tf infra/variables.tf infra/storage.tf infra/iam.tf infra/indexer.tf infra/outputs.tf infra/.gitignore
git commit -m "feat: add Terraform stack for indexer Lambda and scheduler"
```

---

### Task 8: Deploy and verify against real AWS

**Files:**
- Create: `tests/indexer/test_handler_integration.py`
- Modify: `README.md` (add a deployment section)

**Interfaces:**
- Consumes: everything from Tasks 1-7.
- Produces: a deployed, scheduled indexer, and an integration test that proves the deployed function behaves as designed.

**This task makes real AWS calls and creates real resources.** Every test in it is `@pytest.mark.integration` and deselected by default.

- [ ] **Step 1: Apply the stack**

```bash
cd infra
terraform apply -var="index_bucket=notes-rag-index-207423186995"
cd ..
```

Expected: creates the resources and prints `index_bucket` and `indexer_function_name`.

- [ ] **Step 2: Invoke once by hand and read the result**

```bash
aws lambda invoke --function-name notes-rag-indexer \
  --region us-east-2 --cli-binary-format raw-in-base64-out \
  --payload '{}' /dev/stdout | head -20
```

Expected: `{"status": "rebuilt", "changed": 4, "removed": 0, "chunks_written": 24, "vectors_embedded": 24, "vectors_reused": 0}`

The counts come from the two real videos currently in the bucket: 4 source objects, 24 chunks. If `chunks_written` is 0, the source bucket or prefixes are wrong.

- [ ] **Step 3: Invoke again and confirm the no-op path**

```bash
aws lambda invoke --function-name notes-rag-indexer \
  --region us-east-2 --cli-binary-format raw-in-base64-out \
  --payload '{}' /dev/stdout | head -20
```

Expected: `{"status": "no-op", "changed": 0, "removed": 0, "chunks_written": 0, "vectors_embedded": 0, "vectors_reused": 0}`

Then check the duration of that invocation:

```bash
aws logs tail /aws/lambda/notes-rag-indexer --region us-east-2 --since 5m \
  --format short | grep "REPORT" | tail -2
```

Expected: the no-op invocation's `Duration` is well under a second. If it is multiple seconds, the fast path is downloading the index and the manifest logic is wrong.

- [ ] **Step 4: Write the integration test**

Create `tests/indexer/test_handler_integration.py`:

```python
"""Integration checks against the deployed indexer. Deselected by default.

Run with:
    .venv/bin/pytest -m integration tests/indexer/test_handler_integration.py
"""

import json
import os

import pytest

pytestmark = pytest.mark.integration

REGION = "us-east-2"
FUNCTION = os.environ.get("INDEXER_FUNCTION", "notes-rag-indexer")
INDEX_BUCKET = os.environ.get("INDEX_BUCKET", "notes-rag-index-207423186995")


def invoke() -> dict:
    import boto3

    client = boto3.client("lambda", region_name=REGION)
    response = client.invoke(
        FunctionName=FUNCTION, InvocationType="RequestResponse", Payload=b"{}"
    )
    assert response["StatusCode"] == 200
    assert "FunctionError" not in response, response.get("FunctionError")
    return json.loads(response["Payload"].read())


def test_invoking_the_deployed_indexer_returns_a_recognised_status():
    result = invoke()
    assert result["status"] in {"rebuilt", "no-op"}


def test_a_second_invocation_is_a_no_op():
    """Whatever the first call did, the one after it must find nothing changed.

    This is the property the whole cost model rests on: ~8,639 of ~8,640 monthly
    runs must take the cheap path.
    """
    invoke()
    assert invoke()["status"] == "no-op"


def test_both_artifacts_and_the_manifest_exist_in_the_index_bucket():
    import boto3

    client = boto3.client("s3", region_name=REGION)
    keys = {
        item["Key"]
        for item in client.list_objects_v2(Bucket=INDEX_BUCKET, Prefix="index/").get(
            "Contents", []
        )
    }
    assert {"index/full.db", "index/public.db", "index/manifest.json"} <= keys


def test_public_db_is_smaller_than_full_db():
    """public.db is the corpus=video subset. With only video sources indexed today
    they are close in size, but public.db must never be the larger of the two."""
    import boto3

    client = boto3.client("s3", region_name=REGION)
    full = client.head_object(Bucket=INDEX_BUCKET, Key="index/full.db")["ContentLength"]
    public = client.head_object(Bucket=INDEX_BUCKET, Key="index/public.db")["ContentLength"]
    assert public <= full


def test_the_manifest_lists_the_source_objects():
    import boto3

    client = boto3.client("s3", region_name=REGION)
    raw = client.get_object(Bucket=INDEX_BUCKET, Key="index/manifest.json")["Body"].read()
    manifest = json.loads(raw)
    assert manifest["version"] == 1
    assert all(
        key.startswith(("summaries/", "transcripts/")) for key in manifest["etags"]
    )
    assert manifest["etags"], "manifest recorded no source objects"
```

- [ ] **Step 5: Run the integration test**

Run: `.venv/bin/pytest -m integration tests/indexer/test_handler_integration.py -v`
Expected: 5 passed

- [ ] **Step 6: Confirm the schedule is firing**

Wait 6 minutes, then:

```bash
aws logs tail /aws/lambda/notes-rag-indexer --region us-east-2 --since 10m --format short \
  | grep -c "no source changes"
```

Expected: at least 1 — the scheduler fired and took the no-op path unprompted.

- [ ] **Step 7: Verify the built index answers questions**

Download `public.db` and run the existing eval harness against it, proving the cloud-built artifact is equivalent to the locally-built one:

```bash
aws s3 cp "s3://notes-rag-index-207423186995/index/public.db" /tmp/public.db --region us-east-2
.venv/bin/python -m eval.run --index /tmp/public.db --questions eval/questions.yaml --k 6 --min-recall 0.8
```

Expected: `recall@6: 1.000`, `MRR: 0.967`, exit code 0 — matching the local baseline recorded in `eval/questions.yaml`.

- [ ] **Step 8: Write the README**

The repo has no `README.md`. Create one — this is the first thing a portfolio reader opens.

````markdown
# Notes RAG

Ask questions across a semester of study material and get answers with citations
back to the source: for videos, a deep link to the exact timestamp.

Chunkers turn Video Vault summaries and transcripts into `Chunk`s; a shared
normalizer merges, splits, context-prefixes and hashes them; vectors and metadata
live together in one `sqlite-vec` file. An indexer Lambda rebuilds that file on a
schedule, re-embedding only the chunks whose content actually changed.

## Local use

```bash
uv venv .venv --python python3.12
uv pip install --python .venv/bin/python -e '.[dev]'

# Build an index from a directory of Video Vault artifacts
.venv/bin/notes-rag-index ./artifacts --out index.db --vault-id "Class Notes"

# Score retrieval against the golden question set
.venv/bin/python -m eval.run --index index.db --questions eval/questions.yaml --k 6
```

Add `--fake-embedder` to either command to run with the deterministic embedder
and no AWS credentials.

## Tests

```bash
.venv/bin/pytest                  # unit tests; no credentials needed
.venv/bin/pytest -m integration   # touches AWS; needs credentials in us-east-2
```

## Deployment

Requires Terraform >= 1.10 and AWS credentials for `us-east-2`.

```bash
# 1. Once per account: create the Terraform state bucket.
cd infra/bootstrap
terraform init
terraform apply -var="state_bucket=notes-rag-tfstate-<account-id>"
cd ../..

# 2. Build the Lambda bundle. Re-run this before ANY apply that should pick up
#    code changes: Terraform derives source_code_hash from the zip, so skipping
#    it deploys the previous code with no error and no diff.
./scripts/build_lambda.sh

# 3. Deploy.
cd infra
terraform init -backend-config="bucket=notes-rag-tfstate-<account-id>"
terraform apply -var="index_bucket=notes-rag-index-<account-id>"
```

The indexer then runs every 5 minutes. Trigger one immediately with:

```bash
aws lambda invoke --function-name notes-rag-indexer --region us-east-2 \
  --cli-binary-format raw-in-base64-out --payload '{}' /dev/stdout
```

### Why the bundle ships its own SQLite

AWS Lambda's managed `python3.12` runtime builds the stdlib `sqlite3` **without**
loadable-extension support, so `sqlite-vec` cannot load through it. The bundle
includes `pysqlite3-binary`, which carries its own SQLite with extensions
enabled, and `src/notes_rag/store/sqlite_vec.py` prefers it when present. Dropping
that dependency breaks the indexer on its first database connection.
````

- [ ] **Step 9: Run the full suite, lint, and commit**

```bash
.venv/bin/pytest
.venv/bin/ruff check src tests eval && .venv/bin/ruff format src tests eval
git add tests/indexer/test_handler_integration.py README.md
git commit -m "test: add deployed indexer integration checks"
```

---

## Definition of done

- `.venv/bin/pytest` passes with zero failures and no AWS credentials present.
- `.venv/bin/ruff check src tests eval` and `ruff format --check` are clean.
- `./scripts/build_lambda.sh` produces a bundle whose `SQLITE_MODULE` is `pysqlite3` when imported inside `public.ecr.aws/lambda/python:3.12`.
- `terraform apply` creates the index bucket, the indexer Lambda, and the scheduler.
- A manual invoke returns `status: "rebuilt"` on a changed corpus and `status: "no-op"` on the next call, with the no-op invocation completing in well under a second.
- `.venv/bin/pytest -m integration tests/indexer/test_handler_integration.py` passes.
- `eval/run.py` against the cloud-built `public.db` reproduces the local baseline: recall@6 1.000, MRR 0.967.
- The scheduler has been observed firing on its own.

## What this plan deliberately does not cover

- **Query and generation** — the query Lambda, retrieval over the uploaded artifact, Haiku 4.5 generation, and citation assembly. That is the next plan, and it is where `public.db` finally gets read by something other than a test.
- **The GitHub vault source** — deferred by decision: the only notes repo today holds solely `Video Vault/`, which spec §4.4 excludes from the vault ingester as a dedupe rule. The `SOURCE_PREFIXES` config and the `sources/` package are shaped so adding it later is a new file plus config, not a refactor.
- **Cognito, CloudFront, and the web frontend** — spec build-order step 8.
- **Class-material parsers (docx/pptx/xlsx/pdf) and the S3 ObjectCreated rule** — spec build-order step 9.
- **Gated wikilink expansion** — spec build-order step 10, and still blocked on a golden set large enough to measure it.
