# Vault Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Index the Obsidian vault alongside the video corpus, so `full.db` holds notes, `public.db` does not, and note citations have something real to point at.

**Architecture:** The indexer currently reads one bucket, named by `SOURCE_BUCKET` + `SOURCE_PREFIXES`. This replaces that pair with a JSON source list — each entry a bucket, its prefixes, and (for note sources) a `vault_id`. A new private bucket receives the vault via `aws s3 sync`; Terraform derives both the env var and the IAM grant from one variable, so they cannot drift. Everything downstream already works: `chunk_markdown` sets `corpus="note"`, `public.db`'s `corpus='video'` filter excludes it, and `copy_filtered` strips backlinks.

**Tech Stack:** Python 3.12, boto3, sqlite-vec, Terraform ≥ 1.10, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-08-07-rag-query-design.md` §5.

## Global Constraints

- Python 3.12. Run everything through `uv run` — `python` is not on PATH in this environment.
- Line length 100 (`[tool.ruff]` in `pyproject.toml`). `uv run ruff check .` must pass.
- Full suite must stay green: `uv run pytest -q` → `183 passed, 6 deselected` before this plan, more after. Integration tests stay behind `-m integration` and are deselected by default.
- Every chunk `source_path` is the **full S3 key**, posix-separated. `build_index` deletes by `source_path`, so this is the identity the index is keyed on.
- `content_hash` is computed over the context-prefixed text. Any change to a chunk's `context` is a full re-embed of that corpus. No markdown is indexed today, which is the only reason the `display_path` change in Task 3 is free.
- **Code and Terraform ship in the same `terraform apply`.** This plan renames the Lambda's environment contract (`SOURCE_BUCKET`/`SOURCE_PREFIXES`/`VAULT_ID` → `SOURCE_LIST`). A deploy of one without the other raises `KeyError` at cold start on every invocation.
- Never run `terraform apply -auto-approve`. Use `terraform plan -out=<file>` then `terraform apply <file>`.

## Two spec corrections applied here

1. **§5.3's source-list JSON drops `corpus`.** `chunk_markdown` already sets `corpus="note"` and the video chunkers set `"video"`; corpus is derived from artifact shape. A declared `corpus` field would be a second source of truth that can silently disagree with the first.
2. **Note chunks gain a `display_path`.** `source_path` stays the S3 key (uniqueness, and `build_index` deletes by it), but the embedded context prefix uses a vault-relative path. Without this every note embeds as `joshiosimoe / notes/joshiosimoe/Foo.md / Heading` — the vault name twice plus an S3 prefix inside the vector.

---

### Task 1: Source specification and config

Replace the single-bucket config with a validated list.

**Files:**
- Modify: `src/notes_rag/indexer/handler.py:42-81` (`DEFAULT_PREFIXES`, `IndexerConfig`)
- Test: `tests/indexer/test_config.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `SourceSpec(bucket: str, prefixes: tuple[str, ...], vault_id: str | None)` with `SourceSpec.from_dict(payload: Mapping) -> SourceSpec`; `IndexerConfig.sources: tuple[SourceSpec, ...]`. `IndexerConfig` no longer has `source_bucket`, `source_prefixes`, or `vault_id`. `IndexerConfig.from_env` reads `SOURCE_LIST` (JSON array) instead of `SOURCE_BUCKET`/`SOURCE_PREFIXES`/`VAULT_ID`.

- [ ] **Step 1: Write the failing tests**

Create `tests/indexer/test_config.py`:

```python
import json

import pytest

from notes_rag.indexer.handler import IndexerConfig, SourceSpec


def test_from_dict_reads_a_note_source():
    spec = SourceSpec.from_dict(
        {"bucket": "notes-bucket", "prefixes": ["notes/josh/"], "vault_id": "josh"}
    )
    assert spec.bucket == "notes-bucket"
    assert spec.prefixes == ("notes/josh/",)
    assert spec.vault_id == "josh"


def test_from_dict_leaves_vault_id_none_when_absent():
    spec = SourceSpec.from_dict({"bucket": "video", "prefixes": ["summaries/"]})
    assert spec.vault_id is None


def test_from_dict_treats_explicit_null_vault_id_as_absent():
    # Terraform's jsonencode emits `"vault_id": null` for an unset optional
    # attribute, so this is the shape the deployed Lambda actually receives.
    spec = SourceSpec.from_dict(
        {"bucket": "video", "prefixes": ["summaries/"], "vault_id": None}
    )
    assert spec.vault_id is None


def test_from_dict_rejects_a_prefix_without_a_trailing_slash():
    # "notes" as an s3:prefix condition also matches "notes-private/", so the
    # IAM grant Terraform derives from this list would be wider than intended.
    with pytest.raises(ValueError, match="must end in"):
        SourceSpec.from_dict({"bucket": "b", "prefixes": ["notes"]})


def test_from_dict_rejects_an_empty_prefix_list():
    with pytest.raises(ValueError, match="at least one prefix"):
        SourceSpec.from_dict({"bucket": "b", "prefixes": []})


def test_from_dict_rejects_a_missing_bucket():
    with pytest.raises(ValueError, match="bucket"):
        SourceSpec.from_dict({"prefixes": ["notes/"]})


def test_from_env_parses_the_source_list():
    config = IndexerConfig.from_env(
        {
            "INDEX_BUCKET": "index",
            "SOURCE_LIST": json.dumps(
                [
                    {"bucket": "video", "prefixes": ["summaries/", "transcripts/"]},
                    {"bucket": "notes", "prefixes": ["notes/josh/"], "vault_id": "josh"},
                ]
            ),
        }
    )
    assert [s.bucket for s in config.sources] == ["video", "notes"]
    assert config.sources[1].vault_id == "josh"


def test_from_env_rejects_an_empty_source_list():
    # An empty list lists nothing, so every run is a no-op and the index
    # quietly stops tracking reality. Fail at cold start instead.
    with pytest.raises(ValueError, match="at least one source"):
        IndexerConfig.from_env({"INDEX_BUCKET": "index", "SOURCE_LIST": "[]"})


def test_from_env_requires_source_list():
    with pytest.raises(KeyError):
        IndexerConfig.from_env({"INDEX_BUCKET": "index"})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/indexer/test_config.py -v`
Expected: FAIL — `ImportError: cannot import name 'SourceSpec'`

- [ ] **Step 3: Implement `SourceSpec` and rework `IndexerConfig`**

In `src/notes_rag/indexer/handler.py`, add `import json` to the imports, delete the `DEFAULT_PREFIXES` constant, and replace the `IndexerConfig` block with:

```python
@dataclass(frozen=True)
class SourceSpec:
    """One bucket and the prefixes within it that the indexer reads.

    `vault_id` is required for sources holding markdown and meaningless for
    everything else: it becomes `Chunk.vault_id`, which is the only thing an
    `obsidian://open?vault=` citation can be built from. A markdown document
    that arrives without one is skipped rather than indexed unlinkably - see
    build_chunks.
    """

    bucket: str
    prefixes: tuple[str, ...]
    vault_id: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping) -> "SourceSpec":
        bucket = payload.get("bucket")
        if not bucket:
            raise ValueError(f"source entry needs a bucket: {payload!r}")

        prefixes = tuple(p for p in (payload.get("prefixes") or ()) if p)
        if not prefixes:
            raise ValueError(f"source entry needs at least one prefix: {payload!r}")

        # A prefix without a trailing slash is a real security problem, not a
        # style nit: Terraform derives the IAM s3:prefix condition from this
        # list as "${prefix}*", so "notes" would also grant "notes-private/".
        unslashed = [p for p in prefixes if not p.endswith("/")]
        if unslashed:
            raise ValueError(f"source prefixes must end in '/': {unslashed}")

        return cls(bucket=bucket, prefixes=prefixes, vault_id=payload.get("vault_id") or None)


@dataclass(frozen=True)
class IndexerConfig:
    index_bucket: str
    sources: tuple[SourceSpec, ...]
    full_db_key: str = "index/full.db"
    public_db_key: str = "index/public.db"
    manifest_key: str = "index/manifest.json"
    dimensions: int = 1024
    bedrock_region: str = "us-east-2"
    work_dir: str = "/tmp"

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "IndexerConfig":
        """Build config from environment variables.

        SOURCE_LIST and INDEX_BUCKET are required; a missing one raises at cold
        start, which is the right time to find out. SOURCE_LIST is JSON rather
        than a delimited string because each entry carries three fields, and
        Terraform can hand it over with jsonencode of the same variable the IAM
        policy is derived from - so the grant and the code cannot drift.
        """
        sources = tuple(SourceSpec.from_dict(item) for item in json.loads(env["SOURCE_LIST"]))
        if not sources:
            raise ValueError("SOURCE_LIST must contain at least one source")

        return cls(
            index_bucket=env["INDEX_BUCKET"],
            sources=sources,
            full_db_key=env.get("FULL_DB_KEY", "index/full.db"),
            public_db_key=env.get("PUBLIC_DB_KEY", "index/public.db"),
            manifest_key=env.get("MANIFEST_KEY", "index/manifest.json"),
            dimensions=int(env.get("EMBED_DIMENSIONS", "1024")),
            bedrock_region=env.get("BEDROCK_REGION", "us-east-2"),
            work_dir=env.get("WORK_DIR", "/tmp"),
        )
```

`run_index` still references `config.source_bucket` and `config.source_prefixes` and will not import cleanly yet — Task 4 fixes it. Run only the new test file in Step 4.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/indexer/test_config.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
git add src/notes_rag/indexer/handler.py tests/indexer/test_config.py
git commit -m "feat: describe indexer sources as a validated list, not one bucket"
```

---

### Task 2: Bucket-qualified S3 objects and manifest keys

Two buckets can hold the same key. The manifest is one flat map, so its keys have to say which bucket they came from.

**Files:**
- Modify: `src/notes_rag/sources/s3.py:14-50` (`S3Object`, `list_objects`)
- Modify: `src/notes_rag/indexer/manifest.py:36-59` (`Manifest.of`, `Manifest.diff`)
- Test: `tests/sources/test_s3.py` (modify), `tests/indexer/test_manifest.py` (modify)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `S3Object(bucket: str, key: str, etag: str)` with a `qualified_key` property returning `f"{bucket}/{key}"`. `Manifest.etags` is now keyed by `qualified_key`; `Manifest.diff` returns qualified keys in `changed` and `removed`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/indexer/test_manifest.py`:

```python
def test_manifest_keys_are_bucket_qualified():
    from notes_rag.indexer.manifest import Manifest
    from notes_rag.sources.s3 import S3Object

    manifest = Manifest.of(
        [
            S3Object(bucket="video", key="summaries/a.json", etag="e1"),
            S3Object(bucket="notes", key="notes/josh/a.json", etag="e2"),
        ]
    )
    assert manifest.etags == {
        "video/summaries/a.json": "e1",
        "notes/notes/josh/a.json": "e2",
    }


def test_same_key_in_two_buckets_is_two_entries():
    # Without qualification these collide, and a change to one silently masks
    # the other: the manifest would report no diff and the index would never
    # pick the change up.
    from notes_rag.indexer.manifest import Manifest
    from notes_rag.sources.s3 import S3Object

    objects = [
        S3Object(bucket="a", key="shared.json", etag="e1"),
        S3Object(bucket="b", key="shared.json", etag="e2"),
    ]
    manifest = Manifest.of(objects)
    assert len(manifest.etags) == 2

    moved = [
        S3Object(bucket="a", key="shared.json", etag="e1"),
        S3Object(bucket="b", key="shared.json", etag="CHANGED"),
    ]
    assert manifest.diff(moved).changed == ("b/shared.json",)
```

Append to `tests/sources/test_s3.py`:

```python
def test_list_objects_records_the_bucket(make_s3):
    from notes_rag.sources.s3 import list_objects

    client = make_s3({"summaries/a.json": b"{}"})
    found = list_objects(client, "video-bucket", ["summaries/"])
    assert [obj.bucket for obj in found] == ["video-bucket"]
    assert found[0].qualified_key == "video-bucket/summaries/a.json"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/indexer/test_manifest.py tests/sources/test_s3.py -v`
Expected: FAIL — `TypeError: S3Object.__init__() got an unexpected keyword argument 'bucket'`

- [ ] **Step 3: Implement**

In `src/notes_rag/sources/s3.py`, replace the `S3Object` dataclass:

```python
@dataclass(frozen=True)
class S3Object:
    bucket: str
    key: str
    etag: str

    @property
    def qualified_key(self) -> str:
        """The object's identity in the manifest.

        The manifest is one flat map across every source, and two buckets can
        legitimately hold the same key. Keying on `key` alone would let a
        change to one mask the other, producing an empty diff and an index that
        never catches up.
        """
        return f"{self.bucket}/{self.key}"
```

In the same file, inside `list_objects`, replace the append with:

```python
                found.append(
                    S3Object(
                        bucket=bucket,
                        key=item["Key"],
                        etag=_strip_quotes(item["ETag"]),
                    )
                )
```

In `src/notes_rag/indexer/manifest.py`, replace `Manifest.of` and the first line of `diff`:

```python
    @classmethod
    def of(cls, objects: Sequence[S3Object]) -> "Manifest":
        """The manifest describing a listing - what gets written after a build."""
        return cls(etags={obj.qualified_key: obj.etag for obj in objects})
```

```python
        current = {obj.qualified_key: obj.etag for obj in objects}
```

Then update the docstring at the top of `manifest.py`: the map is from "every source object key" to its ETag — change that phrase to "every source object, qualified by bucket," so the file does not describe a scheme it no longer uses.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/indexer/test_manifest.py tests/sources/test_s3.py -v`
Expected: PASS. Other suites still fail — `run_index` is untouched until Task 4.

- [ ] **Step 5: Commit**

```bash
git add src/notes_rag/sources/s3.py src/notes_rag/indexer/manifest.py \
        tests/indexer/test_manifest.py tests/sources/test_s3.py
git commit -m "fix: key the manifest by bucket and key, not key alone"
```

---

### Task 3: Per-document vault id and vault-relative display paths

`build_chunks` currently takes one `vault_id` for the whole run. With a source list there can be several vaults, and non-note sources have none.

**Files:**
- Modify: `src/notes_rag/chunkers/markdown.py:31-58` (`chunk_markdown`)
- Modify: `src/notes_rag/indexer/collect.py:26-30, 46-52, 75-82, 107-138`
- Modify: `src/notes_rag/indexer/cli.py` (`_read_documents`, `main`)
- Test: `tests/chunkers/test_markdown.py` (modify), `tests/indexer/test_collect.py` (modify)

**Interfaces:**
- Consumes: nothing from Tasks 1–2.
- Produces: `SourceDocument(source_path: str, raw: bytes, vault_id: str | None = None, display_path: str | None = None)`. `CollectedDocuments.markdown_notes` is `tuple[tuple[str, SourceDocument], ...]` — decoded text paired with its document. `build_chunks(collected: CollectedDocuments) -> tuple[list[Chunk], tuple[Skip, ...]]` — **the `vault_id` keyword argument is gone**. `chunk_markdown(text, *, source_path, vault_id, display_path=None)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/chunkers/test_markdown.py`:

```python
def test_display_path_drives_context_not_source_path():
    from notes_rag.chunkers.markdown import chunk_markdown

    chunks = chunk_markdown(
        "# Heading\n\nBody text long enough to survive the normalizer's merge rules. "
        * 20,
        source_path="notes/josh/Daily/Monday.md",
        vault_id="josh",
        display_path="Daily/Monday.md",
    )
    assert chunks
    # source_path stays the S3 key - build_index deletes by it.
    assert chunks[0].source_path == "notes/josh/Daily/Monday.md"
    # ...but the embedded prefix carries the vault-relative path, so the vault
    # name is not repeated and no S3 prefix ends up inside the vector.
    assert chunks[0].context.startswith("josh / Daily/Monday.md / ")
    assert "notes/josh" not in chunks[0].context


def test_display_path_defaults_to_source_path():
    from notes_rag.chunkers.markdown import chunk_markdown

    chunks = chunk_markdown(
        "# Heading\n\nBody text long enough to survive the normalizer. " * 20,
        source_path="Daily/Monday.md",
        vault_id="josh",
    )
    assert chunks[0].context.startswith("josh / Daily/Monday.md / ")
```

Append to `tests/indexer/test_collect.py`:

```python
def test_markdown_without_a_vault_id_is_skipped_not_indexed():
    # A source list entry that points at markdown but forgets vault_id would
    # otherwise produce note chunks that no obsidian:// citation can be built
    # from - unlinkable content that looks fine until someone clicks it.
    from notes_rag.indexer.collect import SourceDocument, build_chunks, classify

    collected = classify([SourceDocument(source_path="notes/a.md", raw=b"# A\n\nbody\n")])
    chunks, skipped = build_chunks(collected)

    assert chunks == []
    assert skipped == (("notes/a.md", "markdown source has no vault_id"),)


def test_markdown_uses_its_own_documents_vault_id():
    from notes_rag.indexer.collect import SourceDocument, build_chunks, classify

    collected = classify(
        [
            SourceDocument(
                source_path="notes/josh/a.md",
                raw=b"# A\n\n" + b"body text " * 100,
                vault_id="josh",
                display_path="a.md",
            ),
            SourceDocument(
                source_path="notes/other/b.md",
                raw=b"# B\n\n" + b"body text " * 100,
                vault_id="other",
                display_path="b.md",
            ),
        ]
    )
    chunks, skipped = build_chunks(collected)

    assert skipped == ()
    assert {chunk.vault_id for chunk in chunks} == {"josh", "other"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/chunkers/test_markdown.py tests/indexer/test_collect.py -v`
Expected: FAIL — `chunk_markdown() got an unexpected keyword argument 'display_path'`, and `build_chunks() missing 1 required keyword-only argument: 'vault_id'`.

- [ ] **Step 3: Implement**

In `src/notes_rag/chunkers/markdown.py`, replace the signature and the two lines that use the path:

```python
def chunk_markdown(
    text: str, *, source_path: str, vault_id: str, display_path: str | None = None
) -> list[Chunk]:
    """Chunk one note. `source_path` is the index identity - the full S3 key -
    while `display_path` is what a reader would call the file inside the vault.

    They differ because the vault is synced under a `notes/<vault_id>/` prefix.
    The prefix belongs in the index (it makes source paths unique across
    vaults, and build_index deletes by source_path) and does not belong in the
    embedded context, where it would repeat the vault name and put an S3
    implementation detail inside the vector.
    """
    frontmatter, body = _split_frontmatter(text)
    if not body.strip():
        return []

    display = display_path or source_path
    title = frontmatter.get("title") or PurePosixPath(display).stem
    links = extract_wikilinks(body)
```

and inside the loop:

```python
                context=f"{vault_id} / {display} / {label}",
```

In `src/notes_rag/indexer/collect.py`, replace the `SourceDocument` dataclass:

```python
@dataclass(frozen=True)
class SourceDocument:
    source_path: str
    raw: bytes
    vault_id: str | None = None
    display_path: str | None = None
```

Change the `markdown_notes` field on `CollectedDocuments`:

```python
    markdown_notes: tuple[tuple[str, SourceDocument], ...]
```

Change the markdown branch inside `classify` (the `markdown_notes.append` call) to keep the document:

```python
            markdown_notes.append((text, document))
```

and its local declaration:

```python
    markdown_notes: list[tuple[str, SourceDocument]] = []
```

Replace `build_chunks`'s signature, docstring tail, and markdown loop:

```python
def build_chunks(collected: CollectedDocuments) -> tuple[list[Chunk], tuple[Skip, ...]]:
    """Chunk everything classified. Returns (chunks, skips from this stage).

    The returned skips are only those discovered while chunking - a transcript
    with no matching summary, or a markdown document whose source carried no
    vault_id. `collected.skipped` from classification is the caller's to
    report; the two are kept separate so a caller can tell a malformed file
    from an unusable one.

    There is no run-wide vault_id: the source list can name several vaults, and
    a video source has none at all, so it travels on each document.
    """
```

```python
    for text, document in collected.markdown_notes:
        if document.vault_id is None:
            skipped.append((document.source_path, "markdown source has no vault_id"))
            continue
        chunks.extend(
            chunk_markdown(
                text,
                source_path=document.source_path,
                vault_id=document.vault_id,
                display_path=document.display_path,
            )
        )
```

In `src/notes_rag/indexer/cli.py`, `_read_documents` must now stamp each document. Replace its body's append with:

```python
        documents.append(
            SourceDocument(
                source_path=rel,
                raw=path.read_bytes(),
                vault_id=vault_id,
                display_path=rel,
            )
        )
```

and change its signature to `def _read_documents(source: Path, vault_id: str) -> tuple[list[SourceDocument], list[Skip]]:`, adding to its docstring:

```
    Every document carries the CLI's --vault-id. Locally there is one vault and
    no S3 prefix, so display_path and source_path are the same value.
```

In `main`, replace the two call sites:

```python
    documents, suffix_skips = _read_documents(Path(args.source), args.vault_id)
    collected = classify(documents)
    chunks, unpairable = build_chunks(collected)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/chunkers tests/indexer/test_collect.py tests/indexer/test_cli.py -v`
Expected: PASS. If `test_cli.py` asserts on a context string containing a bare filename, it still holds — the CLI passes `display_path=rel`, which equals `source_path`.

- [ ] **Step 5: Commit**

```bash
git add src/notes_rag/chunkers/markdown.py src/notes_rag/indexer/collect.py \
        src/notes_rag/indexer/cli.py tests/chunkers/test_markdown.py \
        tests/indexer/test_collect.py
git commit -m "feat: carry vault id per document and keep S3 prefixes out of embeddings"
```

---

### Task 4: Wire the handler to multiple sources

**Files:**
- Modify: `src/notes_rag/indexer/handler.py:101-204` (`run_index`)
- Test: `tests/indexer/test_handler.py` (modify)

**Interfaces:**
- Consumes: `SourceSpec` and `IndexerConfig.sources` (Task 1); `S3Object.bucket` (Task 2); `SourceDocument.vault_id` / `.display_path` and the one-argument `build_chunks` (Task 3).
- Produces: `run_index(config, *, s3, embedder) -> IndexerResult`, signature unchanged. Module-level helper `_relative_to_prefix(key: str, prefixes: Sequence[str]) -> str`.

- [ ] **Step 1: Write the failing test**

In `tests/indexer/test_handler.py`, replace the `config()` helper. It stays **single-source** — same bucket and same keys the existing tests already use — so nothing else in the file has to change:

```python
def config(tmp_path) -> IndexerConfig:
    # One source, matching what stub_with_sources builds. vault_id is now
    # mandatory for the markdown under notes/: after per-document vault ids, a
    # document without one is skipped, and skipping it would leave the corpus
    # all-video and make test_public_db_contains_only_video_corpus_chunks
    # tautological again - the exact defect 702fbc9 fixed.
    return IndexerConfig(
        index_bucket="index",
        sources=(
            SourceSpec(
                bucket="source",
                prefixes=("summaries/", "transcripts/", "notes/"),
                vault_id="Vault",
            ),
        ),
        dimensions=DIMS,
        work_dir=str(tmp_path),
    )
```

Add `SourceSpec` to the import from `notes_rag.indexer.handler`. Then append the multi-bucket tests, which build their own config:

```python
def multi_source_config(tmp_path) -> IndexerConfig:
    return IndexerConfig(
        index_bucket="index",
        sources=(
            SourceSpec(bucket="source", prefixes=("summaries/", "transcripts/")),
            SourceSpec(bucket="notes", prefixes=("notes/josh/",), vault_id="josh"),
        ),
        dimensions=DIMS,
        work_dir=str(tmp_path),
    )


def two_bucket_router(make_s3):
    source = make_s3(
        {
            "summaries/vid1.json": json.dumps(SUMMARY).encode(),
            "transcripts/vid1.json": json.dumps(TRANSCRIPT).encode(),
        }
    )
    notes = make_s3({"notes/josh/Deep Note.md": (NOTE_MD + "body " * 200).encode()})
    return _BucketRouter({"source": source, "notes": notes, "index": make_s3({})})


def test_reads_from_every_source_bucket(tmp_path, make_s3):
    # The stub is per-bucket, so a handler that ignored obj.bucket and always
    # fetched from the first source would raise NoSuchKey on the note.
    router = two_bucket_router(make_s3)
    result = run_index(multi_source_config(tmp_path), s3=router, embedder=FakeEmbedder(DIMS))

    assert result.status == "rebuilt"

    store = SqliteVecStore(tmp_path / "full.db", dimensions=DIMS)
    try:
        rows = store._db.execute(
            "SELECT corpus, vault_id, source_path, context FROM chunks"
        ).fetchall()
    finally:
        store.close()

    assert {row[0] for row in rows} == {"video", "note"}

    note_rows = [row for row in rows if row[0] == "note"]
    assert note_rows, "the note source produced no chunks"
    assert all(row[1] == "josh" for row in note_rows)
    # source_path is the full S3 key...
    assert all(row[2].startswith("notes/josh/") for row in note_rows)
    # ...and the embedded context is vault-relative: no notes/josh/ inside it.
    assert all(row[3].startswith("josh / Deep Note.md / ") for row in note_rows)
    assert all("notes/josh" not in row[3] for row in note_rows)


def test_public_db_excludes_notes_from_a_second_bucket(tmp_path, make_s3):
    router = two_bucket_router(make_s3)
    run_index(multi_source_config(tmp_path), s3=router, embedder=FakeEmbedder(DIMS))

    store = SqliteVecStore(tmp_path / "public.db", dimensions=DIMS)
    try:
        corpora = {row[0] for row in store._db.execute("SELECT corpus FROM chunks")}
    finally:
        store.close()

    assert corpora == {"video"}
```

`store._db` rather than a public accessor: that is the access the existing tests in this file already use (`test_public_db_contains_only_video_corpus_chunks`), and this plan follows the file's convention rather than adding a second one.

The note's `display_path` is `Deep Note.md` — `_relative_to_prefix` strips the `notes/josh/` source prefix — so `chunk_markdown` derives `title` from that stem and the context reads `josh / Deep Note.md / <heading>`.

Add the router near the top of the file, under the fixtures:

```python
class _BucketRouter:
    """Dispatch each call to the StubS3 that owns the named bucket.

    The real client is one object serving every bucket; StubS3 is one object
    per bucket. This is the seam that makes the multi-source path testable
    without moto or credentials.
    """

    def __init__(self, buckets: dict) -> None:
        self._buckets = buckets
        # The adapter catches exceptions off the client, so they must be the
        # same classes every underlying stub raises.
        self.exceptions = next(iter(buckets.values())).exceptions

    def _for(self, kwargs):
        bucket = kwargs["Bucket"]
        try:
            return self._buckets[bucket]
        except KeyError:
            raise AssertionError(f"handler reached for an unconfigured bucket: {bucket}") from None

    def list_objects_v2(self, **kwargs):
        return self._for(kwargs).list_objects_v2(**kwargs)

    def get_object(self, **kwargs):
        return self._for(kwargs).get_object(**kwargs)

    def put_object(self, **kwargs):
        return self._for(kwargs).put_object(**kwargs)

    def head_object(self, **kwargs):
        return self._for(kwargs).head_object(**kwargs)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/indexer/test_handler.py -v`
Expected: FAIL — `AttributeError: 'IndexerConfig' object has no attribute 'source_bucket'`.

- [ ] **Step 3: Implement**

In `src/notes_rag/indexer/handler.py`, add `Sequence` to the `collections.abc` import, and add above `run_index`:

```python
def _relative_to_prefix(key: str, prefixes: Sequence[str]) -> str:
    """`key` with its matching source prefix removed.

    Longest match wins, so nested prefixes under one source resolve to the most
    specific one. A key that matches nothing is returned unchanged rather than
    guessed at.
    """
    matching = [p for p in prefixes if key.startswith(p)]
    if not matching:
        return key
    return key[len(max(matching, key=len)) :]
```

Replace the first three lines of `run_index`:

```python
    # Listed per source and kept beside the spec that produced it: fetching
    # needs the bucket, and chunking needs the vault_id.
    listings = [(source, list_objects(s3, source.bucket, source.prefixes)) for source in config.sources]
    objects = sorted(
        (obj for _, found in listings for obj in found),
        key=lambda obj: (obj.bucket, obj.key),
    )
    previous = Manifest.from_dict(get_json(s3, config.index_bucket, config.manifest_key))
    diff = previous.diff(objects)
```

Replace the document-collection loop (the block beginning `suffix_skips = []`) with:

```python
    suffix_skips = []
    documents = []
    for source, found in listings:
        for obj in found:
            skip = unsupported_suffix_skip(obj.key)
            if skip is not None:
                suffix_skips.append(skip)
                continue
            documents.append(
                SourceDocument(
                    source_path=obj.key,
                    raw=get_bytes(s3, obj.bucket, obj.key),
                    vault_id=source.vault_id,
                    display_path=_relative_to_prefix(obj.key, source.prefixes),
                )
            )
```

Replace the `build_chunks` call:

```python
    chunks, unpairable = build_chunks(collected)
```

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS, no failures. Then `uv run ruff check .` — expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/notes_rag/indexer/handler.py tests/indexer/test_handler.py
git commit -m "feat: index every configured source bucket, not just one"
```

---

### Task 5: Vault sync script

**Files:**
- Create: `scripts/sync_vault.sh`
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing.
- Produces: `scripts/sync_vault.sh <vault-dir> <vault-id>`, reading `NOTES_BUCKET` from the environment. Uploads only `*.md`, under `notes/<vault-id>/`.

- [ ] **Step 1: Write the script**

Create `scripts/sync_vault.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Sync one Obsidian vault into the indexer's source bucket.
#
# --delete is not optional. The indexer's manifest turns a removed object into
# a chunk deletion, so without it a note deleted in Obsidian stays answerable
# forever - the worst kind of stale, because the answer still cites a file the
# user believes they destroyed.
#
# Only *.md is uploaded. The indexer would skip everything else by suffix
# anyway, but not uploading it is cheaper, keeps .obsidian workspace state out
# of a bucket the Lambda can read, and removes an entire class of oversized-
# object failure before it can reach the index.

usage() {
  echo "usage: NOTES_BUCKET=<bucket> $0 <vault-dir> <vault-id> [aws s3 sync flags...]" >&2
  echo "example: NOTES_BUCKET=notes-rag-source-123 $0 ~/vaults/josh josh --dryrun" >&2
  exit 2
}

[ $# -ge 2 ] || usage

VAULT_DIR="$1"
VAULT_ID="$2"
shift 2

: "${NOTES_BUCKET:?set NOTES_BUCKET to the indexer source bucket}"
AWS_REGION="${AWS_REGION:-us-east-2}"

[ -d "$VAULT_DIR" ] || { echo "no such directory: $VAULT_DIR" >&2; exit 1; }

case "$VAULT_ID" in
  */*|"")
    # vault_id becomes an S3 prefix segment and a chunk's vault_id, which is
    # embedded in every content_hash. A slash would silently split the prefix.
    echo "vault-id must be a single path segment, got: '$VAULT_ID'" >&2
    exit 1
    ;;
esac

echo "syncing $VAULT_DIR -> s3://$NOTES_BUCKET/notes/$VAULT_ID/"

aws s3 sync "$VAULT_DIR" "s3://$NOTES_BUCKET/notes/$VAULT_ID/" \
  --region "$AWS_REGION" \
  --delete \
  --exclude '*' \
  --include '*.md' \
  --exclude '.obsidian/*' \
  --exclude '.trash/*' \
  "$@"

echo "done. the indexer picks this up within 5 minutes, or invoke it now:"
echo "  aws lambda invoke --function-name notes-rag-indexer --region $AWS_REGION \\"
echo "    --cli-binary-format raw-in-base64-out --payload '{}' /dev/stdout"
```

- [ ] **Step 2: Make it executable and check it refuses bad input**

```bash
chmod +x scripts/sync_vault.sh
./scripts/sync_vault.sh; echo "exit=$?"
NOTES_BUCKET=x ./scripts/sync_vault.sh /definitely/not/here josh; echo "exit=$?"
NOTES_BUCKET=x ./scripts/sync_vault.sh . 'a/b'; echo "exit=$?"
```

Expected: `exit=2` (usage), then `exit=1` with "no such directory", then `exit=1` with "must be a single path segment".

- [ ] **Step 3: Verify the filter set against the real vault without uploading**

```bash
NOTES_BUCKET=notes-rag-source-207423186995 \
  ./scripts/sync_vault.sh "/mnt/c/Users/joshi/OneDrive/Documents/Obsidian/joshiosimoe" joshiosimoe --dryrun
```

Expected: a `(dryrun) upload:` line per `.md` file (34 of them), nothing from `.obsidian/`. The bucket does not exist yet, so an error naming the bucket after the file list is fine — the file list is what this step checks.

- [ ] **Step 4: Document it in the README**

Add after the deployment section:

```markdown
### Syncing the vault

The indexer reads notes from an S3 prefix, not from GitHub. Push a vault with:

```bash
NOTES_BUCKET=notes-rag-source-<account-id> \
  ./scripts/sync_vault.sh ~/path/to/vault <vault-id>
```

`<vault-id>` must match a `vaults` entry in `infra/variables.tf` — it becomes
the S3 prefix, the chunk's `vault_id`, and part of every note chunk's
`content_hash`. Changing it later re-embeds the whole note corpus.

The script passes `--delete`, so a note deleted locally leaves the index on the
next run. Pass `--dryrun` to see what would move first.
```

- [ ] **Step 5: Commit**

```bash
git add scripts/sync_vault.sh README.md
git commit -m "feat: add a vault sync script that deletes as well as uploads"
```

---

### Task 6: Terraform — source bucket, source list, IAM

**Files:**
- Create: `infra/source.tf`
- Modify: `infra/variables.tf` (replace `source_bucket`, `source_prefixes`, `vault_id`)
- Modify: `infra/iam.tf:21-48` (`ReadSourceBucket`, `ListSourceBucket`)
- Modify: `infra/indexer.tf:33-46` (environment block)
- Modify: `infra/outputs.tf`
- Modify: `README.md`

**Interfaces:**
- Consumes: the `SOURCE_LIST` env contract from Task 1.
- Produces: `local.all_sources` (list of `{bucket, prefixes, vault_id}`), `aws_s3_bucket.source`, output `notes_bucket`.

- [ ] **Step 1: Replace the source variables**

In `infra/variables.tf`, delete the `source_bucket`, `source_prefixes`, and `vault_id` variables and add:

```hcl
variable "notes_bucket" {
  type        = string
  description = "Globally unique name for the bucket the Obsidian vault syncs into. Separate from index_bucket on purpose: the indexer has write access there, and a source its own consumer can overwrite is not a source."
}

variable "vaults" {
  type        = list(string)
  description = "Vault ids hosted in notes_bucket under notes/<id>/. Each id becomes the chunk's vault_id and therefore part of every note chunk's content_hash, so renaming one re-embeds that vault's whole corpus."
  default     = ["joshiosimoe"]
}

variable "external_sources" {
  type = list(object({
    bucket   = string
    prefixes = list(string)
    vault_id = optional(string)
  }))
  description = "Source buckets this stack does not own. Prefixes must end in \"/\" - they are interpolated as \"<prefix>*\" into the IAM s3:prefix condition, so \"notes\" would also grant \"notes-private/\"."
  default = [
    {
      bucket   = "videovaultstack-contentbucket52d4b12c-s0o3jpdq69b4"
      prefixes = ["summaries/", "transcripts/"]
    },
  ]
}
```

- [ ] **Step 2: Create the source bucket**

Create `infra/source.tf`:

```hcl
# Receives the Obsidian vault via scripts/sync_vault.sh. Deliberately not the
# index bucket: the indexer holds PutObject there, and a source its own
# consumer can overwrite is not a source.
resource "aws_s3_bucket" "source" {
  bucket = var.notes_bucket
}

resource "aws_s3_bucket_public_access_block" "source" {
  bucket                  = aws_s3_bucket.source.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "source" {
  bucket = aws_s3_bucket.source.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Versioning is the undo for `sync_vault.sh --delete`: a mistyped vault
# directory deletes every object in the prefix, and without versions that is
# unrecoverable.
resource "aws_s3_bucket_versioning" "source" {
  bucket = aws_s3_bucket.source.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "source" {
  bucket = aws_s3_bucket.source.id

  rule {
    id     = "expire-old-note-versions"
    status = "Enabled"

    filter {}

    # 30 days rather than the index bucket's 1: notes are kilobytes and this is
    # the only copy outside OneDrive, so retention is worth more here than the
    # storage it costs.
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# One list feeds both the Lambda's SOURCE_LIST and the IAM grant below, so the
# two cannot drift - the failure 07b819a had to fix.
locals {
  all_sources = concat(
    var.external_sources,
    [
      for vault in var.vaults : {
        bucket   = aws_s3_bucket.source.id
        prefixes = ["notes/${vault}/"]
        vault_id = vault
      }
    ],
  )
}
```

- [ ] **Step 3: Derive the IAM grant from the same list**

In `infra/iam.tf`, replace the `ReadSourceBucket` and `ListSourceBucket` statements (lines 21–48, keeping the `data "aws_iam_policy_document" "indexer" {` opening line and everything from `ReadWriteIndexBucket` onward) with:

```hcl
data "aws_iam_policy_document" "indexer" {
  # Read-only on each configured source, and only the prefixes the handler
  # actually reads - not the whole bucket. GetObject and ListBucket need
  # different resource shapes (object ARNs vs. the bucket ARN), so they are
  # separate statements. Both are generated from local.all_sources, which is
  # also what becomes SOURCE_LIST: the grant and the code read one list.
  dynamic "statement" {
    for_each = { for index, source in local.all_sources : index => source }

    content {
      sid     = "ReadSource${statement.key}"
      actions = ["s3:GetObject"]
      resources = [
        for prefix in statement.value.prefixes :
        "arn:aws:s3:::${statement.value.bucket}/${prefix}*"
      ]
    }
  }

  dynamic "statement" {
    for_each = { for index, source in local.all_sources : index => source }

    content {
      sid       = "ListSource${statement.key}"
      actions   = ["s3:ListBucket"]
      resources = ["arn:aws:s3:::${statement.value.bucket}"]

      # Without this, ListBucket would still be scoped to the bucket as a
      # whole - the resource ARN for ListBucket can only ever be the bucket
      # itself, never an object path. The s3:prefix condition is what actually
      # confines the *listing* to the watched prefixes.
      condition {
        test     = "StringLike"
        variable = "s3:prefix"
        values   = [for prefix in statement.value.prefixes : "${prefix}*"]
      }
    }
  }
```

- [ ] **Step 4: Hand the list to the Lambda**

In `infra/indexer.tf`, replace the `environment` block:

```hcl
  environment {
    variables = {
      # jsonencode of the same local the IAM policy is derived from. One list,
      # two consumers, no drift.
      SOURCE_LIST      = jsonencode(local.all_sources)
      INDEX_BUCKET     = aws_s3_bucket.index.id
      EMBED_DIMENSIONS = tostring(var.embed_dimensions)
      BEDROCK_REGION   = var.region
    }
  }
```

`VAULT_ID` is gone: vault ids now travel inside `SOURCE_LIST`, per source.

In `infra/outputs.tf`, add:

```hcl
output "notes_bucket" {
  value = aws_s3_bucket.source.id
}
```

- [ ] **Step 5: Format, validate, and plan**

```bash
cd infra
terraform fmt
terraform validate
terraform plan \
  -var="index_bucket=notes-rag-index-207423186995" \
  -var="notes_bucket=notes-rag-source-207423186995" \
  -out=/tmp/vault.tfplan
```

Expected: `Success! The configuration is valid.`

**Known failure mode at this step.** `concat` unifies the element types of both lists, and the generated entries are object literals whose `prefixes` is a tuple while `var.external_sources`' is `list(string)`. If `validate` reports inconsistent element types, make the generated entries match explicitly:

```hcl
      for vault in var.vaults : {
        bucket   = aws_s3_bucket.source.id
        prefixes = tolist(["notes/${vault}/"])
        vault_id = vault
      }
```

Then a plan that **adds** the source bucket and its four sub-resources, **changes** `aws_iam_role_policy.indexer` (two source statements instead of one pair) and `aws_lambda_function.indexer` (new `SOURCE_LIST`, no `SOURCE_BUCKET`/`SOURCE_PREFIXES`/`VAULT_ID`), and **destroys nothing**. If anything shows as destroyed, stop and investigate before applying.

Read the rendered `SOURCE_LIST` in the plan output and confirm it contains two entries, the second with `"vault_id": "joshiosimoe"`.

- [ ] **Step 6: Document the new variable**

In `README.md`, add `notes_bucket` to the deploy command:

```bash
terraform apply -var="index_bucket=notes-rag-index-<account-id>" \
                 -var="notes_bucket=notes-rag-source-<account-id>" \
                 -var="alarm_email=you@example.com"   # optional
```

and note under it:

```markdown
`notes_bucket` is created by this stack and receives the Obsidian vault. It is
separate from `index_bucket` because the indexer can write to the index bucket,
and a source its own consumer can overwrite is not a source.
```

- [ ] **Step 7: Commit**

```bash
git add infra/source.tf infra/variables.tf infra/iam.tf infra/indexer.tf \
        infra/outputs.tf README.md
git commit -m "feat: add a vault source bucket and derive the source list once"
```

---

### Task 7: Deploy and verify against the live account

Nothing before this proves the vault is actually indexed. This task is the deliverable.

**Files:**
- Modify: `tests/indexer/test_handler_integration.py`

**Interfaces:**
- Consumes: everything above.
- Produces: a deployed indexer reading two buckets, and an integration assertion that the corpus split is real.

- [ ] **Step 1: Rebuild the Lambda bundle**

```bash
./scripts/build_lambda.sh
```

Expected: `built .../build/lambda.zip`. Terraform derives `source_code_hash` from this zip, so skipping it deploys the previous code with no error and no diff — and the previous code does not understand `SOURCE_LIST`.

- [ ] **Step 2: Re-plan and apply interactively**

```bash
cd infra
terraform plan \
  -var="index_bucket=notes-rag-index-207423186995" \
  -var="notes_bucket=notes-rag-source-207423186995" \
  -out=/tmp/vault.tfplan
terraform apply /tmp/vault.tfplan
```

Expected: apply completes; `notes_bucket` appears in the outputs.

- [ ] **Step 2b: Verify the deployed `SOURCE_LIST`**

`SOURCE_LIST` is `(known after apply)` at plan time — `aws_s3_bucket.source.id` does not exist until the bucket does — so this is the first moment its literal value can be checked.

```bash
aws lambda get-function-configuration --function-name notes-rag-indexer \
  --region us-east-2 --query 'Environment.Variables.SOURCE_LIST' --output text | python3 -m json.tool
```

Expected: two entries. The second has `"vault_id": "joshiosimoe"` and `"prefixes": ["notes/joshiosimoe/"]` with the trailing slash intact. The first (the Video Vault source) has `"vault_id": null`, not `""` — an empty string would make `SourceSpec.from_dict` coerce it to `None` and silently skip any markdown under that source.

- [ ] **Step 3: Exercise the sync filter chain on a synthetic vault, then sync the real one**

The real vault has nothing under `.obsidian/` or `.trash/`, so syncing it proves nothing about the `--exclude '*' --include '*.md' --exclude '.obsidian/*' --exclude '.trash/*'` ordering. Build a throwaway tree that hits all four branches first:

```bash
T=$(mktemp -d)
mkdir -p "$T/.obsidian" "$T/.trash" "$T/sub"
touch "$T/top.md" "$T/sub/nested.md" "$T/.obsidian/workspace.md" "$T/.trash/deleted.md" "$T/notes.txt"
NOTES_BUCKET=notes-rag-source-207423186995 ./scripts/sync_vault.sh "$T" filtertest --dryrun
rm -rf "$T"
```

Expected: exactly two `(dryrun) upload:` lines — `top.md` and `sub/nested.md`. Nothing from `.obsidian/` or `.trash/`, and no `notes.txt`. If a `.obsidian/` or `.trash/` path appears, the filter order is wrong and the real sync must not run.

Then the real vault:

```bash
NOTES_BUCKET=notes-rag-source-207423186995 \
  ./scripts/sync_vault.sh "/mnt/c/Users/joshi/OneDrive/Documents/Obsidian/joshiosimoe" joshiosimoe
aws s3 ls s3://notes-rag-source-207423186995/notes/joshiosimoe/ --recursive --region us-east-2 | wc -l
```

Expected: 34 objects, all `.md`.

- [ ] **Step 4: Invoke and inspect the result**

```bash
aws lambda invoke --function-name notes-rag-indexer --region us-east-2 \
  --cli-binary-format raw-in-base64-out --payload '{}' /dev/stdout
```

Expected: `{"status": "rebuilt", ...}` with `chunks_written` well above 54 and `vectors_reused` above 0 — the 54 video chunks are unchanged, so their vectors come from cache and only the notes are newly embedded. `vectors_reused: 0` means the manifest key change forced a re-embed of the video corpus too; that is recoverable but worth understanding before moving on.

**Assert on the payload, not on the absence of a crash.** `"status": "no-op"` here is a failure, not a pass: a `SOURCE_LIST` naming a real-but-wrong bucket parses fine, lists nothing, and no-ops forever with no error anywhere. The two conditions that must hold are `status == "rebuilt"` and `chunks_written > 0`.

The manifest key format changed in Task 2, so this first run sees every object as new. That is expected exactly once.

- [ ] **Step 5: Verify the split on the real artifacts**

```bash
aws s3 cp s3://notes-rag-index-207423186995/index/full.db /tmp/full.db --region us-east-2
aws s3 cp s3://notes-rag-index-207423186995/index/public.db /tmp/public.db --region us-east-2
uv run python -c "
import sqlite3
for name in ('full', 'public'):
    c = sqlite3.connect(f'/tmp/{name}.db')
    print(name, c.execute('select corpus, count(*) from chunks group by corpus').fetchall())
c = sqlite3.connect('/tmp/full.db')
print('vault_ids:', c.execute('select distinct vault_id from chunks').fetchall())
print('sample note context:', c.execute(
    \"select context from chunks where corpus='note' limit 1\").fetchone())
print('backlinks in public.db:', sqlite3.connect('/tmp/public.db').execute(
    \"select count(*) from chunks where backlinks not in ('', '[]')\").fetchone())
"
```

Expected: `full` has both `note` and `video` rows; `public` has **only** `video`; `vault_ids` contains `joshiosimoe` (and `None` for video chunks); the sample note context starts `joshiosimoe / ` followed by a vault-relative path with no `notes/` in it; backlinks count in `public.db` is 0.

If `public.db` contains any `note` row, stop — that is the leak the whole split exists to prevent.

- [ ] **Step 6: Extend the integration test**

In `tests/indexer/test_handler_integration.py`, add:

```python
def test_public_db_holds_no_notes(tmp_path):
    """The corpus split, asserted against the deployed artifacts.

    The unit tests cover the same property against a stub. This one covers the
    thing that actually matters: that what is sitting in S3 right now, built by
    the deployed Lambda from the real vault, has no note in the public copy.
    """
    import sqlite3

    import boto3

    client = boto3.client("s3", region_name=REGION)

    paths = {}
    for name in ("full", "public"):
        path = tmp_path / f"{name}.db"
        path.write_bytes(
            client.get_object(Bucket=INDEX_BUCKET, Key=f"index/{name}.db")["Body"].read()
        )
        paths[name] = path

    def corpora(path):
        connection = sqlite3.connect(path)
        try:
            return {row[0] for row in connection.execute("SELECT DISTINCT corpus FROM chunks")}
        finally:
            connection.close()

    assert "note" in corpora(paths["full"]), "the vault is not being indexed"
    assert corpora(paths["public"]) == {"video"}, f"public.db leaked: {corpora(paths['public'])}"
```

No `@pytest.mark.integration` decorator: the module already sets `pytestmark = pytest.mark.integration`, so every test in it is marked. `boto3` and `sqlite3` are imported inside the function, matching how `invoke()` in that file already defers its boto3 import. `REGION` and `INDEX_BUCKET` are the module's existing constants.

- [ ] **Step 7: Run everything**

```bash
uv run pytest -q
uv run pytest -m integration -q
uv run ruff check .
```

Expected: unit suite green with more tests than the 183 baseline; integration suite green; ruff clean.

- [ ] **Step 8: Commit**

```bash
git add tests/indexer/test_handler_integration.py
git commit -m "test: assert the deployed public.db holds no notes"
```

---

## Done when

- `full.db` contains `corpus='note'` chunks with `vault_id='joshiosimoe'`.
- `public.db` contains only `corpus='video'`, with no backlinks.
- Note chunk `context` values are vault-relative and contain no `notes/` prefix.
- `terraform plan` is clean.
- Unit and integration suites pass; ruff clean.

## Known deferred items

Raised by per-task reviews, triaged by the whole-branch review as safe to defer.
None is reachable through Terraform-generated config; all are cosmetic or
ops-clarity.

- Several tests re-import symbols inside function bodies that are already
  imported at module scope (came from this plan's own snippets).
- `SourceSpec.from_dict`'s `payload: Mapping` is unparameterized while
  `from_env` uses `Mapping[str, str]`.
- `payload.get("vault_id") or None` coerces an explicit `""` to `None`, which
  then silently skips that source's markdown rather than erroring. Untested;
  only reachable by hand-writing `vault_id = ""` into `external_sources`.
- A `SOURCE_LIST` that parses to valid JSON but not a list of objects raises
  `AttributeError` rather than a clear `ValueError`. Only reachable via a
  hand-edited env var — Terraform always emits the right shape.
- `lambda_handler` constructs `TitanEmbedder` before entering `run_index`, so
  `handler.py`'s module docstring overstates the no-op path ("returns before
  … constructing a Bedrock client"). Pre-existing; construction is local and
  makes no network call.
- `test_config_rejects_a_missing_required_variable` duplicates
  `test_from_env_requires_source_list` in `tests/indexer/test_config.py`.
- `objects` is sorted by `(bucket, key)` for the manifest while `documents`
  build in source-declaration order. Harmless — both are consumed
  order-independently — but the sort reads as unexplained.
- `sync_vault.sh`'s vault-id check rejects only a slash and the empty string;
  `..`, `--recursive`, and whitespace are accepted as literal S3 key segments.
  No injection path: the value only ever appears inside a quoted `s3://` URI,
  never as a standalone argv token.
- No Terraform `validation` block enforcing the documented trailing-slash rule
  on `external_sources.prefixes`. Backstopped by `SourceSpec.from_dict`, which
  rejects a bad prefix at Lambda cold start rather than silently widening the
  IAM grant.

One finding was fixed rather than deferred: `derive_backlinks` resolved
wikilinks by bare filename stem, so two vaults each holding a `README.md` would
cross-link and share one merged backlink set. Dormant at one vault, live the
moment `var.vaults` gains a second — which is the capability this plan adds.
Fixed in `25992e6` by keying resolution on `(vault_id, stem)`.

## Deliberately not in this plan

- Retrieval and generation — Plan 4.
- The `Agentic-OS` vault. Adding it is one more entry in `var.vaults`, once the first vault is proven.
- Automating the sync. It is a manual script until there is evidence about how often the vault actually changes; a scheduled sync that nobody watches is a way to discover `--delete` the hard way.
