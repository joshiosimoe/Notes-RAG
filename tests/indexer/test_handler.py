import json

import pytest

from notes_rag.embed.fake import FakeEmbedder
from notes_rag.indexer.handler import IndexerConfig, IndexerResult, SourceSpec, run_index
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
# A markdown note alongside the video artifacts. Without at least one non-video
# chunk in the corpus, `corpora == {"video"}` in the public.db test would hold
# whether or not copy_filtered's `WHERE c.corpus = ?` exists at all - this note
# is what makes that assertion genuinely discriminating, and incidentally
# exercises the markdown/vault_id path through the cloud handler.
NOTE_MD = "# A Note\n\nSome private body text nobody outside the vault should see.\n"


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


def stub_with_sources(make_s3):
    return make_s3(
        {
            "summaries/vid1.json": json.dumps(SUMMARY).encode(),
            "transcripts/vid1.json": json.dumps(TRANSCRIPT).encode(),
            "notes/a.md": NOTE_MD.encode(),
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
        assert store.all_source_paths() == {"summaries/vid1.json", "notes/a.md"}
    finally:
        store.close()


def test_public_db_contains_only_video_corpus_chunks(tmp_path, make_s3):
    """The corpus in `stub_with_sources` includes a markdown note (corpus
    "note") alongside the video artifacts, so this genuinely exercises
    copy_filtered's `WHERE c.corpus = ?` - without a non-video chunk present,
    `corpora == {"video"}` would hold whether or not that filter exists."""
    client = stub_with_sources(make_s3)
    run_index(config(tmp_path), s3=client, embedder=FakeEmbedder(dimensions=DIMS))

    full = tmp_path / "full_verify.db"
    full.write_bytes(client.objects["index/full.db"])
    full_store = SqliteVecStore(full, dimensions=DIMS)
    try:
        full_corpora = {
            row[0] for row in full_store._db.execute("SELECT DISTINCT corpus FROM chunks")
        }
    finally:
        full_store.close()
    assert full_corpora == {"video", "note"}, (
        "the note must actually be indexed for this test to mean anything"
    )

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
    # Manifest keys on qualified_key ("bucket/key"), not the bare key, so that
    # two sources can legitimately share a key without one masking the other.
    assert set(manifest["etags"]) == {
        "source/summaries/vid1.json",
        "source/transcripts/vid1.json",
        "source/notes/a.md",
    }


def test_an_unsupported_suffix_object_is_skipped_without_fetching_its_bytes(
    tmp_path, make_s3, caplog
):
    """A large object under a watched prefix with an unsupported suffix must
    never be pulled into memory just to discover it can't be used - that is
    what let one oversized non-JSON object (e.g. Video Vault writing a .vtt
    transcript) wedge the schedule forever: MemoryError before put_json means
    the manifest never advances and the next tick dies on the same object.
    The suffix must be decided from the key alone, before any get_object call
    for that key."""
    client = stub_with_sources(make_s3)
    client.objects["transcripts/x.vtt"] = b"not json, not markdown"

    fetched: list[str] = []
    original_get_object = client.get_object

    def recording_get_object(**kwargs):
        fetched.append(kwargs["Key"])
        return original_get_object(**kwargs)

    client.get_object = recording_get_object

    with caplog.at_level("WARNING"):
        result = run_index(config(tmp_path), s3=client, embedder=FakeEmbedder(dimensions=DIMS))

    assert result.status == "rebuilt"
    assert "transcripts/x.vtt" not in fetched, "bytes were fetched for an unsupported suffix"
    assert "transcripts/x.vtt" in caplog.text
    assert "unhandled file suffix" in caplog.text


def test_missing_full_db_forces_a_rebuild_even_with_an_unchanged_manifest(tmp_path, make_s3):
    """An empty ETag diff alone isn't enough to return no-op: an operator
    deleting full.db to force a re-embed, or a rollback to an S3 object
    version that predates the artifact (infra/storage.tf explicitly
    advertises this recovery path), must not be silently ignored forever."""
    client = stub_with_sources(make_s3)
    run_index(config(tmp_path), s3=client, embedder=FakeEmbedder(dimensions=DIMS))

    del client.objects["index/full.db"]

    result = run_index(config(tmp_path), s3=client, embedder=FakeEmbedder(dimensions=DIMS))
    assert result.status == "rebuilt"
    assert "index/full.db" in client.objects


def test_missing_public_db_forces_a_rebuild_even_with_an_unchanged_manifest(tmp_path, make_s3):
    client = stub_with_sources(make_s3)
    run_index(config(tmp_path), s3=client, embedder=FakeEmbedder(dimensions=DIMS))

    del client.objects["index/public.db"]

    result = run_index(config(tmp_path), s3=client, embedder=FakeEmbedder(dimensions=DIMS))
    assert result.status == "rebuilt"
    assert "index/public.db" in client.objects


def test_a_genuine_first_run_with_zero_sources_skips_the_missing_artifact_check(tmp_path, make_s3):
    """A brand-new deployment with nothing under the watched prefixes yet has
    an empty diff (nothing to compare against nothing) and no prior manifest.
    That is correctly a no-op, not something to "restore" - the missing-
    artifact check must only run when the manifest actually recorded a prior
    build, so head_object must not be called at all here."""
    client = make_s3()

    headed: list[str] = []
    original_head_object = client.head_object

    def recording_head_object(**kwargs):
        headed.append(kwargs["Key"])
        return original_head_object(**kwargs)

    client.head_object = recording_head_object

    result = run_index(config(tmp_path), s3=client, embedder=FakeEmbedder(dimensions=DIMS))
    assert result.status == "no-op"
    assert headed == []


def test_missing_full_db_logs_a_warning_about_re_embedding_the_whole_corpus(
    tmp_path, make_s3, caplog
):
    """download_file's return value must not be discarded: if full.db is
    absent while the manifest is non-empty, the rebuild silently re-embeds
    the entire corpus at full Bedrock cost, with no trace but
    vectors_reused: 0 unless this is logged."""
    client = stub_with_sources(make_s3)
    run_index(config(tmp_path), s3=client, embedder=FakeEmbedder(dimensions=DIMS))

    del client.objects["index/full.db"]

    with caplog.at_level("WARNING"):
        result = run_index(config(tmp_path), s3=client, embedder=FakeEmbedder(dimensions=DIMS))

    assert result.status == "rebuilt"
    assert result.vectors_reused == 0
    assert "re-embedded" in caplog.text


def test_stale_sqlite_side_files_are_removed_before_a_rebuild(tmp_path, make_s3):
    """A container Lambda can reuse /tmp across invocations. An invocation
    killed mid-write can leave a SQLite side file (-journal, -wal, -shm)
    behind; unlinking only full.db/public.db would let the next run's freshly
    downloaded full.db open next to a stale hot journal for a different
    database - a documented way to corrupt SQLite."""
    client = stub_with_sources(make_s3)
    run_index(config(tmp_path), s3=client, embedder=FakeEmbedder(dimensions=DIMS))

    stale_journal = tmp_path / "full.db-journal"
    stale_wal = tmp_path / "public.db-wal"
    stale_journal.write_bytes(b"stale journal")
    stale_wal.write_bytes(b"stale wal")

    changed = dict(SUMMARY)
    changed["summary"] = dict(SUMMARY["summary"], tldr="a different tldr entirely")
    client.objects["summaries/vid1.json"] = json.dumps(changed).encode()

    run_index(config(tmp_path), s3=client, embedder=FakeEmbedder(dimensions=DIMS))

    assert not stale_journal.exists()
    assert not stale_wal.exists()


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
            "SOURCE_LIST": json.dumps(
                [{"bucket": "src", "prefixes": ["summaries/", "transcripts/"]}]
            ),
            "INDEX_BUCKET": "idx",
            "EMBED_DIMENSIONS": "1024",
            "BEDROCK_REGION": "us-east-2",
        }
    )
    assert config.sources == (SourceSpec(bucket="src", prefixes=("summaries/", "transcripts/")),)
    assert config.index_bucket == "idx"
    assert config.dimensions == 1024
    assert config.bedrock_region == "us-east-2"


def test_config_defaults_the_optional_fields():
    config = IndexerConfig.from_env(
        {
            "SOURCE_LIST": json.dumps([{"bucket": "src", "prefixes": ["summaries/"]}]),
            "INDEX_BUCKET": "idx",
        }
    )
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
