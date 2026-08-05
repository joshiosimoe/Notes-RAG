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
