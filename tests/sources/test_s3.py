import hashlib
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


def expected_etag(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def test_list_objects_returns_keys_and_etags(make_s3):
    client = make_s3({"summaries/a.json": b"{}", "summaries/b.json": b"{}"})
    # Both objects have identical content, so they share an ETag - that's
    # correct and realistic, since a real S3 ETag is a content digest.
    assert list_objects(client, "bucket", ["summaries/"]) == [
        S3Object(key="summaries/a.json", etag=expected_etag(b"{}")),
        S3Object(key="summaries/b.json", etag=expected_etag(b"{}")),
    ]


def test_list_objects_strips_the_quotes_s3_wraps_around_etags(make_s3):
    client = make_s3({"summaries/a.json": b"{}"})
    assert not list_objects(client, "bucket", ["summaries/"])[0].etag.startswith('"')


def test_list_objects_spans_every_prefix(make_s3):
    client = make_s3(
        {"summaries/a.json": b"{}", "transcripts/a.json": b"{}", "other/a.json": b"{}"}
    )
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
