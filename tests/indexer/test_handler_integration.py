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
    response = client.invoke(FunctionName=FUNCTION, InvocationType="RequestResponse", Payload=b"{}")
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
        for item in client.list_objects_v2(Bucket=INDEX_BUCKET, Prefix="index/").get("Contents", [])
    }
    assert {"index/full.db", "index/public.db", "index/manifest.json"} <= keys


def test_public_db_contains_only_video_corpus_chunks(tmp_path):
    """public.db is the demo artifact. Per spec decision 5, the isolation
    mechanism is physical file separation, not a runtime query predicate: the
    demo Lambda's IAM role is scoped to read only this file, so a corpus-filter
    bug can't leak other corpora by being reachable at query time - it can only
    leak by having been copied into this file in the first place. This is the
    check that catches that before it ships: if anything other than
    corpus="video" ever lands in public.db, this fails before an
    unauthenticated endpoint can serve it.

    Caveat, stated plainly rather than implied: every source indexed today is
    corpus="video", so this assertion currently passes because there is
    nothing else to filter out, not because the filter has been proven to
    exclude anything. It becomes a genuinely discriminating test once a
    non-video source (the GitHub vault ingester) lands in full.db.
    """
    import boto3

    from notes_rag.store.sqlite_vec import SqliteVecStore

    client = boto3.client("s3", region_name=REGION)
    dest = tmp_path / "public.db"
    client.download_file(INDEX_BUCKET, "index/public.db", str(dest))

    store = SqliteVecStore(dest, dimensions=1024)
    try:
        # No public accessor for distinct corpus values (all_source_paths()
        # is the equivalent for source_path); read the connection directly,
        # same pattern that method uses internally.
        corpora = {row[0] for row in store._db.execute("SELECT DISTINCT corpus FROM chunks")}
    finally:
        store.close()
    assert corpora == {"video"}

    # Secondary, weaker check kept alongside the real one: public.db must
    # never be the larger of the two. On today's all-video corpus this alone
    # could not have caught a filter regression - see the docstring above.
    full = client.head_object(Bucket=INDEX_BUCKET, Key="index/full.db")["ContentLength"]
    public = client.head_object(Bucket=INDEX_BUCKET, Key="index/public.db")["ContentLength"]
    assert public <= full


def test_the_manifest_lists_the_source_objects():
    import boto3

    client = boto3.client("s3", region_name=REGION)
    raw = client.get_object(Bucket=INDEX_BUCKET, Key="index/manifest.json")["Body"].read()
    manifest = json.loads(raw)
    assert manifest["version"] == 1
    assert all(key.startswith(("summaries/", "transcripts/")) for key in manifest["etags"])
    assert manifest["etags"], "manifest recorded no source objects"


def test_public_db_holds_no_notes(tmp_path):
    """Verify that the public.db artifact contains no note chunks.

    The full/public split is the mechanism protecting unauthenticated endpoints
    from serving private notes. This test verifies the physical separation is
    working: a corpus-filter bug would be caught here before the file ships.
    It also verifies that the vault is being indexed (full.db contains notes
    while public.db does not).
    """
    import sqlite3

    import boto3

    client = boto3.client("s3", region_name=REGION)

    # Download both artifacts
    full_path = tmp_path / "full.db"
    public_path = tmp_path / "public.db"
    client.download_file(INDEX_BUCKET, "index/full.db", str(full_path))
    client.download_file(INDEX_BUCKET, "index/public.db", str(public_path))

    # Verify full.db contains note chunks (vault is indexed)
    full_conn = sqlite3.connect(str(full_path))
    try:
        full_notes = full_conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE corpus = 'note'"
        ).fetchone()[0]
        assert full_notes > 0, "full.db contains no note chunks - vault not indexed"
    finally:
        full_conn.close()

    # Verify public.db contains no note chunks
    public_conn = sqlite3.connect(str(public_path))
    try:
        public_notes = public_conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE corpus = 'note'"
        ).fetchone()[0]
        assert public_notes == 0, "public.db leaked note chunks"
    finally:
        public_conn.close()
