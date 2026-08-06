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
    assert all(key.startswith(("summaries/", "transcripts/")) for key in manifest["etags"])
    assert manifest["etags"], "manifest recorded no source objects"
