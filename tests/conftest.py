import hashlib
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


class _NoSuchKey(Exception):
    pass


class _ClientError(Exception):
    """Stand-in for botocore.exceptions.ClientError.

    Real botocore attaches the parsed error body to `.response`; head_object
    on a missing key has no modeled shape, so the adapter checks
    `error.response["Error"]["Code"]` the same way it would against a real
    client.
    """

    def __init__(self, code: str) -> None:
        self.response = {"Error": {"Code": code}}
        super().__init__(code)


class StubS3:
    """Minimal stand-in for a boto3 S3 client.

    Only the calls the adapter makes are implemented. `exceptions.NoSuchKey`
    and `exceptions.ClientError` mirror botocore's generated exception
    classes, which the adapter catches by attribute off the client rather
    than by importing botocore.
    """

    class exceptions:
        NoSuchKey = _NoSuchKey
        ClientError = _ClientError

    def __init__(self, objects: dict[str, bytes] | None = None, page_size: int = 1000) -> None:
        self.objects = dict(objects or {})
        self.page_size = page_size

    def list_objects_v2(self, **kwargs):
        prefix = kwargs["Prefix"]
        token = kwargs.get("ContinuationToken")
        keys = sorted(k for k in self.objects if k.startswith(prefix))
        start = int(token) if token is not None else 0
        page = keys[start : start + self.page_size]
        # A real S3 ETag is a digest of the object body (MD5 for single-part
        # uploads), which is the entire reason the manifest can detect a
        # same-key content change. A key-derived ETag could not represent
        # that case at all, so this stub hashes the actual bytes instead -
        # sha256 rather than MD5 because the digest algorithm doesn't matter
        # for what's under test, and sha256 avoids failing on FIPS-restricted
        # hosts where MD5 is disabled. The literal double quotes are kept:
        # the adapter's quote-stripping is tested against them.
        contents = [
            {"Key": key, "ETag": f'"{hashlib.sha256(self.objects[key]).hexdigest()}"'}
            for key in page
        ]
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

    def head_object(self, **kwargs):
        key = kwargs["Key"]
        if key not in self.objects:
            raise _ClientError("404")
        return {}


@pytest.fixture
def make_s3():
    """Factory for StubS3, so tests never import across test modules."""

    def _make(objects: dict[str, bytes] | None = None, page_size: int = 1000) -> StubS3:
        return StubS3(objects, page_size)

    return _make
