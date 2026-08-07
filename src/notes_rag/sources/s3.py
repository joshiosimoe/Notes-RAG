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
                found.append(
                    S3Object(
                        bucket=bucket,
                        key=item["Key"],
                        etag=_strip_quotes(item["ETag"]),
                    )
                )
            if not response.get("IsTruncated"):
                break
            token = response["NextContinuationToken"]
    return sorted(found, key=lambda obj: obj.key)


def get_bytes(client, bucket: str, key: str) -> bytes:
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


def head_exists(client, bucket: str, key: str) -> bool:
    """Whether `key` exists in `bucket`, checked with HEAD (no body transfer).

    Unlike get_object, S3's HEAD response for a missing key carries no XML
    error body for botocore to map to a modeled shape like NoSuchKey - it is
    always a bare 404, surfaced as the generic ClientError. A missing key is
    an expected state here (an artifact deleted to force a re-embed, or a
    version rollback that drops a newer object) rather than an error; any
    other error code is re-raised.
    """
    try:
        client.head_object(Bucket=bucket, Key=key)
    except client.exceptions.ClientError as error:
        if error.response.get("Error", {}).get("Code") == "404":
            return False
        raise
    return True


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
