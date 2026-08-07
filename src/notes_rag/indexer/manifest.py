"""What the previous run saw, so this run can tell whether anything changed.

Pure data - no IO. The manifest maps every source object, qualified by bucket, to
the ETag it had when the index was last built. Comparing it to a fresh listing is
what lets the overwhelmingly common no-op run exit in milliseconds, before
downloading the index or calling Bedrock.
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
        return cls(etags={obj.qualified_key: obj.etag for obj in objects})

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
        current = {obj.qualified_key: obj.etag for obj in objects}
        changed = tuple(sorted(k for k, etag in current.items() if self.etags.get(k) != etag))
        removed = tuple(sorted(set(self.etags) - set(current)))
        return ManifestDiff(changed=changed, removed=removed)
