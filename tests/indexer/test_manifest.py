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
