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
