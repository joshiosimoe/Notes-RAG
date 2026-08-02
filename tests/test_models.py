import pytest

from notes_rag.models import Chunk, estimate_tokens


def test_estimate_tokens_is_quarter_of_characters():
    assert estimate_tokens("a" * 400) == 100


def test_estimate_tokens_empty_string():
    assert estimate_tokens("") == 0


def test_chunk_is_frozen():
    chunk = Chunk(
        id="video:summaries/x.json#0",
        corpus="video",
        vault_id=None,
        source_path="summaries/x.json",
        chunk_type="summary",
        title="Some Title",
        heading="Custom scheduler",
        context="Some Title — Some Channel — Custom scheduler",
        text="body text",
        content_hash="deadbeef",
    )
    with pytest.raises(AttributeError):
        chunk.text = "mutated"  # type: ignore[misc]


def test_chunk_optional_fields_default_to_none_or_empty():
    chunk = Chunk(
        id="note:Class Notes/a.md#0",
        corpus="note",
        vault_id="Class Notes",
        source_path="Class Notes/a.md",
        chunk_type="note",
        title="a",
        heading=None,
        context="Class Notes / Class Notes/a.md",
        text="body",
        content_hash="cafe",
    )
    assert chunk.video_id is None
    assert chunk.start_seconds is None
    assert chunk.url is None
    assert chunk.links_to == ()
    assert chunk.backlinks == ()
