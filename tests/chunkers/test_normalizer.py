from notes_rag.chunkers.normalizer import normalize
from notes_rag.models import Chunk, estimate_tokens


def make(text: str, *, ordinal: int = 0, context: str = "CTX") -> Chunk:
    return Chunk(
        id=f"video:x#{ordinal}",
        corpus="video",
        vault_id=None,
        source_path="summaries/x.json",
        chunk_type="summary",
        title="T",
        heading=f"H{ordinal}",
        context=context,
        text=text,
        content_hash="",
    )


def test_applies_context_prefix_to_text():
    out = normalize([make("a" * 2000)])
    assert out[0].text.startswith("CTX\n\n")
    assert out[0].text.endswith("a" * 2000)


def test_computes_hash_over_prefixed_text():
    import hashlib

    out = normalize([make("a" * 2000)])
    expected = hashlib.sha256(("CTX\n\n" + "a" * 2000).encode()).hexdigest()
    assert out[0].content_hash == expected


def test_merges_chunk_below_min_into_next():
    small = make("x" * 100, ordinal=0)  # 25 tokens, below min
    big = make("y" * 2000, ordinal=1)  # 500 tokens
    out = normalize([small, big], min_tokens=150, max_tokens=800)
    assert len(out) == 1
    assert "x" * 100 in out[0].text
    assert "y" * 2000 in out[0].text


def test_merged_chunk_keeps_first_chunks_metadata():
    small = make("x" * 100, ordinal=0)
    big = make("y" * 2000, ordinal=1)
    out = normalize([small, big], min_tokens=150, max_tokens=800)
    assert out[0].heading == "H0"
    assert out[0].id == "video:x#0"


def test_trailing_small_chunk_merges_into_previous():
    big = make("y" * 2000, ordinal=0)
    small = make("x" * 100, ordinal=1)
    out = normalize([big, small], min_tokens=150, max_tokens=800)
    assert len(out) == 1
    assert "x" * 100 in out[0].text


def test_splits_chunk_above_max_on_paragraph_boundary():
    para = "z" * 1600  # 400 tokens each
    text = f"{para}\n\n{para}\n\n{para}"  # 1200 tokens total
    out = normalize([make(text)], min_tokens=150, max_tokens=800)
    assert len(out) == 2
    for chunk in out:
        assert estimate_tokens(chunk.text) <= 800 + estimate_tokens("CTX\n\n")


def test_split_chunks_get_distinct_ids():
    para = "z" * 1600
    out = normalize([make(f"{para}\n\n{para}\n\n{para}")], max_tokens=800)
    assert len({chunk.id for chunk in out}) == len(out)


def test_oversized_single_paragraph_is_left_intact():
    # No paragraph boundary to split on — emit as-is rather than cutting mid-word.
    out = normalize([make("q" * 8000)], max_tokens=800)
    assert len(out) == 1


def test_single_small_chunk_with_no_neighbour_survives():
    out = normalize([make("x" * 100)], min_tokens=150)
    assert len(out) == 1
    assert "x" * 100 in out[0].text


def test_empty_input_returns_empty():
    assert normalize([]) == []
