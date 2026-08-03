import pytest

from notes_rag.embed.fake import FakeEmbedder


def test_returns_one_vector_per_text():
    vectors = FakeEmbedder(dimensions=8).embed(["a", "b", "c"])
    assert len(vectors) == 3


def test_vectors_have_the_declared_dimensions():
    embedder = FakeEmbedder(dimensions=8)
    assert all(len(vector) == 8 for vector in embedder.embed(["a"]))
    assert embedder.dimensions == 8


def test_is_deterministic_across_instances():
    assert FakeEmbedder(dimensions=8).embed(["hello"]) == FakeEmbedder(dimensions=8).embed(
        ["hello"]
    )


def test_different_texts_give_different_vectors():
    embedder = FakeEmbedder(dimensions=8)
    assert embedder.embed(["hello"]) != embedder.embed(["world"])


def test_vectors_are_unit_normalised():
    vector = FakeEmbedder(dimensions=8).embed(["hello"])[0]
    magnitude = sum(component**2 for component in vector) ** 0.5
    assert magnitude == pytest.approx(1.0)


def test_empty_input_returns_empty_list():
    assert FakeEmbedder(dimensions=8).embed([]) == []
