import pytest

from notes_rag.models import Chunk
from notes_rag.store.sqlite_vec import SqliteVecStore

DIMS = 4


def make_chunk(chunk_id: str, path: str, *, corpus="video", vault_id=None) -> Chunk:
    return Chunk(
        id=chunk_id,
        corpus=corpus,
        vault_id=vault_id,
        source_path=path,
        chunk_type="summary",
        title="T",
        heading="H",
        context="CTX",
        text=f"text for {chunk_id}",
        content_hash=f"hash-{chunk_id}",
        video_id="vid",
        start_seconds=10,
        url="https://example.com",
    )


@pytest.fixture
def store(tmp_path):
    s = SqliteVecStore(tmp_path / "test.db", dimensions=DIMS)
    yield s
    s.close()


def test_upsert_then_search_returns_the_nearest_chunk(store):
    store.upsert(
        [make_chunk("a", "p1.json"), make_chunk("b", "p2.json")],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )
    hits = store.search([1.0, 0.0, 0.0, 0.0], k=1)
    assert len(hits) == 1
    assert hits[0].chunk.id == "a"


def test_search_returns_hits_ordered_by_distance(store):
    store.upsert(
        [make_chunk("a", "p1.json"), make_chunk("b", "p2.json")],
        [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
    )
    hits = store.search([1.0, 0.0, 0.0, 0.0], k=2)
    assert [hit.chunk.id for hit in hits] == ["a", "b"]
    assert hits[0].distance <= hits[1].distance


def test_search_round_trips_all_chunk_fields(store):
    store.upsert([make_chunk("a", "p1.json")], [[1.0, 0.0, 0.0, 0.0]])
    chunk = store.search([1.0, 0.0, 0.0, 0.0], k=1)[0].chunk
    assert chunk.corpus == "video"
    assert chunk.video_id == "vid"
    assert chunk.start_seconds == 10
    assert chunk.url == "https://example.com"
    assert chunk.text == "text for a"


def test_search_filters_by_corpus(store):
    store.upsert(
        [
            make_chunk("a", "p1.json", corpus="video"),
            make_chunk("b", "p2.md", corpus="note", vault_id="V"),
        ],
        [[1.0, 0.0, 0.0, 0.0], [0.99, 0.01, 0.0, 0.0]],
    )
    hits = store.search([1.0, 0.0, 0.0, 0.0], k=5, corpus="note")
    assert [hit.chunk.id for hit in hits] == ["b"]


def test_search_filters_by_vault_id(store):
    store.upsert(
        [
            make_chunk("a", "p1.md", corpus="note", vault_id="Alpha"),
            make_chunk("b", "p2.md", corpus="note", vault_id="Beta"),
        ],
        [[1.0, 0.0, 0.0, 0.0], [0.99, 0.01, 0.0, 0.0]],
    )
    hits = store.search([1.0, 0.0, 0.0, 0.0], k=5, vault_id="Beta")
    assert [hit.chunk.id for hit in hits] == ["b"]


def test_upsert_replaces_a_chunk_with_the_same_id(store):
    store.upsert([make_chunk("a", "p1.json")], [[1.0, 0.0, 0.0, 0.0]])
    store.upsert([make_chunk("a", "p1.json")], [[0.0, 1.0, 0.0, 0.0]])
    assert len(store.search([0.0, 1.0, 0.0, 0.0], k=10)) == 1


def test_delete_by_path_removes_every_chunk_for_that_path(store):
    store.upsert(
        [make_chunk("a", "p1.json"), make_chunk("b", "p1.json"), make_chunk("c", "p2.json")],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
    )
    assert store.delete_by_path("p1.json") == 2
    remaining = {hit.chunk.id for hit in store.search([1.0, 1.0, 1.0, 0.0], k=10)}
    assert remaining == {"c"}


def test_delete_by_path_on_unknown_path_returns_zero(store):
    assert store.delete_by_path("nope.json") == 0


def test_cached_vectors_returns_known_hashes_only(store):
    store.upsert([make_chunk("a", "p1.json")], [[1.0, 0.0, 0.0, 0.0]])
    cached = store.cached_vectors({"hash-a", "hash-missing"})
    assert set(cached) == {"hash-a"}
    assert cached["hash-a"] == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_cached_vectors_with_empty_input(store):
    assert store.cached_vectors(set()) == {}


def test_all_source_paths(store):
    store.upsert(
        [make_chunk("a", "p1.json"), make_chunk("b", "p2.json")],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )
    assert store.all_source_paths() == {"p1.json", "p2.json"}


def test_copy_filtered_writes_a_video_only_database(store, tmp_path):
    store.upsert(
        [
            make_chunk("a", "p1.json", corpus="video"),
            make_chunk("b", "p2.md", corpus="note", vault_id="V"),
        ],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )
    dest = tmp_path / "public.db"
    store.copy_filtered(dest, corpus="video")

    public = SqliteVecStore(dest, dimensions=DIMS)
    try:
        hits = public.search([1.0, 1.0, 0.0, 0.0], k=10)
        assert {hit.chunk.id for hit in hits} == {"a"}
    finally:
        public.close()


def test_reopening_the_file_preserves_data(tmp_path):
    path = tmp_path / "persist.db"
    first = SqliteVecStore(path, dimensions=DIMS)
    first.upsert([make_chunk("a", "p1.json")], [[1.0, 0.0, 0.0, 0.0]])
    first.close()

    second = SqliteVecStore(path, dimensions=DIMS)
    try:
        assert len(second.search([1.0, 0.0, 0.0, 0.0], k=1)) == 1
    finally:
        second.close()


def test_upsert_rejects_mismatched_lengths(store):
    with pytest.raises(ValueError):
        store.upsert([make_chunk("a", "p1.json")], [])


def test_upsert_rejects_a_vector_of_the_wrong_dimensionality_and_writes_nothing(store):
    with pytest.raises(ValueError):
        store.upsert([make_chunk("a", "p1.json")], [[1.0, 0.0, 0.0]])
    assert store.all_source_paths() == set()
    assert store.search([1.0, 0.0, 0.0, 0.0], k=10) == []


def test_failed_batch_does_not_leak_into_a_later_successful_upsert(store):
    with pytest.raises(ValueError):
        store.upsert(
            [make_chunk("a", "p1.json"), make_chunk("b", "p2.json")],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
    store.upsert([make_chunk("c", "p3.json")], [[0.0, 0.0, 1.0, 0.0]])
    assert store.all_source_paths() == {"p3.json"}


def test_upsert_replace_leaves_no_orphaned_vector_row(store):
    store.upsert([make_chunk("a", "p1.json")], [[1.0, 0.0, 0.0, 0.0]])
    store.upsert([make_chunk("a", "p1.json")], [[0.0, 1.0, 0.0, 0.0]])
    chunk_count = store._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    vec_count = store._db.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
    assert vec_count == chunk_count
