import pytest

from notes_rag.embed.fake import FakeEmbedder
from notes_rag.indexer.build import build_index, derive_backlinks
from notes_rag.models import Chunk
from notes_rag.store.sqlite_vec import SqliteVecStore

DIMS = 8


def note_chunk(chunk_id: str, path: str, *, text: str, links=()) -> Chunk:
    return Chunk(
        id=chunk_id,
        corpus="note",
        vault_id="V",
        source_path=path,
        chunk_type="note",
        title="T",
        heading=None,
        context="CTX",
        text=text,
        content_hash=f"hash-of-{text}",
        links_to=tuple(links),
    )


@pytest.fixture
def store(tmp_path):
    s = SqliteVecStore(tmp_path / "index.db", dimensions=DIMS)
    yield s
    s.close()


@pytest.fixture
def embedder():
    return FakeEmbedder(dimensions=DIMS)


def test_first_build_embeds_everything(store, embedder):
    stats = build_index(
        [note_chunk("a", "a.md", text="one"), note_chunk("b", "b.md", text="two")],
        store,
        embedder,
    )
    assert stats.chunks_written == 2
    assert stats.vectors_embedded == 2
    assert stats.vectors_reused == 0


def test_rebuild_with_identical_content_reuses_every_vector(store, embedder):
    chunks = [note_chunk("a", "a.md", text="one")]
    build_index(chunks, store, embedder)
    stats = build_index(chunks, store, embedder)
    assert stats.vectors_embedded == 0
    assert stats.vectors_reused == 1


def test_rebuild_embeds_only_the_changed_chunk(store, embedder):
    build_index(
        [note_chunk("a", "a.md", text="one"), note_chunk("b", "b.md", text="two")],
        store,
        embedder,
    )
    stats = build_index(
        [note_chunk("a", "a.md", text="one"), note_chunk("b", "b.md", text="CHANGED")],
        store,
        embedder,
    )
    assert stats.vectors_embedded == 1
    assert stats.vectors_reused == 1


def test_paths_absent_from_the_new_chunk_set_are_deleted(store, embedder):
    build_index(
        [note_chunk("a", "a.md", text="one"), note_chunk("b", "b.md", text="two")],
        store,
        embedder,
    )
    stats = build_index([note_chunk("a", "a.md", text="one")], store, embedder)
    assert stats.paths_deleted == 1
    assert store.all_source_paths() == {"a.md"}


def test_deleted_path_chunks_are_gone_from_search(store, embedder):
    build_index(
        [note_chunk("a", "a.md", text="one"), note_chunk("b", "b.md", text="two")],
        store,
        embedder,
    )
    build_index([note_chunk("a", "a.md", text="one")], store, embedder)
    hits = store.search(embedder.embed(["two"])[0], k=10)
    assert {hit.chunk.source_path for hit in hits} == {"a.md"}


def test_embedder_receives_only_uncached_texts(store):
    class CountingEmbedder(FakeEmbedder):
        def __init__(self):
            super().__init__(dimensions=DIMS)
            self.seen: list[str] = []

        def embed(self, texts):
            self.seen.extend(texts)
            return super().embed(texts)

    counting = CountingEmbedder()
    chunks = [note_chunk("a", "a.md", text="one")]
    build_index(chunks, store, counting)
    counting.seen.clear()
    build_index(chunks, store, counting)
    assert counting.seen == []


def test_build_with_no_chunks_deletes_everything(store, embedder):
    build_index([note_chunk("a", "a.md", text="one")], store, embedder)
    stats = build_index([], store, embedder)
    assert stats.paths_deleted == 1
    assert store.all_source_paths() == set()


def test_derive_backlinks_inverts_the_link_relation():
    chunks = [
        note_chunk("a", "Alpha.md", text="one", links=("Beta",)),
        note_chunk("b", "Beta.md", text="two"),
    ]
    out = derive_backlinks(chunks)
    beta = next(chunk for chunk in out if chunk.source_path == "Beta.md")
    assert beta.backlinks == ("Alpha",)


def test_derive_backlinks_leaves_unlinked_notes_empty():
    chunks = [note_chunk("a", "Alpha.md", text="one")]
    assert derive_backlinks(chunks)[0].backlinks == ()


def test_derive_backlinks_ignores_links_to_unknown_notes():
    chunks = [note_chunk("a", "Alpha.md", text="one", links=("Nonexistent",))]
    out = derive_backlinks(chunks)
    assert out[0].backlinks == ()


def test_derive_backlinks_deduplicates_and_sorts():
    chunks = [
        note_chunk("a1", "Alpha.md", text="one", links=("Beta",)),
        note_chunk("a2", "Alpha.md", text="two", links=("Beta",)),
        note_chunk("z", "Zeta.md", text="three", links=("Beta",)),
        note_chunk("b", "Beta.md", text="four"),
    ]
    beta = next(c for c in derive_backlinks(chunks) if c.source_path == "Beta.md")
    assert beta.backlinks == ("Alpha", "Zeta")


def test_backlinks_survive_a_build_round_trip(store, embedder):
    chunks = derive_backlinks(
        [
            note_chunk("a", "Alpha.md", text="one", links=("Beta",)),
            note_chunk("b", "Beta.md", text="two"),
        ]
    )
    build_index(chunks, store, embedder)
    hits = store.search(embedder.embed(["two"])[0], k=10)
    beta = next(hit.chunk for hit in hits if hit.chunk.source_path == "Beta.md")
    assert beta.backlinks == ("Alpha",)
