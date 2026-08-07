from dataclasses import replace

import pytest

from notes_rag.embed.fake import FakeEmbedder
from notes_rag.indexer.build import build_index, derive_backlinks
from notes_rag.models import Chunk
from notes_rag.store.sqlite_vec import SqliteVecStore

DIMS = 8


def note_chunk(chunk_id: str, path: str, *, text: str, links=(), vault_id="V") -> Chunk:
    return Chunk(
        id=chunk_id,
        corpus="note",
        vault_id=vault_id,
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


def test_embed_failure_leaves_the_store_untouched(store, embedder):
    """embed() is the flaky, expensive step (a real Bedrock call can throttle
    or time out). If it raises mid-build, the store must still hold exactly
    what it held on entry — not be left with stale/incoming paths already
    deleted and nothing written back, which would force a full re-embed on
    retry and defeat the cache (spec §3).
    """
    original = [note_chunk("a", "a.md", text="one"), note_chunk("b", "b.md", text="two")]
    build_index(original, store, embedder)

    class FailingEmbedder(FakeEmbedder):
        def embed(self, texts):
            raise RuntimeError("embedder unavailable")

    with pytest.raises(RuntimeError):
        build_index(
            [note_chunk("a", "a.md", text="one"), note_chunk("b", "b.md", text="CHANGED")],
            store,
            FailingEmbedder(dimensions=DIMS),
        )

    assert store.all_source_paths() == {"a.md", "b.md"}
    hits = store.search(embedder.embed(["two"])[0], k=10)
    assert {hit.chunk.source_path for hit in hits} == {"a.md", "b.md"}


def test_upsert_failure_leaves_the_store_untouched(store, embedder):
    """Regression test for the CRITICAL finding: the old build_index deleted
    every stale path AND every incoming path (each delete committing on its
    own, since delete_by_path commits) before ever calling upsert. A failure
    inside upsert's own validation - e.g. a fresh vector of the wrong width -
    rolled back only the insert, leaving the deletes durable and the store
    completely empty. This must fail against the pre-fix code: `after`
    should differ from `before` there (empty set vs. two paths).
    """
    original = [note_chunk("a", "a.md", text="one"), note_chunk("b", "b.md", text="two")]
    build_index(original, store, embedder)
    before = store.all_source_paths()
    assert before == {"a.md", "b.md"}

    class WrongWidthEmbedder(FakeEmbedder):
        def embed(self, texts):
            # One dimension short of what the store expects - trips the
            # write path's dimension validation, not the embed() call itself.
            return [vector[:-1] for vector in super().embed(texts)]

    with pytest.raises(ValueError):
        build_index(
            [note_chunk("a", "a.md", text="one"), note_chunk("b", "b.md", text="CHANGED")],
            store,
            WrongWidthEmbedder(dimensions=DIMS),
        )

    after = store.all_source_paths()
    assert after == before == {"a.md", "b.md"}
    hits = store.search(embedder.embed(["two"])[0], k=10)
    assert {hit.chunk.source_path for hit in hits} == {"a.md", "b.md"}


def test_derive_backlinks_resolves_correctly_for_nested_paths():
    chunks = [
        note_chunk("a", "Class Notes/Alpha.md", text="one", links=("Beta",)),
        note_chunk("b", "Class Notes/Beta.md", text="two"),
    ]
    out = derive_backlinks(chunks)
    beta = next(chunk for chunk in out if chunk.source_path == "Class Notes/Beta.md")
    assert beta.backlinks == ("Alpha",)


def test_stem_collision_merges_backlinks_across_folders_known_limitation():
    """Known limitation, not a bug: derive_backlinks matches wikilink targets
    against source-path STEM only, by design, because wikilinks name a note,
    not a path. Two notes that share a filename in different folders
    therefore collapse into the same inbound bucket and receive an identical
    `backlinks` tuple. `backlinks` is not read by retrieval in v1, so this is
    an accepted side effect of stem matching. Disambiguating by folder would
    be a deliberate future change, not something this test asks for.
    """
    chunks = [
        note_chunk("a1", "Class Notes/Alpha.md", text="one"),
        note_chunk("a2", "Work/Alpha.md", text="two"),
        note_chunk("g", "Gamma.md", text="three", links=("Alpha",)),
    ]
    out = derive_backlinks(chunks)
    class_alpha = next(c for c in out if c.source_path == "Class Notes/Alpha.md")
    work_alpha = next(c for c in out if c.source_path == "Work/Alpha.md")
    assert class_alpha.backlinks == ("Gamma",)
    assert work_alpha.backlinks == ("Gamma",)


def test_derive_backlinks_replaces_existing_backlinks_with_empty_when_nothing_links():
    stale = replace(note_chunk("a", "Alpha.md", text="one"), backlinks=("Stale",))
    out = derive_backlinks([stale])
    assert out[0].backlinks == ()


def test_derive_backlinks_does_not_resolve_wikilinks_across_vaults():
    """Two different vaults each contain a README.md - an extremely common
    filename. A [[README]] link written inside a note in vault A must
    resolve only to vault A's README, never to vault B's unrelated note that
    happens to share the same bare stem.
    """
    chunks = [
        note_chunk("linker", "Linker.md", text="one", links=("README",), vault_id="A"),
        note_chunk("a-readme", "README.md", text="target-a", vault_id="A"),
        note_chunk("b-readme", "README.md", text="target-b", vault_id="B"),
    ]
    out = derive_backlinks(chunks)
    target_a = next(c for c in out if c.vault_id == "A" and c.source_path == "README.md")
    target_b = next(c for c in out if c.vault_id == "B" and c.source_path == "README.md")
    assert target_a.backlinks == ("Linker",)
    assert target_b.backlinks == ()
