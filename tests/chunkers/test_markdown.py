from notes_rag.chunkers.markdown import chunk_markdown, extract_wikilinks

PATH = "Class Notes/Kubernetes Scheduling.md"
VAULT = "Class Notes"


def test_extract_plain_wikilink():
    assert extract_wikilinks("see [[Bloom Filters]] here") == ("Bloom Filters",)


def test_extract_aliased_wikilink_returns_target_not_alias():
    assert extract_wikilinks("[[Node Affinity|affinity rules]]") == ("Node Affinity",)


def test_extract_heading_anchored_wikilink_returns_note_only():
    assert extract_wikilinks("[[Scoring#Priorities]]") == ("Scoring",)


def test_extract_embed_wikilink():
    assert extract_wikilinks("![[Diagram]]") == ("Diagram",)


def test_extract_deduplicates_preserving_order():
    assert extract_wikilinks("[[A]] [[B]] [[A]]") == ("A", "B")


def test_extract_returns_empty_when_no_links():
    assert extract_wikilinks("plain text") == ()


def test_splits_on_headings(note_sample):
    chunks = chunk_markdown(note_sample, source_path=PATH, vault_id=VAULT)
    combined = "\n".join(chunk.text for chunk in chunks)
    assert "Filtering phase" in combined
    assert "Scoring phase" in combined


def test_preamble_before_first_heading_is_kept(note_sample):
    chunks = chunk_markdown(note_sample, source_path=PATH, vault_id=VAULT)
    combined = "\n".join(chunk.text for chunk in chunks)
    assert "Intro paragraph before any heading." in combined


def test_frontmatter_is_stripped_from_text(note_sample):
    chunks = chunk_markdown(note_sample, source_path=PATH, vault_id=VAULT)
    combined = "\n".join(chunk.text for chunk in chunks)
    assert "status: reviewed" not in combined


def test_title_comes_from_frontmatter_when_present(note_sample):
    chunks = chunk_markdown(note_sample, source_path=PATH, vault_id=VAULT)
    assert all(chunk.title == "Kubernetes Scheduling" for chunk in chunks)


def test_title_falls_back_to_filename_stem():
    chunks = chunk_markdown("no frontmatter here", source_path=PATH, vault_id=VAULT)
    assert chunks[0].title == "Kubernetes Scheduling"


def test_links_are_collected_across_the_whole_note(note_sample):
    chunks = chunk_markdown(note_sample, source_path=PATH, vault_id=VAULT)
    all_links = {link for chunk in chunks for link in chunk.links_to}
    assert all_links == {"Bloom Filters", "Node Affinity", "Scoring"}


def test_backlinks_are_empty_at_chunk_time(note_sample):
    chunks = chunk_markdown(note_sample, source_path=PATH, vault_id=VAULT)
    assert all(chunk.backlinks == () for chunk in chunks)


def test_corpus_and_vault_metadata(note_sample):
    chunks = chunk_markdown(note_sample, source_path=PATH, vault_id=VAULT)
    for chunk in chunks:
        assert chunk.corpus == "note"
        assert chunk.chunk_type == "note"
        assert chunk.vault_id == VAULT
        assert chunk.video_id is None
        assert chunk.start_seconds is None


def test_context_prefix_includes_vault_and_path(note_sample):
    chunks = chunk_markdown(note_sample, source_path=PATH, vault_id=VAULT)
    assert all(VAULT in chunk.text and PATH in chunk.text for chunk in chunks)


def test_empty_note_yields_no_chunks():
    assert chunk_markdown("", source_path=PATH, vault_id=VAULT) == []


def test_frontmatter_only_note_yields_no_chunks():
    assert chunk_markdown("---\ntitle: X\n---\n", source_path=PATH, vault_id=VAULT) == []
