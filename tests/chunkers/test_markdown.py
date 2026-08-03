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


def test_heading_splitting_produces_distinct_chunks_with_correct_headings():
    """Verify that section headings are correctly captured and that chunks don't merge.

    Sections are sized > 150 tokens (600+ chars) to force normalize() to leave them
    separate. This test would fail if _split_on_headings were replaced with a no-op
    like 'return [(None, body.strip())]'.
    """
    # Preamble: 600+ chars so it survives as its own chunk
    preamble = (
        "This is the introductory content before any heading. "
        "It provides important context about the document. "
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, "
        "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
        "Nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit. "
        "In voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
        "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt. "
        "This preamble contains a unique marker: PREAMBLE_ONLY_MARKER_12345. "
        "Additional text to ensure sufficient length for this section. "
    )

    # Section 1: Filtering phase, 600+ chars with unique marker
    filtering = (
        "## Filtering Phase\n\n"
        "This section discusses the filtering phase in detail. "
        "The scheduler removes nodes that cannot host the pod. "
        "Various constraints are evaluated during this phase. "
        "This includes CPU, memory, storage, and network requirements. "
        "Affinity and anti-affinity rules are checked here. "
        "Node selectors and taints are processed in this phase. "
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, "
        "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
        "Nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit. "
        "The filtering phase contains a unique marker: FILTERING_PHASE_MARKER_67890. "
        "Additional content specific to filtering logic goes here. "
    )

    # Section 2: Scoring phase, 600+ chars with unique marker
    scoring = (
        "## Scoring Phase\n\n"
        "This section explains the scoring phase in detail. "
        "Remaining nodes are ranked according to priority weights. "
        "Multiple scoring functions can be used to rank nodes. "
        "The highest-scoring node is selected to host the pod. "
        "Different metrics can influence the scoring process. "
        "Custom scoring plugins can be integrated here. "
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, "
        "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
        "Nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit. "
        "The scoring phase contains a unique marker: SCORING_PHASE_MARKER_54321. "
        "Additional details about scoring algorithms and implementation. "
    )

    note_text = f"{preamble}\n\n{filtering}\n\n{scoring}"

    chunks = chunk_markdown(note_text, source_path=PATH, vault_id=VAULT)

    # Should have 3 separate chunks (not merged) because each is large enough
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"

    # Preamble chunk should have heading=None
    preamble_chunk = chunks[0]
    assert preamble_chunk.heading is None, "Preamble chunk should have heading=None"
    assert "PREAMBLE_ONLY_MARKER_12345" in preamble_chunk.text

    # Filtering chunk should have the correct heading
    filtering_chunk = chunks[1]
    assert filtering_chunk.heading == "Filtering Phase", (
        f"Expected heading 'Filtering Phase', got '{filtering_chunk.heading}'"
    )
    assert "FILTERING_PHASE_MARKER_67890" in filtering_chunk.text
    assert "PREAMBLE_ONLY_MARKER_12345" not in filtering_chunk.text, (
        "Filtering chunk shouldn't contain preamble marker"
    )
    assert "SCORING_PHASE_MARKER_54321" not in filtering_chunk.text, (
        "Filtering chunk shouldn't contain scoring marker"
    )

    # Scoring chunk should have the correct heading
    scoring_chunk = chunks[2]
    assert scoring_chunk.heading == "Scoring Phase", (
        f"Expected heading 'Scoring Phase', got '{scoring_chunk.heading}'"
    )
    assert "SCORING_PHASE_MARKER_54321" in scoring_chunk.text
    assert "PREAMBLE_ONLY_MARKER_12345" not in scoring_chunk.text, (
        "Scoring chunk shouldn't contain preamble marker"
    )
    assert "FILTERING_PHASE_MARKER_67890" not in scoring_chunk.text, (
        "Scoring chunk shouldn't contain filtering marker"
    )
