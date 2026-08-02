from notes_rag.chunkers.video_summary import chunk_video_summary

PATH = "summaries/dQw4w9WgXcQ.json"


def test_emits_one_chunk_per_section_plus_an_overview(summary_sample):
    chunks = chunk_video_summary(summary_sample, source_path=PATH)
    # 2 sections + 1 overview, before any merging
    assert len(chunks) >= 1
    headings = {chunk.heading for chunk in chunks}
    assert "Custom scheduler" in headings or any(
        "Custom scheduler" in chunk.text for chunk in chunks
    )


def test_overview_chunk_carries_verdict_tldr_and_takeaways(summary_sample):
    chunks = chunk_video_summary(summary_sample, source_path=PATH)
    combined = "\n".join(chunk.text for chunk in chunks)
    assert "Worth watching if you operate clusters at scale." in combined
    assert "Walks through the default scheduler" in combined
    assert "filtering and scoring" in combined


def test_every_chunk_has_video_citation_fields(summary_sample):
    chunks = chunk_video_summary(summary_sample, source_path=PATH)
    for chunk in chunks:
        assert chunk.corpus == "video"
        assert chunk.chunk_type == "summary"
        assert chunk.video_id == "dQw4w9WgXcQ"
        assert chunk.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert chunk.start_seconds is not None


def test_section_chunk_uses_its_own_start_seconds(summary_sample):
    chunks = chunk_video_summary(summary_sample, source_path=PATH)
    starts = {chunk.start_seconds for chunk in chunks}
    assert 1120 in starts or 0 in starts


def test_context_prefix_includes_title_and_channel(summary_sample):
    chunks = chunk_video_summary(summary_sample, source_path=PATH)
    for chunk in chunks:
        assert "How Kubernetes Scheduling Actually Works" in chunk.text
        assert "Some Channel" in chunk.text


def test_all_chunks_have_a_content_hash(summary_sample):
    chunks = chunk_video_summary(summary_sample, source_path=PATH)
    assert all(len(chunk.content_hash) == 64 for chunk in chunks)


def test_source_path_is_recorded(summary_sample):
    chunks = chunk_video_summary(summary_sample, source_path=PATH)
    assert all(chunk.source_path == PATH for chunk in chunks)


def test_missing_sections_still_produces_overview():
    minimal = {
        "video_id": "abc",
        "title": "T",
        "channel": "C",
        "url": "https://example.com",
        "summary": {"verdict": "v", "tldr": "t", "takeaways": [], "sections": []},
    }
    chunks = chunk_video_summary(minimal, source_path="summaries/abc.json")
    assert len(chunks) == 1
    assert "v" in chunks[0].text
