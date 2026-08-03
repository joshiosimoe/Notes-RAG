from notes_rag.chunkers.video_transcript import chunk_video_transcript

PATH = "transcripts/dQw4w9WgXcQ.json"


def test_groups_segments_by_section_boundary(transcript_sample, summary_sample):
    chunks = chunk_video_transcript(transcript_sample, summary_sample, source_path=PATH)
    starts = sorted({chunk.start_seconds for chunk in chunks})
    assert starts == [0, 1120]


def test_pre_first_boundary_segments_land_in_the_leading_bucket(transcript_sample, summary_sample):
    chunks = chunk_video_transcript(transcript_sample, summary_sample, source_path=PATH)
    leading = next(chunk for chunk in chunks if chunk.start_seconds == 0)
    assert "Welcome back to the channel." in leading.text
    assert "Filtering removes nodes" in leading.text


def test_segments_after_boundary_land_in_that_section(transcript_sample, summary_sample):
    chunks = chunk_video_transcript(transcript_sample, summary_sample, source_path=PATH)
    later = next(chunk for chunk in chunks if chunk.start_seconds == 1120)
    assert "write our own scheduler" in later.text
    assert "schedulerName" in later.text


def test_chunk_type_is_transcript(transcript_sample, summary_sample):
    chunks = chunk_video_transcript(transcript_sample, summary_sample, source_path=PATH)
    assert all(chunk.chunk_type == "transcript" for chunk in chunks)
    assert all(chunk.corpus == "video" for chunk in chunks)


def test_carries_citation_fields_from_summary(transcript_sample, summary_sample):
    chunks = chunk_video_transcript(transcript_sample, summary_sample, source_path=PATH)
    for chunk in chunks:
        assert chunk.video_id == "dQw4w9WgXcQ"
        assert chunk.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_summary_with_no_sections_yields_one_bucket(transcript_sample):
    summary = {
        "video_id": "dQw4w9WgXcQ",
        "title": "T",
        "channel": "C",
        "url": "https://example.com",
        "summary": {"sections": []},
    }
    chunks = chunk_video_transcript(transcript_sample, summary, source_path=PATH)
    assert {chunk.start_seconds for chunk in chunks} == {0}


def test_empty_transcript_yields_no_chunks(summary_sample):
    empty = {"video_id": "dQw4w9WgXcQ", "language": "en", "segments": []}
    assert chunk_video_transcript(empty, summary_sample, source_path=PATH) == []
