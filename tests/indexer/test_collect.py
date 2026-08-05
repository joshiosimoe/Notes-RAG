import json

from notes_rag.indexer.collect import CollectedDocuments, SourceDocument, build_chunks, classify

SUMMARY = {
    "video_id": "vid1",
    "title": "T",
    "channel": "C",
    "url": "https://example.com/watch?v=vid1",
    "summary": {
        "verdict": "v",
        "tldr": "t",
        "takeaways": ["a"],
        "sections": [{"start_seconds": 0, "title": "Intro", "summary": "s"}],
    },
}
TRANSCRIPT = {
    "video_id": "vid1",
    "language": "en",
    "segments": [{"start_seconds": 0, "text": "hello there"}],
}


def doc(path: str, payload) -> SourceDocument:
    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return SourceDocument(source_path=path, raw=raw)


def test_classifies_a_summary_by_its_summary_object():
    collected = classify([doc("summaries/vid1.json", SUMMARY)])
    assert [path for _, path in collected.summaries] == ["summaries/vid1.json"]
    assert collected.transcripts == ()


def test_classifies_a_transcript_by_its_segments_list():
    collected = classify([doc("transcripts/vid1.json", TRANSCRIPT)])
    assert [path for _, path in collected.transcripts] == ["transcripts/vid1.json"]
    assert collected.summaries == ()


def test_classifies_markdown_by_suffix():
    collected = classify([SourceDocument(source_path="notes/a.md", raw=b"# hi")])
    assert [path for _, path in collected.markdown_notes] == ["notes/a.md"]


def test_skips_json_whose_top_level_is_not_an_object():
    collected = classify([doc("summaries/bad.json", [1, 2, 3])])
    assert collected.summaries == ()
    assert [path for path, _ in collected.skipped] == ["summaries/bad.json"]


def test_skips_json_of_an_unrecognized_shape():
    collected = classify([doc("summaries/odd.json", {"nothing": "useful"})])
    assert [path for path, _ in collected.skipped] == ["summaries/odd.json"]


def test_skips_malformed_json_rather_than_raising():
    collected = classify([doc("summaries/broken.json", b"{not json")])
    assert [path for path, _ in collected.skipped] == ["summaries/broken.json"]


def test_skips_a_file_with_an_unhandled_suffix():
    collected = classify([SourceDocument(source_path="notes/photo.png", raw=b"\x89PNG")])
    assert [path for path, _ in collected.skipped] == ["notes/photo.png"]


def test_every_skip_carries_a_reason():
    collected = classify([doc("summaries/broken.json", b"{not json")])
    assert all(reason for _, reason in collected.skipped)


def test_classify_of_nothing_is_empty():
    collected = classify([])
    assert collected == CollectedDocuments(
        summaries=(), transcripts=(), markdown_notes=(), skipped=()
    )


def test_build_chunks_emits_summary_and_transcript_chunks():
    collected = classify(
        [doc("summaries/vid1.json", SUMMARY), doc("transcripts/vid1.json", TRANSCRIPT)]
    )
    chunks, skipped = build_chunks(collected, vault_id="V")
    assert skipped == ()
    assert {chunk.chunk_type for chunk in chunks} == {"summary", "transcript"}
    assert all(chunk.video_id == "vid1" for chunk in chunks)


def test_build_chunks_pairs_a_transcript_with_its_summary_by_video_id():
    other_summary = dict(SUMMARY, video_id="vid2", url="https://example.com/watch?v=vid2")
    collected = classify(
        [
            doc("summaries/vid1.json", SUMMARY),
            doc("summaries/vid2.json", other_summary),
            doc("transcripts/vid2.json", dict(TRANSCRIPT, video_id="vid2")),
        ]
    )
    chunks, _ = build_chunks(collected, vault_id="V")
    transcript_chunks = [c for c in chunks if c.chunk_type == "transcript"]
    assert transcript_chunks
    assert all(c.video_id == "vid2" for c in transcript_chunks)


def test_build_chunks_skips_a_transcript_with_no_matching_summary():
    collected = classify([doc("transcripts/orphan.json", dict(TRANSCRIPT, video_id="missing"))])
    chunks, skipped = build_chunks(collected, vault_id="V")
    assert chunks == []
    assert [path for path, _ in skipped] == ["transcripts/orphan.json"]
    assert "summary" in skipped[0][1]


def test_build_chunks_applies_vault_id_to_markdown_chunks():
    collected = classify([SourceDocument(source_path="notes/a.md", raw=b"# hi\n\nbody")])
    chunks, _ = build_chunks(collected, vault_id="Class Notes")
    assert all(chunk.vault_id == "Class Notes" for chunk in chunks)
