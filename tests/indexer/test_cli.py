import json
from pathlib import Path

from notes_rag.embed.fake import FakeEmbedder
from notes_rag.indexer import cli
from notes_rag.store.sqlite_vec import SqliteVecStore

DIMS = 8


def _write_summary(path, *, video_id="vid1"):
    path.write_text(
        json.dumps(
            {
                "video_id": video_id,
                "title": "Test Video",
                "channel": "Test Channel",
                "url": f"https://example.com/watch?v={video_id}",
                "summary": {
                    "verdict": "Good stuff, worth the watch for anyone curious about the topic.",
                    "tldr": "A short walkthrough of the subject matter from start to finish.",
                    "takeaways": [
                        "Point one is quite important.",
                        "Point two matters too.",
                    ],
                    "sections": [
                        {
                            "start_seconds": 0,
                            "title": "Intro",
                            "summary": "Sets the stage for what is covered in the rest of the video.",
                        },
                        {
                            "start_seconds": 120,
                            "title": "Deep dive",
                            "summary": "Goes into detail about the mechanics underlying the topic.",
                        },
                    ],
                },
            }
        )
    )


def _write_transcript(path, *, video_id="vid1"):
    path.write_text(
        json.dumps(
            {
                "video_id": video_id,
                "segments": [
                    {
                        "start_seconds": 0,
                        "text": "Welcome to the video, let's get started right away.",
                    },
                    {
                        "start_seconds": 120,
                        "text": "Now we go into the deep dive on the underlying mechanics.",
                    },
                ],
            }
        )
    )


def _build_source_tree(tmp_path):
    source = tmp_path / "source"
    (source / "summaries").mkdir(parents=True)
    (source / "transcripts").mkdir(parents=True)
    (source / "notes").mkdir(parents=True)

    _write_summary(source / "summaries" / "vid1.json")
    _write_transcript(source / "transcripts" / "vid1.json")
    # No matching summary for this video_id - exercises the skip path.
    _write_transcript(source / "transcripts" / "orphan.json", video_id="orphan")

    (source / "notes" / "Alpha.md").write_text(
        "# Alpha\n\n"
        "This note links to [[Beta]] for more detail on the topic being discussed here.\n"
    )
    (source / "notes" / "Beta.md").write_text(
        "# Beta\n\n"
        "Beta covers the material referenced from Alpha in more depth than a summary allows.\n"
    )
    return source


def test_cli_builds_a_queryable_index_from_all_three_artifact_shapes(tmp_path, capsys):
    source = _build_source_tree(tmp_path)
    db_path = tmp_path / "index.db"

    exit_code = cli.main(
        [
            str(source),
            "--out",
            str(db_path),
            "--vault-id",
            "TestVault",
            "--dimensions",
            str(DIMS),
            "--fake-embedder",
        ]
    )
    assert exit_code == 0

    store = SqliteVecStore(db_path, dimensions=DIMS)
    try:
        paths = store.all_source_paths()
        assert paths == {
            "summaries/vid1.json",
            "transcripts/vid1.json",
            "notes/Alpha.md",
            "notes/Beta.md",
        }

        # A large k with any query vector returns every row regardless of
        # similarity (OVERFETCH=8 comfortably covers this handful of chunks) -
        # this is "a search returns them", not a semantic-relevance check.
        embedder = FakeEmbedder(dimensions=DIMS)
        all_hits = store.search(embedder.embed(["query"])[0], k=50)
        assert {hit.chunk.source_path for hit in all_hits} == paths

        beta_chunks = [hit.chunk for hit in all_hits if hit.chunk.source_path == "notes/Beta.md"]
        assert beta_chunks
        assert all(chunk.backlinks == ("Alpha",) for chunk in beta_chunks)
    finally:
        store.close()

    err = capsys.readouterr().err
    assert "orphan.json" in err
    assert "no summary found" in err


def test_cli_reports_build_stats_on_stdout(tmp_path, capsys):
    source = _build_source_tree(tmp_path)
    db_path = tmp_path / "index.db"

    cli.main(
        [
            str(source),
            "--out",
            str(db_path),
            "--vault-id",
            "TestVault",
            "--dimensions",
            str(DIMS),
            "--fake-embedder",
        ]
    )

    out = capsys.readouterr().out
    assert "chunks_written:" in out
    assert "vectors_embedded:" in out
    assert "vectors_reused:" in out
    assert "paths_deleted:" in out


def test_cli_skips_an_unsupported_suffix_file_without_reading_its_bytes(
    tmp_path, capsys, monkeypatch
):
    """Mirrors the same guard in the Lambda handler: the pre-refactor `_collect`
    only ever read `.md` and `.json` files, but a naive rewrite that walks
    every file under `source` and calls `read_bytes()` unconditionally would
    load an arbitrarily large unsupported file into memory just to discover
    it can't be used. The suffix must be decided from the path alone, before
    any read."""
    source = tmp_path / "source"
    (source / "notes").mkdir(parents=True)
    unsupported = source / "notes" / "video.mp4"
    unsupported.write_bytes(b"\x00\x01\x02\x03")

    read_paths: list[Path] = []
    original_read_bytes = Path.read_bytes

    def recording_read_bytes(self):
        read_paths.append(self)
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", recording_read_bytes)

    db_path = tmp_path / "index.db"
    exit_code = cli.main(
        [
            str(source),
            "--out",
            str(db_path),
            "--vault-id",
            "TestVault",
            "--dimensions",
            str(DIMS),
            "--fake-embedder",
        ]
    )
    assert exit_code == 0
    assert unsupported not in read_paths, "bytes were read for an unsupported suffix"

    err = capsys.readouterr().err
    assert "video.mp4" in err
    assert "unhandled file suffix" in err


def test_transcript_without_a_matching_summary_is_skipped_not_indexed(tmp_path, capsys):
    source = tmp_path / "source"
    (source / "transcripts").mkdir(parents=True)
    _write_transcript(source / "transcripts" / "lonely.json", video_id="lonely")
    db_path = tmp_path / "index.db"

    exit_code = cli.main(
        [
            str(source),
            "--out",
            str(db_path),
            "--vault-id",
            "TestVault",
            "--dimensions",
            str(DIMS),
            "--fake-embedder",
        ]
    )
    assert exit_code == 0

    store = SqliteVecStore(db_path, dimensions=DIMS)
    try:
        assert store.all_source_paths() == set()
    finally:
        store.close()

    err = capsys.readouterr().err
    assert "lonely.json" in err
    assert "no summary found" in err
