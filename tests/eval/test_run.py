import pytest

import eval.run as run_module
from eval.run import Expectation, Question, evaluate, load_questions, matches
from notes_rag.embed.fake import FakeEmbedder
from notes_rag.models import Chunk
from notes_rag.store.base import SearchHit
from notes_rag.store.sqlite_vec import SqliteVecStore

DIMS = 8


def video_chunk(chunk_id: str, *, start: int, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        corpus="video",
        vault_id=None,
        source_path="summaries/vid.json",
        chunk_type="summary",
        title="T",
        heading="H",
        context="CTX",
        text=text,
        content_hash=f"hash-{chunk_id}",
        video_id="vid",
        start_seconds=start,
        url="https://example.com",
    )


def hit(chunk: Chunk, distance: float = 0.1) -> SearchHit:
    return SearchHit(chunk=chunk, distance=distance)


def test_matches_when_start_seconds_falls_inside_the_span():
    expectation = Expectation(corpus="video", video_id="vid", start_seconds=(1000, 1500))
    assert matches(hit(video_chunk("a", start=1120, text="x")), expectation)


def test_does_not_match_outside_the_span():
    expectation = Expectation(corpus="video", video_id="vid", start_seconds=(1000, 1500))
    assert not matches(hit(video_chunk("a", start=200, text="x")), expectation)


def test_span_boundaries_are_inclusive():
    expectation = Expectation(corpus="video", video_id="vid", start_seconds=(1000, 1500))
    assert matches(hit(video_chunk("a", start=1000, text="x")), expectation)
    assert matches(hit(video_chunk("a", start=1500, text="x")), expectation)


def test_does_not_match_a_different_video():
    expectation = Expectation(corpus="video", video_id="other", start_seconds=(0, 9999))
    assert not matches(hit(video_chunk("a", start=100, text="x")), expectation)


def test_matches_a_note_by_source_path():
    chunk = Chunk(
        id="n",
        corpus="note",
        vault_id="V",
        source_path="Class Notes/a.md",
        chunk_type="note",
        title="a",
        heading=None,
        context="CTX",
        text="x",
        content_hash="h",
    )
    expectation = Expectation(corpus="note", source_path="Class Notes/a.md")
    assert matches(hit(chunk), expectation)


def test_does_not_match_a_different_corpus():
    chunk = video_chunk("a", start=1120, text="x")
    expectation = Expectation(corpus="note", video_id="vid", start_seconds=(1000, 1500))
    assert not matches(hit(chunk), expectation)


def test_does_not_match_a_different_source_path():
    chunk = Chunk(
        id="n",
        corpus="note",
        vault_id="V",
        source_path="Class Notes/a.md",
        chunk_type="note",
        title="a",
        heading=None,
        context="CTX",
        text="x",
        content_hash="h",
    )
    expectation = Expectation(corpus="note", source_path="Class Notes/b.md")
    assert not matches(hit(chunk), expectation)


def test_does_not_match_when_chunk_has_no_start_seconds():
    chunk = Chunk(
        id="n",
        corpus="video",
        vault_id=None,
        source_path="summaries/vid.json",
        chunk_type="summary",
        title="T",
        heading="H",
        context="CTX",
        text="x",
        content_hash="h",
        video_id="vid",
        start_seconds=None,
        url="https://example.com",
    )
    expectation = Expectation(corpus="video", video_id="vid", start_seconds=(1000, 1500))
    assert not matches(hit(chunk), expectation)


def test_load_questions_parses_a_video_expectation(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text(
        "- id: q001\n"
        "  question: What about custom schedulers?\n"
        "  expects:\n"
        "    - corpus: video\n"
        "      video_id: vid\n"
        "      start_seconds: [1120, 1400]\n"
    )
    questions = load_questions(path)
    assert len(questions) == 1
    assert questions[0].id == "q001"
    assert questions[0].expects[0].start_seconds == (1120, 1400)


def test_load_questions_parses_a_note_expectation(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text(
        "- id: q002\n"
        "  question: What is affinity?\n"
        "  expects:\n"
        "    - corpus: note\n"
        "      source_path: Class Notes/a.md\n"
    )
    questions = load_questions(path)
    assert questions[0].expects[0].source_path == "Class Notes/a.md"
    assert questions[0].expects[0].start_seconds is None


@pytest.fixture
def populated_store(tmp_path):
    store = SqliteVecStore(tmp_path / "eval.db", dimensions=DIMS)
    embedder = FakeEmbedder(dimensions=DIMS)
    chunks = [
        video_chunk("a", start=1120, text="writing a custom scheduler"),
        video_chunk("b", start=0, text="introduction to clusters"),
    ]
    store.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))
    yield store, embedder
    store.close()


def test_evaluate_scores_a_hit_at_rank_one(populated_store):
    store, embedder = populated_store
    questions = [
        Question(
            id="q1",
            question="writing a custom scheduler",
            expects=[Expectation(corpus="video", video_id="vid", start_seconds=(1100, 1200))],
        )
    ]
    report = evaluate(questions, store, embedder, k=2)
    assert report.recall_at_k == 1.0
    assert report.mrr == 1.0


def test_evaluate_scores_a_miss(populated_store):
    store, embedder = populated_store
    questions = [
        Question(
            id="q1",
            question="writing a custom scheduler",
            expects=[Expectation(corpus="video", video_id="vid", start_seconds=(9000, 9999))],
        )
    ]
    report = evaluate(questions, store, embedder, k=2)
    assert report.recall_at_k == 0.0
    assert report.mrr == 0.0


def test_mrr_halves_for_a_hit_at_rank_two(populated_store):
    store, embedder = populated_store
    questions = [
        Question(
            id="q1",
            question="writing a custom scheduler",
            expects=[Expectation(corpus="video", video_id="vid", start_seconds=(0, 0))],
        )
    ]
    report = evaluate(questions, store, embedder, k=2)
    assert report.recall_at_k == 1.0
    assert report.mrr == pytest.approx(0.5)


def test_per_question_results_are_reported(populated_store):
    store, embedder = populated_store
    questions = [
        Question(
            id="q1",
            question="writing a custom scheduler",
            expects=[Expectation(corpus="video", video_id="vid", start_seconds=(1100, 1200))],
        )
    ]
    report = evaluate(questions, store, embedder, k=2)
    assert len(report.per_question) == 1
    assert report.per_question[0].question_id == "q1"
    assert report.per_question[0].rank == 1


def test_evaluate_with_no_questions_returns_zero(populated_store):
    store, embedder = populated_store
    report = evaluate([], store, embedder, k=2)
    assert report.recall_at_k == 0.0
    assert report.mrr == 0.0


def _patch_titan_embedder(monkeypatch, *, dimensions=DIMS):
    # main() imports TitanEmbedder lazily from notes_rag.embed.bedrock inside
    # its own body. Patching the attribute on that module, rather than on
    # eval.run, is what the lazy `from ... import TitanEmbedder` resolves at
    # call time - this must never construct a real client or touch AWS.
    from notes_rag.embed import bedrock as bedrock_module

    monkeypatch.setattr(
        bedrock_module,
        "TitanEmbedder",
        lambda *args, **kwargs: FakeEmbedder(dimensions=dimensions),
    )


def _write_main_index_and_questions(tmp_path, *, start_seconds, dimensions=DIMS):
    db_path = tmp_path / "eval.db"
    store = SqliteVecStore(db_path, dimensions=dimensions)
    embedder = FakeEmbedder(dimensions=dimensions)
    chunks = [video_chunk("a", start=1120, text="writing a custom scheduler")]
    store.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))
    store.close()

    questions_path = tmp_path / "q.yaml"
    low, high = start_seconds
    questions_path.write_text(
        "- id: q1\n"
        "  question: writing a custom scheduler\n"
        "  expects:\n"
        "    - corpus: video\n"
        "      video_id: vid\n"
        f"      start_seconds: [{low}, {high}]\n"
    )
    return db_path, questions_path


def test_main_returns_nonzero_when_recall_is_below_min_recall(tmp_path, monkeypatch):
    # main() opens the index with SqliteVecStore(args.index) - no dimensions
    # override, so it always assumes the real Titan width (1024). The fixture
    # index must be written at that same width or SqliteVecStore's opening
    # check (added alongside the atomic replace() fix) correctly rejects it.
    _patch_titan_embedder(monkeypatch, dimensions=1024)
    # A span that cannot match the chunk's start_seconds=1120 forces a miss,
    # so recall@k is 0.0 - below the 0.5 threshold.
    db_path, questions_path = _write_main_index_and_questions(
        tmp_path, start_seconds=(9000, 9999), dimensions=1024
    )

    exit_code = run_module.main(
        [
            "--index",
            str(db_path),
            "--questions",
            str(questions_path),
            "--k",
            "2",
            "--min-recall",
            "0.5",
        ]
    )
    assert exit_code == 1


def test_main_returns_zero_when_recall_meets_min_recall(tmp_path, monkeypatch):
    # See the width comment on the sibling test above: main() always opens
    # at the default 1024 dimensions, so the fixture index must match.
    _patch_titan_embedder(monkeypatch, dimensions=1024)
    # A span that matches the chunk's start_seconds=1120 forces a hit, so
    # recall@k is 1.0 - at or above the 0.5 threshold.
    db_path, questions_path = _write_main_index_and_questions(
        tmp_path, start_seconds=(1100, 1200), dimensions=1024
    )

    exit_code = run_module.main(
        [
            "--index",
            str(db_path),
            "--questions",
            str(questions_path),
            "--k",
            "2",
            "--min-recall",
            "0.5",
        ]
    )
    assert exit_code == 0
