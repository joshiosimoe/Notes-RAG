"""Retrieval evaluation: recall@k and MRR against a golden question set.

Deterministic and free - no LLM judge, no generation. This is the fast inner
loop that runs on every commit; groundedness and citation scoring live behind
a separate --judge flag (not in this plan).

Known limitation: ~15 questions over 2 videos moves recall@k in ~6.7%
increments. This detects "something broke", not "this is 3% better". Report it
as a regression tripwire, not a measurement.
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from notes_rag.embed.base import Embedder
from notes_rag.store.base import SearchHit, VectorStore


@dataclass(frozen=True)
class Expectation:
    corpus: str
    video_id: str | None = None
    start_seconds: tuple[int, int] | None = None
    source_path: str | None = None


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    expects: list[Expectation] = field(default_factory=list)


@dataclass(frozen=True)
class QuestionResult:
    question_id: str
    rank: int | None  # 1-based rank of the first relevant hit, None if missed


@dataclass(frozen=True)
class EvalReport:
    recall_at_k: float
    mrr: float
    per_question: list[QuestionResult]


def load_questions(path: str | Path) -> list[Question]:
    raw = yaml.safe_load(Path(path).read_text()) or []
    questions: list[Question] = []
    for entry in raw:
        expects = [
            Expectation(
                corpus=item["corpus"],
                video_id=item.get("video_id"),
                start_seconds=(tuple(item["start_seconds"]) if item.get("start_seconds") else None),
                source_path=item.get("source_path"),
            )
            for item in entry.get("expects", [])
        ]
        questions.append(Question(id=entry["id"], question=entry["question"], expects=expects))
    return questions


def matches(hit: SearchHit, expectation: Expectation) -> bool:
    chunk = hit.chunk
    if chunk.corpus != expectation.corpus:
        return False
    if expectation.video_id is not None and chunk.video_id != expectation.video_id:
        return False
    if expectation.source_path is not None and chunk.source_path != expectation.source_path:
        return False
    if expectation.start_seconds is not None:
        if chunk.start_seconds is None:
            return False
        low, high = expectation.start_seconds
        if not low <= chunk.start_seconds <= high:
            return False
    return True


def evaluate(
    questions: Sequence[Question],
    store: VectorStore,
    embedder: Embedder,
    *,
    k: int = 6,
) -> EvalReport:
    if not questions:
        return EvalReport(recall_at_k=0.0, mrr=0.0, per_question=[])

    results: list[QuestionResult] = []
    for question in questions:
        vector = embedder.embed([question.question])[0]
        hits = store.search(vector, k=k)
        rank = _first_relevant_rank(hits, question.expects)
        results.append(QuestionResult(question_id=question.id, rank=rank))

    hit_count = sum(1 for result in results if result.rank is not None)
    reciprocal = sum(1.0 / result.rank for result in results if result.rank is not None)
    return EvalReport(
        recall_at_k=hit_count / len(results),
        mrr=reciprocal / len(results),
        per_question=results,
    )


def _first_relevant_rank(hits: Sequence[SearchHit], expects: Sequence[Expectation]) -> int | None:
    for index, hit in enumerate(hits, start=1):
        if any(matches(hit, expectation) for expectation in expects):
            return index
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation.")
    parser.add_argument("--index", required=True, help="path to the .db index")
    parser.add_argument("--questions", default="eval/questions.yaml")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.0,
        help="exit non-zero if recall@k falls below this (use in CI)",
    )
    args = parser.parse_args(argv)

    from notes_rag.embed.bedrock import TitanEmbedder
    from notes_rag.store.sqlite_vec import SqliteVecStore

    store = SqliteVecStore(args.index)
    try:
        report = evaluate(load_questions(args.questions), store, TitanEmbedder(), k=args.k)
    finally:
        store.close()

    print(f"recall@{args.k}: {report.recall_at_k:.3f}")
    print(f"MRR:       {report.mrr:.3f}")
    for result in report.per_question:
        status = f"rank {result.rank}" if result.rank else "MISS"
        print(f"  {result.question_id}: {status}")

    if report.recall_at_k < args.min_recall:
        print(
            f"FAIL: recall@{args.k} {report.recall_at_k:.3f} below threshold {args.min_recall:.3f}"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
