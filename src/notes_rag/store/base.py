"""Storage interface. One implementation today (sqlite-vec); the interface exists
so a pgvector implementation can be benchmarked against it without touching callers."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from notes_rag.models import Chunk


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    distance: float


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Insert or replace chunks. `chunks` and `vectors` must be the same length."""

    def delete_by_path(self, source_path: str) -> int:
        """Delete every chunk for a source path. Returns the number deleted."""

    def search(
        self,
        vector: Sequence[float],
        k: int,
        *,
        corpus: str | None = None,
        vault_id: str | None = None,
        chunk_type: str | None = None,
    ) -> list[SearchHit]:
        """Nearest neighbours, optionally filtered, ordered by ascending distance."""

    def cached_vectors(self, hashes: Iterable[str]) -> dict[str, list[float]]:
        """Vectors already stored for these content hashes. Missing hashes are omitted."""

    def all_source_paths(self) -> set[str]:
        """Every distinct source path currently indexed."""

    def close(self) -> None: ...
