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

    def replace(
        self,
        delete_paths: Iterable[str],
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        """Delete `delete_paths` and upsert `chunks`/`vectors` as one atomic write.

        Both the deletes and the inserts land, or neither does: every vector
        is validated against the store's dimensionality before any row is
        touched, and any exception during the write - a bad vector that slips
        past validation, a locked database, disk full - triggers a full
        `rollback()` before re-raising. The store is left exactly as it was
        on entry; it is never partially emptied. Returns the number of
        `delete_paths` that had at least one chunk deleted.
        """

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
