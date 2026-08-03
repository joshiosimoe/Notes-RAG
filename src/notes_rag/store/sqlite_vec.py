"""sqlite-vec backed VectorStore.

The whole index is one file: metadata in a normal table, vectors in a vec0
virtual table joined by rowid. At ~20k chunks a brute-force scan is
milliseconds, so no ANN tuning is required.
"""

import json
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path

import sqlite_vec

from notes_rag.models import Chunk
from notes_rag.store.base import SearchHit

# vec0 MATCH cannot express our metadata filters, so we over-fetch and filter
# after joining. See the design note in the plan for why this is acceptable.
OVERFETCH = 8

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    rowid          INTEGER PRIMARY KEY AUTOINCREMENT,
    id             TEXT UNIQUE NOT NULL,
    corpus         TEXT NOT NULL,
    vault_id       TEXT,
    source_path    TEXT NOT NULL,
    chunk_type     TEXT NOT NULL,
    title          TEXT NOT NULL,
    heading        TEXT,
    context        TEXT NOT NULL,
    text           TEXT NOT NULL,
    content_hash   TEXT NOT NULL,
    video_id       TEXT,
    start_seconds  INTEGER,
    url            TEXT,
    links_to       TEXT NOT NULL DEFAULT '[]',
    backlinks      TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_chunks_source_path ON chunks(source_path);
CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_corpus ON chunks(corpus);
"""

_COLUMNS = (
    "id, corpus, vault_id, source_path, chunk_type, title, heading, context, "
    "text, content_hash, video_id, start_seconds, url, links_to, backlinks"
)


class SqliteVecStore:
    def __init__(self, path: str | Path, *, dimensions: int = 1024) -> None:
        self.path = Path(path)
        self.dimensions = dimensions
        self._db = sqlite3.connect(self.path)
        self._db.enable_load_extension(True)
        sqlite_vec.load(self._db)
        self._db.enable_load_extension(False)
        self._db.executescript(_SCHEMA)
        self._db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks "
            f"USING vec0(embedding float[{dimensions}])"
        )
        self._db.commit()

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks and vectors must be the same length; got {len(chunks)} and {len(vectors)}"
            )
        # Validate every pair before issuing any write, so a malformed batch
        # raises without touching the database at all.
        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"vector for {chunk.id} has {len(vector)} dimensions, "
                    f"expected {self.dimensions}"
                )
        try:
            for chunk, vector in zip(chunks, vectors, strict=True):
                self._delete_ids([chunk.id])
                cursor = self._db.execute(
                    f"INSERT INTO chunks ({_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        chunk.id,
                        chunk.corpus,
                        chunk.vault_id,
                        chunk.source_path,
                        chunk.chunk_type,
                        chunk.title,
                        chunk.heading,
                        chunk.context,
                        chunk.text,
                        chunk.content_hash,
                        chunk.video_id,
                        chunk.start_seconds,
                        chunk.url,
                        json.dumps(list(chunk.links_to)),
                        json.dumps(list(chunk.backlinks)),
                    ),
                )
                self._db.execute(
                    "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                    (cursor.lastrowid, sqlite_vec.serialize_float32(list(vector))),
                )
        except Exception:
            self._db.rollback()
            raise
        self._db.commit()

    def delete_by_path(self, source_path: str) -> int:
        rows = self._db.execute(
            "SELECT id FROM chunks WHERE source_path = ?", (source_path,)
        ).fetchall()
        ids = [row[0] for row in rows]
        self._delete_ids(ids)
        self._db.commit()
        return len(ids)

    def search(
        self,
        vector: Sequence[float],
        k: int,
        *,
        corpus: str | None = None,
        vault_id: str | None = None,
        chunk_type: str | None = None,
    ) -> list[SearchHit]:
        rows = self._db.execute(
            "SELECT rowid, distance FROM vec_chunks "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (sqlite_vec.serialize_float32(list(vector)), k * OVERFETCH),
        ).fetchall()
        if not rows:
            return []

        distances = {rowid: distance for rowid, distance in rows}
        placeholders = ",".join("?" * len(distances))
        clauses = [f"rowid IN ({placeholders})"]
        params: list[object] = list(distances)
        for column, value in (
            ("corpus", corpus),
            ("vault_id", vault_id),
            ("chunk_type", chunk_type),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)

        records = self._db.execute(
            f"SELECT rowid, {_COLUMNS} FROM chunks WHERE {' AND '.join(clauses)}",
            params,
        ).fetchall()

        hits = [
            SearchHit(chunk=_to_chunk(record[1:]), distance=distances[record[0]])
            for record in records
        ]
        hits.sort(key=lambda hit: hit.distance)
        return hits[:k]

    def cached_vectors(self, hashes: Iterable[str]) -> dict[str, list[float]]:
        wanted = list(hashes)
        if not wanted:
            return {}
        placeholders = ",".join("?" * len(wanted))
        rows = self._db.execute(
            f"SELECT c.content_hash, vec_to_json(v.embedding) "
            f"FROM chunks c JOIN vec_chunks v ON v.rowid = c.rowid "
            f"WHERE c.content_hash IN ({placeholders})",
            wanted,
        ).fetchall()
        return {content_hash: json.loads(raw) for content_hash, raw in rows}

    def all_source_paths(self) -> set[str]:
        rows = self._db.execute("SELECT DISTINCT source_path FROM chunks").fetchall()
        return {row[0] for row in rows}

    def copy_filtered(self, dest: str | Path, *, corpus: str) -> None:
        """Write a new database containing only chunks from one corpus.

        This is how public.db is produced from full.db. Physical separation,
        not a query predicate: the demo Lambda's IAM role can read only the
        resulting file, so a filter bug cannot leak other corpora.
        """
        target = SqliteVecStore(dest, dimensions=self.dimensions)
        try:
            rows = self._db.execute(
                f"SELECT c.rowid, {_COLUMNS}, vec_to_json(v.embedding) "
                f"FROM chunks c JOIN vec_chunks v ON v.rowid = c.rowid "
                f"WHERE c.corpus = ?",
                (corpus,),
            ).fetchall()
            for record in rows:
                chunk = _to_chunk(record[1:-1])
                target.upsert([chunk], [json.loads(record[-1])])
        finally:
            target.close()

    def close(self) -> None:
        self._db.close()

    def _delete_ids(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        rows = self._db.execute(
            f"SELECT rowid FROM chunks WHERE id IN ({placeholders})", list(ids)
        ).fetchall()
        for (rowid,) in rows:
            self._db.execute("DELETE FROM vec_chunks WHERE rowid = ?", (rowid,))
        self._db.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", list(ids))


def _to_chunk(record: Sequence) -> Chunk:
    return Chunk(
        id=record[0],
        corpus=record[1],
        vault_id=record[2],
        source_path=record[3],
        chunk_type=record[4],
        title=record[5],
        heading=record[6],
        context=record[7],
        text=record[8],
        content_hash=record[9],
        video_id=record[10],
        start_seconds=record[11],
        url=record[12],
        links_to=tuple(json.loads(record[13])),
        backlinks=tuple(json.loads(record[14])),
    )
