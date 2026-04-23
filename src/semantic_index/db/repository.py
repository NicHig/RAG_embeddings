from __future__ import annotations

import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from semantic_index.pipeline.types import WindowRecord


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class SemanticRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_build(
        self,
        *,
        cgid_scope: str | None,
        window_size_units: int,
        window_stride_units: int,
        embedding_model: str,
        embedding_dimensions: int | None,
        normalization_version: str,
        status: str,
        notes: str | None = None,
    ) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO semantic_builds (
                    created_at,
                    cgid_scope,
                    window_size_units,
                    window_stride_units,
                    embedding_model,
                    embedding_dimensions,
                    normalization_version,
                    status,
                    notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now(),
                    cgid_scope,
                    window_size_units,
                    window_stride_units,
                    embedding_model,
                    embedding_dimensions,
                    normalization_version,
                    status,
                    notes,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def complete_build(self, build_id: int, status: str, notes: str | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE semantic_builds
                SET completed_at = ?, status = ?, notes = COALESCE(?, notes)
                WHERE build_id = ?
                """,
                (utc_now(), status, notes, build_id),
            )
            conn.commit()

    def get_build(self, build_id: int) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM semantic_builds WHERE build_id = ?",
                (build_id,),
            ).fetchone()

    def latest_build(self) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM semantic_builds ORDER BY build_id DESC LIMIT 1"
            ).fetchone()

    def insert_windows(self, rows: Iterable[WindowRecord]) -> list[int]:
        rows = list(rows)
        with self.connect() as conn:
            window_ids: list[int] = []
            for row in rows:
                cursor = conn.execute(
                    """
                    INSERT INTO semantic_windows (
                        build_id, cgid, record_id, doc_id, start_unit_no, end_unit_no,
                        segment_index, segment_count,
                        window_text, char_len, token_count_est, text_hash,
                        source_content_hash, source_max_updated_at, source_unit_signature,
                        created_at
                    ) VALUES (
                        :build_id, :cgid, :record_id, :doc_id, :start_unit_no, :end_unit_no,
                        :segment_index, :segment_count,
                        :window_text, :char_len, :token_count_est, :text_hash,
                        :source_content_hash, :source_max_updated_at, :source_unit_signature,
                        :created_at
                    )
                    """,
                    asdict(row),
                )
                window_ids.append(int(cursor.lastrowid))
            conn.commit()
            return window_ids

    def fetch_windows_for_build(self, build_id: int) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(
                conn.execute(
                    """
                    SELECT *
                    FROM semantic_windows
                    WHERE build_id = ?
                    ORDER BY window_id
                    """,
                    (build_id,),
                )
            )

    def fetch_pending_windows(self, build_id: int, limit: int) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(
                conn.execute(
                    """
                    SELECT
                        w.window_id,
                        w.build_id,
                        w.cgid,
                        w.record_id,
                        w.doc_id,
                        w.start_unit_no,
                        w.end_unit_no,
                        w.segment_index,
                        w.segment_count,
                        w.window_text,
                        w.char_len,
                        w.token_count_est,
                        w.text_hash,
                        w.source_content_hash,
                        w.source_max_updated_at,
                        w.source_unit_signature,
                        w.created_at
                    FROM semantic_windows w
                    LEFT JOIN semantic_vectors v ON v.window_id = w.window_id
                    WHERE w.build_id = ? AND v.window_id IS NULL
                    ORDER BY w.window_id
                    LIMIT ?
                    """,
                    (build_id, limit),
                )
            )

    def replace_window_with_segments(
        self, *, window_id: int, segments: list[WindowRecord]
    ) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM semantic_vectors WHERE window_id = ?", (window_id,))
            conn.execute("DELETE FROM semantic_windows WHERE window_id = ?", (window_id,))
            for row in segments:
                conn.execute(
                    """
                    INSERT INTO semantic_windows (
                        build_id, cgid, record_id, doc_id, start_unit_no, end_unit_no,
                        segment_index, segment_count,
                        window_text, char_len, token_count_est, text_hash,
                        source_content_hash, source_max_updated_at, source_unit_signature,
                        created_at
                    ) VALUES (
                        :build_id, :cgid, :record_id, :doc_id, :start_unit_no, :end_unit_no,
                        :segment_index, :segment_count,
                        :window_text, :char_len, :token_count_est, :text_hash,
                        :source_content_hash, :source_max_updated_at, :source_unit_signature,
                        :created_at
                    )
                    """,
                    asdict(row),
                )
            conn.commit()

    def count_windows_for_build(self, build_id: int) -> int:
        with self.connect() as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) AS count FROM semantic_windows WHERE build_id = ?",
                    (build_id,),
                ).fetchone()["count"]
            )

    def count_pending_windows(self, build_id: int) -> int:
        with self.connect() as conn:
            return int(
                conn.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM semantic_windows w
                    LEFT JOIN semantic_vectors v ON v.window_id = w.window_id
                    WHERE w.build_id = ? AND v.window_id IS NULL
                    """,
                    (build_id,),
                ).fetchone()["count"]
            )

    def latest_completed_build_id(self, cgid_scope: str | None = None) -> int | None:
        query = """
            SELECT build_id
            FROM semantic_builds
            WHERE status = 'completed' AND cgid_scope IS ?
            ORDER BY build_id DESC
            LIMIT 1
        """
        with self.connect() as conn:
            row = conn.execute(query, (cgid_scope,)).fetchone()
            return None if row is None else int(row["build_id"])

    def record_state_for_build(self, build_id: int) -> dict[int, tuple[str | None, str, str]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    record_id,
                    source_content_hash,
                    source_max_updated_at,
                    source_unit_signature
                FROM semantic_windows
                WHERE build_id = ?
                GROUP BY record_id
                """,
                (build_id,),
            ).fetchall()
        return {
            int(row["record_id"]): (
                row["source_content_hash"],
                row["source_max_updated_at"],
                row["source_unit_signature"],
            )
            for row in rows
        }

    def insert_vectors(
        self,
        *,
        build_id: int,
        embedding_model: str,
        items: list[tuple[int, int, bytes]],
    ) -> None:
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO semantic_vectors (
                    window_id, build_id, embedding_model, embedding_dimensions,
                    vector_blob, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (window_id, build_id, embedding_model, dims, blob, utc_now())
                    for window_id, dims, blob in items
                ],
            )
            conn.commit()

    def record_error(
        self,
        *,
        build_id: int,
        stage: str,
        error_message: str,
        record_id: int | None = None,
        window_id: int | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO semantic_errors (
                    build_id, record_id, window_id, stage, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (build_id, record_id, window_id, stage, error_message, utc_now()),
            )
            conn.commit()

    def latest_build_id(self) -> int | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT build_id FROM semantic_builds ORDER BY build_id DESC LIMIT 1"
            ).fetchone()
            return None if row is None else int(row["build_id"])

    def delete_records_for_scope(self, record_ids: list[int]) -> None:
        if not record_ids:
            return
        placeholders = ", ".join(["?"] * len(record_ids))
        with self.connect() as conn:
            window_ids = [
                row["window_id"]
                for row in conn.execute(
                    f"SELECT window_id FROM semantic_windows WHERE record_id IN ({placeholders})",
                    record_ids,
                )
            ]
            if window_ids:
                window_placeholders = ", ".join(["?"] * len(window_ids))
                conn.execute(
                    f"DELETE FROM semantic_vectors WHERE window_id IN ({window_placeholders})",
                    window_ids,
                )
            conn.execute(
                f"DELETE FROM semantic_windows WHERE record_id IN ({placeholders})", record_ids
            )
            conn.commit()

    def delete_build_output(self, build_id: int) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM semantic_vectors WHERE build_id = ?", (build_id,))
            conn.execute("DELETE FROM semantic_windows WHERE build_id = ?", (build_id,))
            conn.commit()
