from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from semantic_index.config.settings import Settings


def _create_corpus_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE records (
                record_id INTEGER PRIMARY KEY,
                doc_id TEXT NOT NULL UNIQUE,
                content_hash TEXT,
                unit_count INTEGER,
                cgid TEXT
            );
            CREATE TABLE record_units (
                record_id INTEGER NOT NULL,
                unit_no INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                cgid TEXT,
                PRIMARY KEY(record_id, unit_no)
            );
            CREATE INDEX idx_records_cgid ON records(cgid);
            CREATE INDEX idx_record_units_cgid ON record_units(cgid);
            """
        )
        conn.executemany(
            """
            INSERT INTO records (record_id, doc_id, content_hash, unit_count, cgid)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (1, "doc-1", "hash-1", 3, "SAMPLE-001"),
                (2, "doc-2", "hash-2", 2, "SAMPLE-002"),
            ],
        )
        conn.executemany(
            """
            INSERT INTO record_units (record_id, unit_no, updated_at, cgid)
            VALUES (?, ?, ?, ?)
            """,
            [
                (1, 1, "2026-04-23T00:00:00+00:00", "SAMPLE-001"),
                (1, 2, "2026-04-23T00:00:00+00:00", "SAMPLE-001"),
                (1, 3, "2026-04-23T00:00:00+00:00", "SAMPLE-001"),
                (2, 1, "2026-04-23T00:00:00+00:00", "SAMPLE-002"),
                (2, 2, "2026-04-23T00:00:00+00:00", "SAMPLE-002"),
            ],
        )
        conn.commit()


def _create_unit_text_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE unit_text (
                record_id INTEGER NOT NULL,
                unit_no INTEGER NOT NULL,
                text TEXT,
                updated_at TEXT NOT NULL,
                cgid TEXT,
                PRIMARY KEY(record_id, unit_no)
            );
            CREATE INDEX idx_unit_text_cgid ON unit_text(cgid);
            """
        )
        conn.executemany(
            """
            INSERT INTO unit_text (record_id, unit_no, text, updated_at, cgid)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (1, 1, "alpha page", "2026-04-23T00:00:00+00:00", "SAMPLE-001"),
                (1, 2, "beta page", "2026-04-23T00:00:00+00:00", "SAMPLE-001"),
                (1, 3, "gamma page", "2026-04-23T00:00:00+00:00", "SAMPLE-001"),
                (2, 1, "delta page", "2026-04-23T00:00:00+00:00", "SAMPLE-002"),
                (2, 2, None, "2026-04-23T00:00:00+00:00", "SAMPLE-002"),
            ],
        )
        conn.commit()


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    corpus_db = tmp_path / "corpus.sqlite"
    unit_text_db = tmp_path / "unit_text.sqlite"
    semantic_db = tmp_path / "semantic_index.sqlite"
    _create_corpus_db(corpus_db)
    _create_unit_text_db(unit_text_db)
    return Settings(
        corpus_db_path=corpus_db,
        unit_text_db_path=unit_text_db,
        semantic_index_db_path=semantic_db,
        openai_api_key="test-key",
        embedding_mode="deterministic",
    )
