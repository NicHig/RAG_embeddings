from __future__ import annotations

import sqlite3
from pathlib import Path


def initialize_database(db_path: Path) -> None:
    schema_path = Path(__file__).with_name("schema.sql")
    schema = schema_path.read_text()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema)
        _migrate_schema(conn)
        conn.commit()


def _migrate_schema(conn: sqlite3.Connection) -> None:
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(semantic_windows)")
    }
    if "segment_index" not in columns:
        conn.execute(
            "ALTER TABLE semantic_windows ADD COLUMN segment_index INTEGER NOT NULL DEFAULT 0"
        )
    if "segment_count" not in columns:
        conn.execute(
            "ALTER TABLE semantic_windows ADD COLUMN segment_count INTEGER NOT NULL DEFAULT 1"
        )
