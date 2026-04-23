from __future__ import annotations

import sqlite3
from pathlib import Path


def initialize_database(db_path: Path) -> None:
    schema_path = Path(__file__).with_name("schema.sql")
    schema = schema_path.read_text()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema)
        conn.commit()
