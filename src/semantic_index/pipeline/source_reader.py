from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

from semantic_index.pipeline.types import SourceUnit


class SourceReader:
    def __init__(self, corpus_db_path: Path, unit_text_db_path: Path):
        self.corpus_db_path = corpus_db_path
        self.unit_text_db_path = unit_text_db_path

    def fetch_units(self, cgid: str | None = None) -> dict[int, list[SourceUnit]]:
        unit_text = sqlite3.connect(self.unit_text_db_path)
        unit_text.row_factory = sqlite3.Row

        cgid_filter = ""
        params: list[object] = []
        if cgid is not None:
            cgid_filter = "WHERE r.cgid = ?"
            params.append(cgid)

        query = f"""
            SELECT
                r.cgid AS record_cgid,
                r.record_id,
                r.doc_id,
                r.content_hash,
                ru.unit_no,
                ru.updated_at AS unit_updated_at,
                ut.text,
                ut.updated_at AS text_updated_at
            FROM records r
            JOIN record_units ru ON ru.record_id = r.record_id
            LEFT JOIN unit_text ut
              ON ut.record_id = ru.record_id
             AND ut.unit_no = ru.unit_no
            {cgid_filter}
            ORDER BY r.record_id, ru.unit_no
        """

        units_by_record: dict[int, list[SourceUnit]] = defaultdict(list)
        try:
            unit_text.execute(f"ATTACH DATABASE '{self.corpus_db_path}' AS corpus_db")
            rows = unit_text.execute(
                query.replace("records", "corpus_db.records").replace(
                    "record_units", "corpus_db.record_units"
                ),
                params,
            ).fetchall()
        finally:
            unit_text.close()

        for row in rows:
            units_by_record[int(row["record_id"])].append(
                SourceUnit(
                    cgid=row["record_cgid"],
                    record_id=int(row["record_id"]),
                    doc_id=row["doc_id"],
                    unit_no=int(row["unit_no"]),
                    text=row["text"],
                    content_hash=row["content_hash"],
                    unit_updated_at=row["unit_updated_at"],
                    text_updated_at=row["text_updated_at"],
                )
            )
        return dict(units_by_record)
