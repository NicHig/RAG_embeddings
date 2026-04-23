import sqlite3

from semantic_index.build.runner import build_incremental, build_plaintiff, init_db, validate_build


def test_plaintiff_build_creates_scoped_windows_and_vectors(settings):
    init_db(settings)
    result = build_plaintiff(settings, cgid="SAMPLE-001")
    assert result["status"] == "completed"
    assert result["record_count"] == 1
    assert result["window_count"] == 1

    with sqlite3.connect(settings.semantic_index_db_path) as conn:
        window_count = conn.execute("SELECT COUNT(*) FROM semantic_windows").fetchone()[0]
        vector_count = conn.execute("SELECT COUNT(*) FROM semantic_vectors").fetchone()[0]
        cgid = conn.execute("SELECT cgid FROM semantic_windows").fetchone()[0]

    assert window_count == 1
    assert vector_count == 1
    assert cgid == "SAMPLE-001"


def test_validate_latest_build(settings):
    init_db(settings)
    build_plaintiff(settings, cgid="SAMPLE-001")
    result = validate_build(settings)
    assert result["valid"] is True
    assert result["window_count"] == 1


def test_incremental_build_only_rebuilds_dirty_records(settings):
    init_db(settings)
    build_plaintiff(settings, cgid="SAMPLE-001")
    first_incremental = build_incremental(settings)
    assert first_incremental["record_count"] == 1

    with sqlite3.connect(settings.unit_text_db_path) as conn:
        conn.execute(
            """
            UPDATE unit_text
            SET text = ?, updated_at = ?
            WHERE record_id = ? AND unit_no = ?
            """,
            ("delta page changed", "2026-04-24T00:00:00+00:00", 2, 2),
        )
        conn.commit()

    second_incremental = build_incremental(settings)
    assert second_incremental["record_count"] == 1
    assert second_incremental["window_count"] == 1
