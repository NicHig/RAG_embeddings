import sqlite3
from types import SimpleNamespace

import pytest

from semantic_index.build import runner
from semantic_index.build.runner import (
    AdaptiveRateController,
    _parse_rate_limit_tpm_limit,
    _parse_retry_after_seconds,
    build_incremental,
    build_plaintiff,
    init_db,
    resume_build,
    validate_build,
)


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
    assert first_incremental["record_count"] == 2

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


def test_resume_build_embeds_only_pending_windows(settings):
    init_db(settings)
    build_result = build_plaintiff(settings, cgid="SAMPLE-001")

    with sqlite3.connect(settings.semantic_index_db_path) as conn:
        conn.execute(
            "DELETE FROM semantic_vectors WHERE window_id = (SELECT window_id FROM semantic_windows LIMIT 1)"
        )
        conn.execute(
            "UPDATE semantic_builds SET status = 'failed', completed_at = NULL WHERE build_id = ?",
            (build_result["build_id"],),
        )
        conn.commit()

    resumed = resume_build(settings, build_id=build_result["build_id"])
    assert resumed["status"] == "completed"
    assert resumed["processed_windows"] == 1
    assert resumed["pending_windows"] == 0


def test_failed_embedding_marks_build_failed(settings, monkeypatch):
    class FailingEmbedder:
        def embed_texts(self, texts):
            raise RuntimeError("embedding exploded")

    monkeypatch.setattr(runner, "_get_embedder", lambda settings: FailingEmbedder())

    init_db(settings)
    with pytest.raises(RuntimeError, match="embedding exploded"):
        runner.build_plaintiff(settings, cgid="SAMPLE-001")

    with sqlite3.connect(settings.semantic_index_db_path) as conn:
        row = conn.execute(
            "SELECT status, notes FROM semantic_builds ORDER BY build_id DESC LIMIT 1"
        ).fetchone()
    assert row[0] == "failed"
    assert "embedding exploded" in row[1]


def test_rate_limit_parsing_helpers():
    message = (
        "Rate limit reached for text-embedding-3-small on tokens per min (TPM): "
        "Limit 1000000, Used 882346, Requested 150974. Please try again in 1.999s."
    )
    assert _parse_rate_limit_tpm_limit(message) == 1000000
    assert _parse_retry_after_seconds(message) == 1.999


def test_rate_limit_controller_shrinks_after_429(settings):
    controller = AdaptiveRateController(settings)
    error = SimpleNamespace(
        body={
            "error": {
                "message": (
                    "Rate limit reached for text-embedding-3-small on tokens per min (TPM): "
                    "Limit 1000000, Used 882346, Requested 150974. Please try again in 1.999s."
                )
            }
        }
    )
    sleep_for = controller.record_rate_limit(error)  # type: ignore[arg-type]
    assert controller.tpm_limit == 1000000
    assert controller.effective_batch_size() == 50
    assert controller.effective_token_budget() == 125000
    assert sleep_for >= 2.0
