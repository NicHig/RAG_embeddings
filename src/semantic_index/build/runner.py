from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from semantic_index.config.settings import Settings
from semantic_index.db.init_db import initialize_database
from semantic_index.db.repository import SemanticRepository
from semantic_index.pipeline.embedder import DeterministicEmbedder, OpenAIEmbedder
from semantic_index.pipeline.source_reader import SourceReader
from semantic_index.pipeline.validator import validate_build_output
from semantic_index.pipeline.vector_store import vector_to_blob
from semantic_index.pipeline.window_builder import build_windows

LOGGER = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(Path("logs") / "semantic_index.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def _get_embedder(settings: Settings):
    if settings.embedding_mode == "sync":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for sync embedding mode")
        return OpenAIEmbedder(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
            dimensions=settings.openai_embedding_dimensions,
        )
    if settings.embedding_mode == "deterministic":
        return DeterministicEmbedder()
    raise ValueError(f"Unsupported embedding mode: {settings.embedding_mode}")


def init_db(settings: Settings) -> dict[str, object]:
    settings.validate_output_path()
    _configure_logging(settings.log_level)
    initialize_database(settings.semantic_index_db_path)
    return {"status": "ok", "semantic_index_db_path": str(settings.semantic_index_db_path)}


def build_full(settings: Settings) -> dict[str, object]:
    return _run_build(settings, cgid=None, incremental=False)


def build_plaintiff(settings: Settings, *, cgid: str) -> dict[str, object]:
    return _run_build(settings, cgid=cgid, incremental=False)


def build_incremental(settings: Settings) -> dict[str, object]:
    return _run_build(settings, cgid=None, incremental=True)


def resume_build(settings: Settings, *, build_id: int | None = None) -> dict[str, object]:
    settings.validate()
    _configure_logging(settings.log_level)
    initialize_database(settings.semantic_index_db_path)
    repository = SemanticRepository(settings.semantic_index_db_path)
    embedder = _get_embedder(settings)

    target = repository.get_build(build_id) if build_id is not None else repository.latest_build()
    if target is None:
        return {"status": "error", "errors": ["No build found to resume"]}

    target_build_id = int(target["build_id"])
    pending_before = repository.count_pending_windows(target_build_id)
    if pending_before == 0:
        repository.complete_build(target_build_id, status="completed")
        return {
            "status": "completed",
            "build_id": target_build_id,
            "window_count": repository.count_windows_for_build(target_build_id),
            "pending_windows": 0,
            "resumed": True,
        }

    try:
        processed = _embed_pending_windows(
            settings=settings,
            repository=repository,
            embedder=embedder,
            build_id=target_build_id,
        )
        final_pending = repository.count_pending_windows(target_build_id)
        repository.complete_build(
            target_build_id,
            status="completed" if final_pending == 0 else "failed",
            notes=f"resumed processed_windows={processed}",
        )
        return {
            "status": "completed" if final_pending == 0 else "failed",
            "build_id": target_build_id,
            "window_count": repository.count_windows_for_build(target_build_id),
            "processed_windows": processed,
            "pending_windows": final_pending,
            "resumed": True,
        }
    except Exception as exc:
        repository.record_error(
            build_id=target_build_id,
            stage="resume_build",
            error_message=str(exc),
        )
        repository.complete_build(target_build_id, status="failed", notes=str(exc))
        raise


def _run_build(settings: Settings, *, cgid: str | None, incremental: bool) -> dict[str, object]:
    settings.validate()
    _configure_logging(settings.log_level)
    initialize_database(settings.semantic_index_db_path)
    repository = SemanticRepository(settings.semantic_index_db_path)
    embedder = _get_embedder(settings)
    source_reader = SourceReader(settings.corpus_db_path, settings.unit_text_db_path)

    build_id = repository.create_build(
        cgid_scope=cgid,
        window_size_units=settings.window_size_units,
        window_stride_units=settings.window_stride_units,
        embedding_model=settings.openai_embedding_model,
        embedding_dimensions=settings.openai_embedding_dimensions,
        normalization_version=settings.normalization_version,
        status="running",
        notes="incremental" if incremental else "full",
    )

    try:
        units_by_record = source_reader.fetch_units(cgid=cgid)
        if incremental:
            dirty_record_ids = _determine_dirty_record_ids(repository, units_by_record, cgid)
            repository.delete_records_for_scope(dirty_record_ids)
            units_by_record = {k: v for k, v in units_by_record.items() if k in dirty_record_ids}

        warnings, window_count = _persist_windows(
            settings=settings,
            repository=repository,
            build_id=build_id,
            units_by_record=units_by_record,
        )

        processed = _embed_pending_windows(
            settings=settings,
            repository=repository,
            embedder=embedder,
            build_id=build_id,
        )
        pending = repository.count_pending_windows(build_id)
        repository.complete_build(
            build_id,
            status="completed" if pending == 0 else "failed",
            notes=f"processed_windows={processed}",
        )
        return {
            "status": "completed" if pending == 0 else "failed",
            "build_id": build_id,
            "record_count": len(units_by_record),
            "window_count": window_count,
            "processed_windows": processed,
            "pending_windows": pending,
            "warnings": warnings,
            "incremental": incremental,
        }
    except Exception as exc:
        repository.record_error(build_id=build_id, stage="build", error_message=str(exc))
        repository.complete_build(build_id, status="failed", notes=str(exc))
        raise


def _persist_windows(
    *,
    settings: Settings,
    repository: SemanticRepository,
    build_id: int,
    units_by_record,
) -> tuple[list[str], int]:
    warnings: list[str] = []
    total_windows = 0

    for record_units in units_by_record.values():
        windows, record_warnings = build_windows(
            build_id=build_id,
            units=record_units,
            embedding_model=settings.openai_embedding_model,
            window_size_units=settings.window_size_units,
            window_stride_units=settings.window_stride_units,
        )
        if windows:
            repository.insert_windows(windows)
            total_windows += len(windows)
        warnings.extend(record_warnings)
        for warning in record_warnings:
            repository.record_error(
                build_id=build_id,
                record_id=record_units[0].record_id,
                stage="window_builder",
                error_message=warning,
            )

    return warnings, total_windows


def _embed_pending_windows(
    *,
    settings: Settings,
    repository: SemanticRepository,
    embedder,
    build_id: int,
) -> int:
    processed = 0
    while True:
        rows = repository.fetch_pending_windows(
            build_id=build_id,
            limit=max(settings.embed_batch_size * 2, settings.embed_batch_size),
        )
        if not rows:
            break

        batch_rows = _select_embedding_batch(
            rows=rows,
            batch_size=settings.embed_batch_size,
            token_budget=settings.embed_token_budget,
        )
        texts = [row["window_text"] for row in batch_rows]
        LOGGER.info(
            "Embedding batch for build_id=%s windows=%s approx_tokens=%s",
            build_id,
            len(batch_rows),
            sum(int(row["token_count_est"]) for row in batch_rows),
        )
        embeddings = embedder.embed_texts(texts)
        vector_items = []
        for row, embedding in zip(batch_rows, embeddings, strict=True):
            dims, blob = vector_to_blob(embedding)
            vector_items.append((int(row["window_id"]), dims, blob))
        repository.insert_vectors(
            build_id=build_id,
            embedding_model=settings.openai_embedding_model,
            items=vector_items,
        )
        processed += len(batch_rows)

    return processed


def _select_embedding_batch(
    *, rows: list[Any], batch_size: int, token_budget: int
) -> list[Any]:
    selected: list[Any] = []
    tokens = 0
    for row in rows:
        row_tokens = int(row["token_count_est"])
        if selected and (len(selected) >= batch_size or tokens + row_tokens > token_budget):
            break
        selected.append(row)
        tokens += row_tokens
    return selected or rows[:1]


def _determine_dirty_record_ids(
    repository: SemanticRepository, units_by_record, cgid_scope: str | None
) -> list[int]:
    if not units_by_record:
        return []
    baseline_build_id = repository.latest_completed_build_id(cgid_scope)
    existing = (
        repository.record_state_for_build(baseline_build_id)
        if baseline_build_id is not None
        else {}
    )
    dirty: list[int] = []
    for record_id, units in units_by_record.items():
        source_content_hash = units[0].content_hash
        source_max_updated_at = max((u.text_updated_at or u.unit_updated_at) for u in units)
        source_unit_signature = ",".join(str(u.unit_no) for u in units)
        current = (
            source_content_hash,
            source_max_updated_at,
            source_unit_signature,
        )
        if existing.get(record_id) != current:
            dirty.append(record_id)
    return dirty


def validate_build(settings: Settings, *, build_id: int | None = None) -> dict[str, object]:
    settings.validate()
    _configure_logging(settings.log_level)
    initialize_database(settings.semantic_index_db_path)
    repository = SemanticRepository(settings.semantic_index_db_path)
    target_build_id = build_id or repository.latest_build_id()
    if target_build_id is None:
        return {"valid": False, "errors": ["No builds found"]}
    return validate_build_output(repository, target_build_id)
