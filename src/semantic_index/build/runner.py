from __future__ import annotations

import logging
from pathlib import Path

from semantic_index.config.settings import Settings
from semantic_index.db.init_db import initialize_database
from semantic_index.db.repository import SemanticRepository
from semantic_index.pipeline.embedder import DeterministicEmbedder, OpenAIEmbedder
from semantic_index.pipeline.source_reader import SourceReader
from semantic_index.pipeline.validator import validate_build_output
from semantic_index.pipeline.vector_store import vector_to_blob
from semantic_index.pipeline.window_builder import build_windows


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(Path("logs") / "semantic_index.log"),
            logging.StreamHandler(),
        ],
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

    units_by_record = source_reader.fetch_units(cgid=cgid)
    if incremental:
        dirty_record_ids = _determine_dirty_record_ids(repository, units_by_record)
        repository.delete_records_for_scope(dirty_record_ids)
        units_by_record = {k: v for k, v in units_by_record.items() if k in dirty_record_ids}

    all_windows = []
    warnings: list[str] = []
    for record_units in units_by_record.values():
        windows, record_warnings = build_windows(
            build_id=build_id,
            units=record_units,
            embedding_model=settings.openai_embedding_model,
            window_size_units=settings.window_size_units,
            window_stride_units=settings.window_stride_units,
        )
        all_windows.extend(windows)
        warnings.extend(record_warnings)
        for warning in record_warnings:
            repository.record_error(
                build_id=build_id,
                record_id=record_units[0].record_id,
                stage="window_builder",
                error_message=warning,
            )

    if all_windows:
        window_ids = repository.insert_windows(all_windows)
        embeddings = embedder.embed_texts([window.window_text for window in all_windows])
        vector_items = []
        for window_id, embedding in zip(window_ids, embeddings, strict=True):
            dims, blob = vector_to_blob(embedding)
            vector_items.append((window_id, dims, blob))
        repository.insert_vectors(
            build_id=build_id,
            embedding_model=settings.openai_embedding_model,
            items=vector_items,
        )

    repository.complete_build(build_id, status="completed")
    return {
        "status": "completed",
        "build_id": build_id,
        "record_count": len(units_by_record),
        "window_count": len(all_windows),
        "warnings": warnings,
        "incremental": incremental,
    }


def _determine_dirty_record_ids(
    repository: SemanticRepository, units_by_record
) -> list[int]:
    if not units_by_record:
        return []
    with repository.connect() as conn:
        existing_rows = conn.execute(
            """
            SELECT record_id, source_content_hash, source_max_updated_at, source_unit_signature
            FROM semantic_windows
            GROUP BY record_id
            """
        ).fetchall()
    existing = {
        int(row["record_id"]): (
            row["source_content_hash"],
            row["source_max_updated_at"],
            row["source_unit_signature"],
        )
        for row in existing_rows
    }
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
