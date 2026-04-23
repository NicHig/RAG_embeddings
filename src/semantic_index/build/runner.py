from __future__ import annotations

import logging
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

from openai import RateLimitError

from semantic_index.config.settings import Settings
from semantic_index.db.init_db import initialize_database
from semantic_index.db.repository import SemanticRepository
from semantic_index.pipeline.embedder import DeterministicEmbedder, OpenAIEmbedder
from semantic_index.pipeline.source_reader import SourceReader
from semantic_index.pipeline.tokenizer import estimate_tokens, split_text_by_tokens
from semantic_index.pipeline.types import WindowRecord
from semantic_index.pipeline.validator import validate_build_output
from semantic_index.pipeline.vector_store import vector_to_blob
from semantic_index.pipeline.window_builder import build_windows

LOGGER = logging.getLogger(__name__)
RATE_LIMIT_SECONDS_RE = re.compile(r"try again in ([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)
RATE_LIMIT_LIMIT_RE = re.compile(r"Limit\s+([0-9]+)", re.IGNORECASE)


class AdaptiveRateController:
    def __init__(self, settings: Settings):
        self.config_batch_size = settings.embed_batch_size
        self.config_token_budget = settings.embed_token_budget
        self.current_batch_size = settings.embed_batch_size
        self.current_token_budget = settings.embed_token_budget
        self.min_batch_size = settings.embed_min_batch_size
        self.min_token_budget = settings.embed_min_token_budget
        self.tpm_limit = settings.openai_tpm_limit
        self.tpm_target_ratio = settings.openai_tpm_target_ratio
        self.base_sleep = settings.rate_limit_base_sleep_seconds
        self.max_sleep = settings.rate_limit_max_sleep_seconds
        self.sent_batches: deque[tuple[float, int]] = deque()

    def effective_batch_size(self) -> int:
        return max(self.min_batch_size, self.current_batch_size)

    def effective_token_budget(self) -> int:
        return max(self.min_token_budget, self.current_token_budget)

    def wait_for_capacity(self, next_batch_tokens: int) -> None:
        if self.tpm_limit is None:
            return
        while True:
            self._prune_sent()
            used_tokens = sum(tokens for _, tokens in self.sent_batches)
            target_limit = int(self.tpm_limit * self.tpm_target_ratio)
            if used_tokens + next_batch_tokens <= target_limit:
                return
            now = time.time()
            oldest_timestamp, _ = self.sent_batches[0]
            sleep_for = max(1.0, 60.0 - (now - oldest_timestamp) + 0.25)
            LOGGER.info(
                "Pausing for TPM capacity used_tokens=%s next_batch_tokens=%s target_limit=%s sleep=%.2fs",
                used_tokens,
                next_batch_tokens,
                target_limit,
                sleep_for,
            )
            time.sleep(min(sleep_for, self.max_sleep))

    def record_success(self, batch_tokens: int) -> None:
        self.sent_batches.append((time.time(), batch_tokens))
        self._prune_sent()
        self.current_batch_size = min(
            self.config_batch_size, max(self.min_batch_size, self.current_batch_size + 5)
        )
        self.current_token_budget = min(
            self.config_token_budget,
            max(self.min_token_budget, int(self.current_token_budget * 1.1)),
        )

    def record_rate_limit(self, error: RateLimitError) -> float:
        message = _rate_limit_message(error)
        parsed_limit = _parse_rate_limit_tpm_limit(message)
        if parsed_limit is not None:
            self.tpm_limit = parsed_limit
        self.current_batch_size = max(self.min_batch_size, max(1, self.current_batch_size // 2))
        self.current_token_budget = max(
            self.min_token_budget, max(1, self.current_token_budget // 2)
        )
        retry_after = _parse_retry_after_seconds(message) or self.base_sleep
        sleep_for = min(max(retry_after + 0.5, self.base_sleep), self.max_sleep)
        LOGGER.warning(
            "Rate limit encountered tpm_limit=%s next_batch_size=%s next_token_budget=%s sleep=%.2fs",
            self.tpm_limit,
            self.current_batch_size,
            self.current_token_budget,
            sleep_for,
        )
        return sleep_for

    def _prune_sent(self) -> None:
        cutoff = time.time() - 60.0
        while self.sent_batches and self.sent_batches[0][0] < cutoff:
            self.sent_batches.popleft()


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
            max_embedding_input_tokens=settings.max_embedding_input_tokens,
            oversized_window_overlap_tokens=settings.oversized_window_overlap_tokens,
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
    controller = AdaptiveRateController(settings)
    while True:
        rows = repository.fetch_pending_windows(
            build_id=build_id,
            limit=max(controller.effective_batch_size() * 2, controller.effective_batch_size()),
        )
        if not rows:
            break

        oversized_row = next(
            (
                row
                for row in rows
                if int(row["token_count_est"]) > settings.max_embedding_input_tokens
            ),
            None,
        )
        if oversized_row is not None and _split_oversized_pending_window(
            settings=settings,
            repository=repository,
            row=oversized_row,
        ):
            continue

        batch_rows = _select_embedding_batch(
            rows=rows,
            batch_size=controller.effective_batch_size(),
            token_budget=controller.effective_token_budget(),
        )
        batch_tokens = sum(int(row["token_count_est"]) for row in batch_rows)
        controller.wait_for_capacity(batch_tokens)
        texts = [row["window_text"] for row in batch_rows]
        LOGGER.info(
            "Embedding batch for build_id=%s windows=%s approx_tokens=%s",
            build_id,
            len(batch_rows),
            batch_tokens,
        )
        embeddings = None
        while True:
            try:
                embeddings = embedder.embed_texts(texts)
                break
            except RateLimitError as exc:
                sleep_for = controller.record_rate_limit(exc)
                smaller_batch_needed = (
                    len(batch_rows) > controller.effective_batch_size()
                    or batch_tokens > controller.effective_token_budget()
                )
                time.sleep(sleep_for)
                if smaller_batch_needed:
                    LOGGER.info(
                        "Reselecting smaller batch after rate limit build_id=%s prior_windows=%s prior_tokens=%s",
                        build_id,
                        len(batch_rows),
                        batch_tokens,
                    )
                    embeddings = None
                    break
                LOGGER.info(
                    "Retrying same embedding batch after rate limit build_id=%s windows=%s approx_tokens=%s",
                    build_id,
                    len(batch_rows),
                    batch_tokens,
                )
        if embeddings is None:
            continue
        vector_items = []
        for row, embedding in zip(batch_rows, embeddings, strict=True):
            dims, blob = vector_to_blob(embedding)
            vector_items.append((int(row["window_id"]), dims, blob))
        repository.insert_vectors(
            build_id=build_id,
            embedding_model=settings.openai_embedding_model,
            items=vector_items,
        )
        controller.record_success(batch_tokens)
        processed += len(batch_rows)

    return processed


def _split_oversized_pending_window(
    *, settings: Settings, repository: SemanticRepository, row
) -> bool:
    if int(row["token_count_est"]) <= settings.max_embedding_input_tokens:
        return False

    segments = split_text_by_tokens(
        text=row["window_text"],
        model=settings.openai_embedding_model,
        max_tokens=settings.max_embedding_input_tokens,
        overlap_tokens=settings.oversized_window_overlap_tokens,
    )
    replacement_rows = [
        WindowRecord(
            build_id=int(row["build_id"]),
            cgid=row["cgid"],
            record_id=int(row["record_id"]),
            doc_id=row["doc_id"],
            start_unit_no=int(row["start_unit_no"]),
            end_unit_no=int(row["end_unit_no"]),
            segment_index=segment_index,
            segment_count=len(segments),
            window_text=segment_text,
            char_len=len(segment_text),
            token_count_est=estimate_tokens(segment_text, settings.openai_embedding_model),
            text_hash=_hash_text(segment_text),
            source_content_hash=row["source_content_hash"],
            source_max_updated_at=row["source_max_updated_at"],
            source_unit_signature=row["source_unit_signature"],
            created_at=row["created_at"],
        )
        for segment_index, segment_text in enumerate(segments)
    ]
    repository.replace_window_with_segments(
        window_id=int(row["window_id"]),
        segments=replacement_rows,
    )
    LOGGER.warning(
        "Split oversized window build_id=%s window_id=%s original_tokens=%s new_segments=%s",
        row["build_id"],
        row["window_id"],
        row["token_count_est"],
        len(replacement_rows),
    )
    return True


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


def _rate_limit_message(error: RateLimitError) -> str:
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        nested = body.get("error")
        if isinstance(nested, dict) and nested.get("message"):
            return str(nested["message"])
    return str(error)


def _parse_retry_after_seconds(message: str) -> float | None:
    match = RATE_LIMIT_SECONDS_RE.search(message)
    return None if match is None else float(match.group(1))


def _parse_rate_limit_tpm_limit(message: str) -> int | None:
    match = RATE_LIMIT_LIMIT_RE.search(message)
    return None if match is None else int(match.group(1))


def _hash_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
