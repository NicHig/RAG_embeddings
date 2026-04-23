from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import UTC, datetime

from semantic_index.pipeline.normalizer import normalize_text
from semantic_index.pipeline.tokenizer import estimate_tokens, split_text_by_tokens
from semantic_index.pipeline.types import SourceUnit, WindowRecord


def _window_slices(length: int, size: int, stride: int) -> list[tuple[int, int]]:
    if length <= size:
        return [(0, length)]
    return [(start, start + size) for start in range(0, length - size + 1, stride)]


def _max_timestamp(units: Iterable[SourceUnit]) -> str:
    timestamps = [u.text_updated_at or u.unit_updated_at for u in units]
    return max(timestamps)


def build_windows(
    *,
    build_id: int,
    units: list[SourceUnit],
    embedding_model: str,
    window_size_units: int,
    window_stride_units: int,
    max_embedding_input_tokens: int,
    oversized_window_overlap_tokens: int,
) -> tuple[list[WindowRecord], list[str]]:
    windows: list[WindowRecord] = []
    warnings: list[str] = []
    ordered = sorted(units, key=lambda item: item.unit_no)

    if any(unit.text is None for unit in ordered):
        missing = [str(unit.unit_no) for unit in ordered if unit.text is None]
        warnings.append(
            f"record_id={ordered[0].record_id} missing extracted text for unit_no={','.join(missing)}"
        )

    populated = [unit for unit in ordered if unit.text]
    if not populated:
        return windows, warnings

    unit_signature = ",".join(str(unit.unit_no) for unit in ordered)
    created_at = datetime.now(UTC).isoformat()

    for start_idx, end_idx in _window_slices(
        len(populated), window_size_units, window_stride_units
    ):
        slice_units = populated[start_idx:end_idx]
        body = "\n\n".join(
            f"[Unit {unit.unit_no}]\n{unit.text.strip()}" for unit in slice_units
        )
        normalized = normalize_text(body)
        segments = split_text_by_tokens(
            text=normalized,
            model=embedding_model,
            max_tokens=max_embedding_input_tokens,
            overlap_tokens=oversized_window_overlap_tokens,
        )
        for segment_index, segment_text in enumerate(segments):
            windows.append(
                WindowRecord(
                    build_id=build_id,
                    cgid=slice_units[0].cgid,
                    record_id=slice_units[0].record_id,
                    doc_id=slice_units[0].doc_id,
                    start_unit_no=slice_units[0].unit_no,
                    end_unit_no=slice_units[-1].unit_no,
                    segment_index=segment_index,
                    segment_count=len(segments),
                    window_text=segment_text,
                    char_len=len(segment_text),
                    token_count_est=estimate_tokens(segment_text, embedding_model),
                    text_hash=hashlib.sha256(segment_text.encode("utf-8")).hexdigest(),
                    source_content_hash=slice_units[0].content_hash,
                    source_max_updated_at=_max_timestamp(slice_units),
                    source_unit_signature=unit_signature,
                    created_at=created_at,
                )
            )
    return windows, warnings
