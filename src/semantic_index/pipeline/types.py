from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceUnit:
    cgid: str | None
    record_id: int
    doc_id: str
    unit_no: int
    text: str | None
    content_hash: str | None
    unit_updated_at: str
    text_updated_at: str | None


@dataclass(frozen=True)
class WindowRecord:
    build_id: int
    cgid: str | None
    record_id: int
    doc_id: str
    start_unit_no: int
    end_unit_no: int
    window_text: str
    char_len: int
    token_count_est: int
    text_hash: str
    source_content_hash: str | None
    source_max_updated_at: str
    source_unit_signature: str
    created_at: str
