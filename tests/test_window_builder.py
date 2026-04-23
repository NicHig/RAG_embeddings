from semantic_index.pipeline.tokenizer import estimate_tokens
from semantic_index.pipeline.types import SourceUnit
from semantic_index.pipeline.window_builder import build_windows


def test_build_windows_for_three_units():
    units = [
        SourceUnit("C1", 10, "doc-10", 1, "first", "hash", "2026-01-01T00:00:00+00:00", None),
        SourceUnit("C1", 10, "doc-10", 2, "second", "hash", "2026-01-01T00:00:00+00:00", None),
        SourceUnit("C1", 10, "doc-10", 3, "third", "hash", "2026-01-01T00:00:00+00:00", None),
    ]
    windows, warnings = build_windows(
        build_id=1,
        units=units,
        embedding_model="text-embedding-3-small",
        window_size_units=3,
        window_stride_units=1,
    )
    assert warnings == []
    assert len(windows) == 1
    assert windows[0].start_unit_no == 1
    assert windows[0].end_unit_no == 3
    assert windows[0].token_count_est == estimate_tokens(
        windows[0].window_text, "text-embedding-3-small"
    )


def test_build_windows_skips_missing_text_and_records_warning():
    units = [
        SourceUnit("C1", 11, "doc-11", 1, "first", "hash", "2026-01-01T00:00:00+00:00", None),
        SourceUnit("C1", 11, "doc-11", 2, None, "hash", "2026-01-01T00:00:00+00:00", None),
    ]
    windows, warnings = build_windows(
        build_id=1,
        units=units,
        embedding_model="text-embedding-3-small",
        window_size_units=3,
        window_stride_units=1,
    )
    assert len(windows) == 1
    assert warnings
