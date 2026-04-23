CREATE TABLE IF NOT EXISTS semantic_builds (
    build_id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    cgid_scope TEXT,
    window_size_units INTEGER NOT NULL,
    window_stride_units INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dimensions INTEGER,
    normalization_version TEXT NOT NULL,
    status TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS semantic_windows (
    window_id INTEGER PRIMARY KEY,
    build_id INTEGER NOT NULL,
    cgid TEXT,
    record_id INTEGER NOT NULL,
    doc_id TEXT NOT NULL,
    start_unit_no INTEGER NOT NULL,
    end_unit_no INTEGER NOT NULL,
    segment_index INTEGER NOT NULL DEFAULT 0,
    segment_count INTEGER NOT NULL DEFAULT 1,
    window_text TEXT NOT NULL,
    char_len INTEGER NOT NULL,
    token_count_est INTEGER NOT NULL,
    text_hash TEXT NOT NULL,
    source_content_hash TEXT,
    source_max_updated_at TEXT,
    source_unit_signature TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(build_id) REFERENCES semantic_builds(build_id)
);

CREATE INDEX IF NOT EXISTS idx_windows_build_id ON semantic_windows(build_id);
CREATE INDEX IF NOT EXISTS idx_windows_cgid ON semantic_windows(cgid);
CREATE INDEX IF NOT EXISTS idx_windows_record_id ON semantic_windows(record_id);

CREATE TABLE IF NOT EXISTS semantic_vectors (
    window_id INTEGER PRIMARY KEY,
    build_id INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dimensions INTEGER NOT NULL,
    vector_blob BLOB NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(window_id) REFERENCES semantic_windows(window_id),
    FOREIGN KEY(build_id) REFERENCES semantic_builds(build_id)
);

CREATE INDEX IF NOT EXISTS idx_vectors_build_id ON semantic_vectors(build_id);

CREATE TABLE IF NOT EXISTS semantic_errors (
    error_id INTEGER PRIMARY KEY,
    build_id INTEGER NOT NULL,
    record_id INTEGER,
    window_id INTEGER,
    stage TEXT NOT NULL,
    error_message TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(build_id) REFERENCES semantic_builds(build_id)
);

CREATE INDEX IF NOT EXISTS idx_errors_build_id ON semantic_errors(build_id);
