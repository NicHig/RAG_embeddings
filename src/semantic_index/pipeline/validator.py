from __future__ import annotations

from semantic_index.db.repository import SemanticRepository


def validate_build_output(repository: SemanticRepository, build_id: int) -> dict[str, object]:
    windows = repository.fetch_windows_for_build(build_id)
    if not windows:
        return {"build_id": build_id, "valid": False, "errors": ["No windows for build"]}

    with repository.connect() as conn:
        vector_count = conn.execute(
            "SELECT COUNT(*) AS count FROM semantic_vectors WHERE build_id = ?", (build_id,)
        ).fetchone()["count"]
        missing_vectors = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM semantic_windows w
            LEFT JOIN semantic_vectors v ON v.window_id = w.window_id
            WHERE w.build_id = ? AND v.window_id IS NULL
            """,
            (build_id,),
        ).fetchone()["count"]
        null_cgid = conn.execute(
            "SELECT COUNT(*) AS count FROM semantic_windows WHERE build_id = ? AND cgid IS NULL",
            (build_id,),
        ).fetchone()["count"]
        dim_rows = conn.execute(
            """
            SELECT embedding_dimensions, COUNT(*) AS count
            FROM semantic_vectors
            WHERE build_id = ?
            GROUP BY embedding_dimensions
            """,
            (build_id,),
        ).fetchall()

    errors: list[str] = []
    if missing_vectors:
        errors.append(f"Missing vectors for {missing_vectors} window(s)")
    if null_cgid:
        errors.append(f"NULL cgid on {null_cgid} window(s)")
    if len(dim_rows) > 1:
        errors.append("Multiple embedding dimensions found in build output")

    return {
        "build_id": build_id,
        "valid": not errors,
        "window_count": len(windows),
        "vector_count": int(vector_count),
        "embedding_dimensions": None if not dim_rows else dim_rows[0]["embedding_dimensions"],
        "errors": errors,
    }
