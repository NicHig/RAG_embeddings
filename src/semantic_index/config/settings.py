from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


@dataclass(frozen=True)
class Settings:
    corpus_db_path: Path
    unit_text_db_path: Path
    semantic_index_db_path: Path
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int | None = None
    normalization_version: str = "v1"
    window_size_units: int = 3
    window_stride_units: int = 1
    embed_batch_size: int = 100
    embed_token_budget: int = 250000
    embedding_mode: str = "sync"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv(Path.cwd() / ".env")

        def read_path(name: str) -> Path:
            value = os.environ.get(name)
            if not value:
                raise ValueError(f"Missing required environment variable: {name}")
            return Path(value).expanduser()

        dimensions = os.environ.get("OPENAI_EMBEDDING_DIMENSIONS")
        return cls(
            corpus_db_path=read_path("CORPUS_DB_PATH"),
            unit_text_db_path=read_path("UNIT_TEXT_DB_PATH"),
            semantic_index_db_path=read_path("SEMANTIC_INDEX_DB_PATH"),
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            openai_embedding_model=os.environ.get(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            openai_embedding_dimensions=int(dimensions) if dimensions else None,
            normalization_version=os.environ.get("NORMALIZATION_VERSION", "v1"),
            window_size_units=int(os.environ.get("WINDOW_SIZE_UNITS", "3")),
            window_stride_units=int(os.environ.get("WINDOW_STRIDE_UNITS", "1")),
            embed_batch_size=int(os.environ.get("EMBED_BATCH_SIZE", "100")),
            embed_token_budget=int(os.environ.get("EMBED_TOKEN_BUDGET", "250000")),
            embedding_mode=os.environ.get("EMBEDDING_MODE", "sync"),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )

    def validate(self) -> None:
        missing = [
            path
            for path in (self.corpus_db_path, self.unit_text_db_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing source database(s): " + ", ".join(str(path) for path in missing)
            )
        self.validate_output_path()

    def validate_output_path(self) -> None:
        self.semantic_index_db_path.parent.mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
