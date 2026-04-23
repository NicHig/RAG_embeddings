from __future__ import annotations

import json
from pathlib import Path


def write_batch_jsonl(texts: list[tuple[str, str]], output_path: Path, model: str) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for custom_id, text in texts:
            handle.write(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/embeddings",
                        "body": {"model": model, "input": text},
                    }
                )
            )
            handle.write("\n")
