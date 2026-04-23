from __future__ import annotations

import numpy as np


def vector_to_blob(vector: list[float]) -> tuple[int, bytes]:
    arr = np.array(vector, dtype=np.float32)
    return int(arr.shape[0]), arr.tobytes()


def blob_to_vector(blob: bytes, dimensions: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dimensions)
