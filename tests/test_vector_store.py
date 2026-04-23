import numpy as np

from semantic_index.pipeline.vector_store import blob_to_vector, vector_to_blob


def test_vector_blob_round_trip():
    dims, blob = vector_to_blob([0.1, 0.2, 0.3])
    vec = blob_to_vector(blob, dims)
    assert dims == 3
    assert np.allclose(vec, np.array([0.1, 0.2, 0.3], dtype=np.float32))
