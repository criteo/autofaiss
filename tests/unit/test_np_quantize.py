import numpy as np
from autofaiss import build_index


def test_np_quantize():
    """test multi_array_split fct number 1"""
    embs = np.ones((100, 512), "float32")
    index, _ = build_index(embs, save_on_disk=False)
    _, I = index.search(embs, 1)
    assert I[0][0] == 0
