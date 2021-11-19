import numpy as np
from autofaiss import build_index, tune_index, score_index


def test_scoring_tuning():
    embs = np.ones((100, 512), "float32")
    index, index_infos = build_index(embs, save_on_disk=False)
    index = tune_index(index, index_infos["index_key"], save_on_disk=False)
    infos = score_index(index, embs, save_on_disk=False)
