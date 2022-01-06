import numpy as np
from autofaiss import build_index, tune_index, score_index

embs = np.float32(np.random.rand(100, 512))
index, index_infos = build_index(embs, save_on_disk=False)
index = tune_index(index, index_infos["index_key"], save_on_disk=False)
infos = score_index(index, embs, save_on_disk=False)
