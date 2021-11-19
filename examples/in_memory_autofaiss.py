from autofaiss import build_index
import numpy as np

embeddings = np.ones((100, 512), "float32")
index, index_infos = build_index(embeddings, save_on_disk=False)
_, I = index.search(embeddings, 1)
print(I)
