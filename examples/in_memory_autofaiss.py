from autofaiss import quantize
import numpy as np

embeddings = np.ones((100, 512), "float32")
index, _ = quantize(embeddings)
_, I = index.search(embeddings, 1)
print(I)
