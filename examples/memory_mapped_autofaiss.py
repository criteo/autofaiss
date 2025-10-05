import faiss
import numpy as np
from autofaiss import build_index

embeddings = np.float32(np.random.rand(5000, 100))

# Example on how to build a memory-mapped index and load it from disk
_, index_infos = build_index(
    embeddings,
    save_on_disk=True,
    should_be_memory_mappable=True,
    index_path="my_index_folder/knn.index",
    max_index_memory_usage="4G",
    max_index_query_time_ms=50,
)
index = faiss.read_index("my_index_folder/knn.index", faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
