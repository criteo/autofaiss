"""
An example of running autofaiss by pyspark to produce N indices.

You need to install pyspark before using the following example.
"""
from typing import Dict

import faiss
import numpy as np

from autofaiss import build_index

# You'd better create a spark session before calling build_index,
# otherwise, a spark session would be created by autofaiss with the least configuration.

index_path2_metric_infos: Dict[str, Dict] = build_index(
    embeddings="hdfs://root/path/to/your/embeddings/folder",
    distributed="pyspark",
    file_format="parquet",
    temporary_indices_folder="hdfs://root/tmp/distributed_autofaiss_indices",
    current_memory_available="10G",
    max_index_memory_usage="100G",
    nb_indices_to_keep=10,
)

index_paths = list(index_path2_metric_infos.keys())

###########################################
# Use case 1: merging 10 indices into one #
###########################################
merged = faiss.read_index(index_paths[0])
for rest_index_file in index_paths[1:]:
    index = faiss.read_index(rest_index_file)
    faiss.merge_into(merged, index, shift_ids=True)
with open("merged-knn.index", "wb") as f:
    faiss.write_index(merged, faiss.PyCallbackIOWriter(f.write))

########################################
# Use case 2: searching from N indices #
########################################
K, DIM, all_distances, all_ids = 5, 512, [], []
queries = faiss.rand((2, DIM))
for rest_index_file in index_paths:
    index = faiss.read_index(rest_index_file)
    distances, ids = index.search(queries, k=K)
    all_distances.append(distances)
    all_ids.append(ids)

dists_arr = np.stack(all_distances, axis=1).reshape(2, -1)
knn_ids_arr = np.stack(all_ids, axis=1).reshape(2, -1)

sorted_k_indices = np.argsort(-dists_arr)[:, :K]
sorted_k_dists = np.take_along_axis(dists_arr, sorted_k_indices, axis=1)
sorted_k_ids = np.take_along_axis(knn_ids_arr, sorted_k_indices, axis=1)
print(f"{K} nearest distances: {sorted_k_dists}")
print(f"{K} nearest ids: {sorted_k_ids}")
