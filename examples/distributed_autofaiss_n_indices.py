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

_, index_path2_metric_infos = build_index(
    embeddings="hdfs://root/path/to/your/embeddings/folder",
    distributed="pyspark",
    file_format="parquet",
    temporary_indices_folder="hdfs://root/tmp/distributed_autofaiss_indices",
    current_memory_available="10G",
    max_index_memory_usage="100G",
    nb_indices_to_keep=10,
)

index_paths = sorted(index_path2_metric_infos.keys())

###########################################
# Use case 1: merging 10 indices into one #
###########################################
merged = faiss.read_index(index_paths[0])
for rest_index_file in index_paths[1:]:
    index = faiss.read_index(rest_index_file)
    faiss.merge_into(merged, index, shift_ids=False)
with open("merged-knn.index", "wb") as f:
    faiss.write_index(merged, faiss.PyCallbackIOWriter(f.write))

########################################
# Use case 2: searching from N indices #
########################################
K, DIM, all_distances, all_ids, NB_QUERIES = 5, 512, [], [], 2
queries = faiss.rand((NB_QUERIES, DIM))
for rest_index_file in index_paths:
    index = faiss.read_index(rest_index_file)
    distances, ids = index.search(queries, k=K)
    all_distances.append(distances)
    all_ids.append(ids)

dists_arr = np.stack(all_distances, axis=1).reshape(NB_QUERIES, -1)
knn_ids_arr = np.stack(all_ids, axis=1).reshape(NB_QUERIES, -1)

sorted_k_indices = np.argsort(-dists_arr)[:, :K]
sorted_k_dists = np.take_along_axis(dists_arr, sorted_k_indices, axis=1)
sorted_k_ids = np.take_along_axis(knn_ids_arr, sorted_k_indices, axis=1)
print(f"{K} nearest distances: {sorted_k_dists}")
print(f"{K} nearest ids: {sorted_k_ids}")


############################################
# Use case 3: on disk merging of N indices #
############################################


# using faiss.merge_ondisk (https://github.com/facebookresearch/faiss/blob/30abcd6a865afef7cf86df7e8b839a41b5161505/contrib/ondisk.py )
# https://github.com/facebookresearch/faiss/blob/151e3d7be54aec844b6328dc3e7dd0b83fcfa5bc/demos/demo_ondisk_ivf.py
# to merge indices on disk without using memory
# this is useful in particular to use a very large index with almost no memory usage.

from faiss.contrib.ondisk import merge_ondisk
import faiss

block_fnames = index_paths
empty_index = faiss.read_index(block_fnames[0], faiss.IO_FLAG_MMAP)
empty_index.ntotal = 0

merge_ondisk(empty_index, block_fnames, "merged_index.ivfdata")

faiss.write_index(empty_index, "populated.index")

pop = faiss.read_index("populated.index", faiss.IO_FLAG_ONDISK_SAME_DIR)

########################################################
# Use case 4: use N indices using  HStackInvertedLists #
########################################################

# This allows using N indices as a single combined index
# without changing anything on disk or loading anything to memory
# it works well but it's slower than first using merge_ondisk
# because it requires explore N pieces of inverted list for each
# list to explore
import os


class CombinedIndex:
    """
    combines a set of inverted lists into a hstack
    adds these inverted lists to an empty index that contains
    the info on how to perform searches
    """

    def __init__(self, invlist_fnames):
        ilv = faiss.InvertedListsPtrVector()

        for fname in invlist_fnames:
            if os.path.exists(fname):
                index = faiss.read_index(fname, faiss.IO_FLAG_MMAP)
                index_ivf = faiss.extract_index_ivf(index)
                il = index_ivf.invlists
                index_ivf.own_invlists = False
            else:
                raise FileNotFoundError
            ilv.push_back(il)

        self.big_il = faiss.HStackInvertedLists(ilv.size(), ilv.data())
        ntotal = self.big_il.compute_ntotal()

        self.index = faiss.read_index(invlist_fnames[0], faiss.IO_FLAG_MMAP)

        index_ivf = faiss.extract_index_ivf(self.index)
        index_ivf.replace_invlists(self.big_il, True)
        index_ivf.ntotal = self.index.ntotal = ntotal

    def search(self, x, k):
        D, I = self.index.search(x, k)
        return D, I


index = CombinedIndex(index_paths)
index.search(queries, K)
