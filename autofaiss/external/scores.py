""" Functions to compute metrics on an index """
import os
from typing import Dict, Union, Optional

import numpy as np
import faiss

from autofaiss.datasets.readers.local_iterators import read_embeddings_local
from autofaiss.datasets.readers.remote_iterators import read_embeddings_remote
from autofaiss.indices.index_utils import get_index_size, search_speed_test
from autofaiss.indices.memory_efficient_flat_index import MemEfficientFlatIndex
from autofaiss.metrics.recalls import one_recall_at_r, r_recall_at_r
from autofaiss.metrics.reconstruction import quantize_vec_without_modifying_index, reconstruction_error
from autofaiss.utils.cast import cast_memory_to_bytes
from autofaiss.utils.decorators import Timeit


def compute_fast_metrics(
    embeddings_path: Union[np.ndarray, str],
    index: faiss.Index,
    omp_threads: Optional[int] = None,
    query_max: Optional[int] = 1000,
) -> Dict[str, Union[str, int, float]]:
    """compute query speed, size and reconstruction of an index"""
    infos: Dict[str, Union[str, int, float]] = {}

    size_bytes = get_index_size(index)
    infos["size in bytes"] = size_bytes

    if isinstance(embeddings_path, str):
        # pylint: disable=bare-except
        try:
            query_embeddings = next(read_embeddings_local(embeddings_path, verbose=False))
        except:
            query_embeddings = next(read_embeddings_remote(embeddings_path, verbose=False))
    else:
        query_embeddings = embeddings_path

    query_embeddings = query_embeddings[:query_max]
    if omp_threads:
        faiss.omp_set_num_threads(1)
    speeds_ms = search_speed_test(index, query_embeddings, ksearch=40, timout_s=10.0)
    if omp_threads:
        faiss.omp_set_num_threads(omp_threads)

    infos.update(speeds_ms)

    # quantize query embeddings if the index uses quantization
    quantized_embeddings = quantize_vec_without_modifying_index(index, query_embeddings)
    rec_error = reconstruction_error(query_embeddings, quantized_embeddings)
    infos["reconstruction error %"] = 100 * rec_error

    infos["nb vectors"] = index.ntotal

    infos["vectors dimension"] = index.d

    infos["compression ratio"] = 4.0 * index.ntotal * index.d / size_bytes

    return infos


def compute_medium_metrics(
    embeddings_path: Union[np.ndarray, str],
    index: faiss.Index,
    memory_available: Union[str, float],
    ground_truth: Optional[np.ndarray] = None,
    eval_item_ids: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute recall@R and intersection recall@R of an index"""

    nb_test_points = 500

    if isinstance(embeddings_path, str):
        # pylint: disable=bare-except
        try:
            embedding_block = next(read_embeddings_local(embeddings_path, verbose=False))
        except:
            embedding_block = next(read_embeddings_remote(embeddings_path, verbose=False))

        if embedding_block.shape[0] < nb_test_points:
            stacks = int(nb_test_points // embedding_block.shape[0]) + 1
            # pylint: disable=bare-except
            try:
                embedding_block = next(read_embeddings_local(embeddings_path, batch_size=nb_test_points, verbose=False))
            except:
                embedding_block = next(read_embeddings_remote(embeddings_path, stack_input=stacks, verbose=False))

        query_embeddings = embedding_block[:nb_test_points]
    else:
        embedding_block = embeddings_path

        query_embeddings = embedding_block[:nb_test_points]

    if ground_truth is None:
        if isinstance(embeddings_path, str):
            ground_truth_path = f"{embeddings_path}/small_ground_truth_test.npy"
            if not os.path.exists(ground_truth_path):

                with Timeit("-> Compute small ground truth", indent=1):

                    ground_truth = get_ground_truth(
                        index.metric_type, embeddings_path, query_embeddings, memory_available
                    )

                    with open(ground_truth_path, "wb") as gt_file:
                        np.save(gt_file, ground_truth)

            else:
                with Timeit("-> Load small ground truth", indent=1):
                    with open(ground_truth_path, "rb") as gt_file:
                        ground_truth = np.load(gt_file)
        else:
            ground_truth = get_ground_truth(index.metric_type, embedding_block, query_embeddings, memory_available)

    with Timeit("-> Compute recalls", indent=1):
        one_recall = one_recall_at_r(query_embeddings, ground_truth, index, 40, eval_item_ids)
        intersection_recall = r_recall_at_r(query_embeddings, ground_truth, index, 40, eval_item_ids)

    infos: Dict[str, float] = {}

    infos["1-recall@20"] = one_recall[20 - 1]
    infos["1-recall@40"] = one_recall[40 - 1]
    infos["20-recall@20"] = intersection_recall[20 - 1]
    infos["40-recall@40"] = intersection_recall[40 - 1]

    return infos


def get_ground_truth(
    faiss_metric_type: int,
    embeddings_path: Union[np.ndarray, str],
    query_embeddings: np.ndarray,
    memory_available: Union[str, float],
):
    """compute the ground truth (result with a perfect index) of the query on the embeddings"""

    dim = query_embeddings.shape[-1]

    if isinstance(embeddings_path, str):
        perfect_index = MemEfficientFlatIndex(dim, faiss_metric_type)
        perfect_index.add_files(embeddings_path)
    else:
        perfect_index = faiss.IndexFlat(dim, faiss_metric_type)
        perfect_index.add(embeddings_path.astype("float32"))  # pylint: disable= no-value-for-parameter

    memory_available = cast_memory_to_bytes(memory_available) if isinstance(memory_available, str) else memory_available

    batch_size = int(memory_available / (dim * 4) / 4)  # using 1/4 of the given memory

    if isinstance(embeddings_path, str):
        _, ground_truth = perfect_index.search_files(query_embeddings, k=40, batch_size=batch_size)
    else:
        _, ground_truth = perfect_index.search(query_embeddings, k=40)

    return ground_truth
