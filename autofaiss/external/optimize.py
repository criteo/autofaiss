""" Functions to find optimal index parameters """

import re
from functools import partial, reduce
from math import log2, sqrt, floor
from operator import mul
from typing import List, Optional

from autofaiss.indices.index_utils import set_search_hyperparameters, speed_test_ms_per_query
from autofaiss.utils.algorithms import discrete_binary_search
from autofaiss.utils.cast import cast_memory_to_bytes
from autofaiss.utils.decorators import Timeit


def index_key_to_nb_cluster(index_key: str):
    """
    Function that takes an index key and returns the number of clusters
    """

    matching = re.findall(r"IVF\d+|IMI\d+x\d+", index_key)

    if matching:
        # case IVF index
        if re.findall(r"IVF\d+", matching[0]):
            nb_clusters = int(matching[0][3:])
        # case IMI index
        elif re.findall(r"IMI\d+x\d+", matching[0]):
            nb_clusters = 2 ** reduce(mul, [int(num) for num in re.findall(r"\d+", matching[0])])
        else:
            raise ValueError("Unable to determine the number of clusters for index {}".format(index_key))
    else:
        raise ValueError("Unable to determine the number of clusters for index {}".format(index_key))

    return nb_clusters


def compute_memory_necessary_for_training(nb_training_vectors: int, index_key: str, vec_dim: int):
    """
    Function that computes the memory necessary to train an index with nb_training_vectors vectors
    """

    memory_per_vector = 4.0 * vec_dim
    matching_opq = re.search(r"OPQ\d+_(\d+)", index_key)
    if matching_opq:
        input_size_opq = int(matching_opq.group(1))
        memory_per_vector += input_size_opq * 4.0
    return memory_per_vector * nb_training_vectors


def get_optimal_train_size(
    nb_vectors: int, index_key: str, current_memory_available: Optional[str], vec_dim: Optional[int]
) -> int:
    """
    Function that determines the number of training points necessary to
    train the index, based on faiss heuristics for k-means clustering.
    """

    matching = re.findall(r"IVF\d+|IMI\d+x\d+", index_key)

    if matching:
        nb_clusters = index_key_to_nb_cluster(index_key)
        points_per_cluster: float = 100

        # compute best possible number of vectors to give to train the index
        # given memory constraints
        if current_memory_available and vec_dim:
            memory_per_cluster_set = compute_memory_necessary_for_training(nb_clusters, index_key, vec_dim)
            size = cast_memory_to_bytes(current_memory_available)
            points_per_cluster = max(min(size / memory_per_cluster_set, points_per_cluster), 31.0)

        # You will need between 30 * nb_clusters and 256 * nb_clusters to train the index
        train_size = min(round(points_per_cluster * nb_clusters), nb_vectors)

    else:
        raise ValueError(f"Unknown index type: {index_key}")

    return train_size


def get_optimal_batch_size(vec_dim: int, current_memory_available: str) -> int:
    """compute optimal batch size to use the RAM at its full potential for adding vectors"""

    memory = cast_memory_to_bytes(current_memory_available)

    batch_size = int(min(memory, 10 ** 9) / (vec_dim * 4))  # using more than 1GB of ram is not faster here

    return batch_size


def get_optimal_nb_clusters(nb_vectors: int) -> List[int]:
    """
    Returns a list with the recommended number of clusters for an index containing nb_vectors vectors.
    The first value is the most recommended one.
    see: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    """

    nb_clusters_list = []

    if nb_vectors < 1_000_000:
        # no need to use HNSW for small number of clusters
        x_initial = 4 * sqrt(nb_vectors)  # between 4xsqrt(n) and 16xsqrt(n)
        x_closest_power = 2 ** round(log2(x_initial))
        nb_clusters_list.append(round(x_closest_power))
        nb_clusters_list.append(2 * x_closest_power)
        nb_clusters_list.append(x_closest_power)
        nb_clusters_list.append(round(x_initial))
    elif nb_vectors < 10_000_000:
        nb_clusters_list.append(16_384)
        nb_clusters_list.append(65_536)
    elif nb_vectors < 300_000_000:
        nb_clusters_list.append(65_536)
        nb_clusters_list.append(2 ** 17)
        nb_clusters_list.append(2 ** 18)  # slow training !
    else:
        nb_clusters_list.append(2 ** 17)
        nb_clusters_list.append(2 ** 18)  # slow training !
        nb_clusters_list.append(65_536)
        nb_clusters_list.append(2 ** 20)  # very slow training !

    nb_clusters_list = [int(x) for x in nb_clusters_list if x >= 256]

    if not nb_clusters_list:
        return [256]  # minimum possible value for Faiss IVF

    return nb_clusters_list


def get_optimal_index_keys_v2(
    nb_vectors: int,
    dim_vector: int,
    max_index_memory_usage: str,
    flat_threshold: int = 1000,
    hnsw_threshold: int = 10000,
    force_pq: Optional[int] = None,
):
    """
    Gives a list of interesting indices to try, *the one at the top is the most promising*

    See: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index for
    detailed explanations.
    """

    # Exception cases:
    if nb_vectors < flat_threshold:  # HNSW Faiss slower when less than 1000 vectors
        return ["Flat"]
    if nb_vectors < hnsw_threshold:  # HNSW is faster and quantization is not possible (>=10_000 vectors needed)
        return ["HNSW15"]
    if force_pq is not None:
        return get_optimal_quantization(nb_vectors, dim_vector, force_quantization_value=force_pq)

    # Get max memory usage
    max_size_in_bytes = cast_memory_to_bytes(max_index_memory_usage)

    # If we can build an HNSW with the given memory constraints, it's the best
    m_hnsw = int(floor((max_size_in_bytes / (4 * nb_vectors) - dim_vector) / 2))
    if m_hnsw >= 8:
        return [f"HNSW{min(m_hnsw, 32)}"]

    # Otherwise, there is not enough space, let's go for quantization
    return get_optimal_quantization(nb_vectors, dim_vector, force_max_index_memory_usage=max_index_memory_usage)


def get_optimal_quantization(
    nb_vectors: int,
    dim_vector: int,
    force_quantization_value: Optional[int] = None,
    force_max_index_memory_usage: Optional[str] = None,
) -> List[str]:
    """
    Function that returns a list of relevant index_keys to create quantized indices.

    Parameters:
    ----------
    nb_vectors: int
        Number of vectors in the dataset.
    dim_vector: int
        Dimension of the vectors in the dataset.
    force_quantization_value: Optional[int]
        Force to use this value as the size of the quantized vectors (PQx).
        It can be used with the force_max_index_memory_usage parameter,
        but the result might be empty.
    force_max_index_memory_usage: Optional[str]
        Add a memory constraint on the index.
        It can be used with the force_quantization_value parameter,
        but the result might be empty.

    Return:
    -------
    index_keys: List[str]
        List of index_keys that would be good choices for quantization.
        The list can be empty if the given constraints are too strong.
    """

    # Default values
    pq_values = [256, 128, 64, 48, 32, 24, 16, 8, 4]
    targeted_compression_ratio = 0.0  # 0 = no constraint

    # Force compression ratio if required
    if force_max_index_memory_usage is not None:
        total_bytes = 4.0 * nb_vectors * dim_vector  # x4 because float32
        max_mem_bytes = float(cast_memory_to_bytes(force_max_index_memory_usage))
        targeted_compression_ratio = total_bytes / max_mem_bytes

    # Force quantization value if required
    if force_quantization_value is not None:
        pq_values = [force_quantization_value]

    # Compute optimal number of clusters
    relevant_list: List[str] = []
    nb_clusters_list = get_optimal_nb_clusters(nb_vectors)

    # Look for matching index keys
    for pq in pq_values:
        if pq < dim_vector:

            for nb_clusters in nb_clusters_list:

                # Compute quantized vector size
                cluster_size_byte = 1 + (log2(nb_clusters) - 1) // 8
                vector_size_byte = pq + cluster_size_byte

                # Compute compression ratio with quantization PQx
                compression_ratio = (4 * dim_vector) / vector_size_byte

                # Add index_key if compression ratio is high enough
                if compression_ratio >= targeted_compression_ratio:

                    # y is a multiple of pq (required)
                    # y <= d, with d the dimension of the input vectors (preferable)
                    # y <= 6*pq (preferable)
                    # here we choose a y slightly bigger than d to avoid losing information
                    # in case such as 101, 128 is better than 64 to avoid losing information
                    # in the linear transform
                    y = (min(dim_vector // pq, 6) + 1) * pq
                    cluster_opt = f"IVF{nb_clusters}" if nb_clusters < 1000 else f"IVF{nb_clusters}_HNSW32"
                    relevant_list.append(f"OPQ{pq}_{y},{cluster_opt},PQ{pq}x8")

    return relevant_list


def get_optimal_hyperparameters(
    index,
    index_key: str,
    max_speed: float,  # milliseconds
    use_gpu: bool = False,
    max_timeout_per_iteration: float = 1.0,  # seconds
    min_ef_search: int = 32,
) -> str:
    """Find the optimal hyperparameters to maximize the recall given a query speed in milliseconds/query"""

    # nb_vectors = index.ntotal

    if any(re.findall(r"OPQ\d+_\d+,IVF\d+,PQ\d+", index_key)):
        with Timeit("starting initial reconstruct"):
            query = index.reconstruct_n(0, 4000)
        get_speed = partial(
            speed_test_ms_per_query,
            query=query,
            ksearch=40,
            timout_s=min(max_timeout_per_iteration, 15 * max_speed / 1000),
        )

        #
        nprobes = list(range(max(1, min_ef_search // 2), 6144))
        nprobes.sort()

        ht = 2048

        def is_not_acceptable_speed(rank_nprobe):

            nprobe = nprobes[rank_nprobe]
            param_str = f"nprobe={nprobe},ht={ht}"
            set_search_hyperparameters(index, param_str, use_gpu)
            speed = get_speed(index)

            return speed >= max_speed

        with Timeit("starting initial binary search"):
            rank_best_nprobe = discrete_binary_search(is_not_acceptable_speed, len(nprobes)) - 1

        # make sure that the query time is respected
        with Timeit("starting final reconstruct"):
            query = index.reconstruct_n(0, min(index.ntotal, 50000))

        get_speed = partial(
            speed_test_ms_per_query,
            query=query,
            ksearch=40,
            timout_s=min(max_timeout_per_iteration, 200 * max_speed / 1000),
        )

        with Timeit("final linear search"):
            while is_not_acceptable_speed(rank_best_nprobe) and rank_best_nprobe > 1:
                rank_best_nprobe -= 1

        rank_best_nprobe = max(0, min(rank_best_nprobe, len(nprobes) - 1))

        nprobe = nprobes[rank_best_nprobe]

        param_str = f"nprobe={nprobe},ht={ht}"

        return param_str

    elif any(re.findall(r"OPQ\d+_\d+,IVF\d+_HNSW\d+,PQ\d+", index_key)):

        with Timeit("starting initial reconstruct"):
            query = index.reconstruct_n(0, 4000)
        get_speed = partial(
            speed_test_ms_per_query,
            query=query,
            ksearch=40,
            timout_s=min(max_timeout_per_iteration, 15 * max_speed / 1000),
        )

        #
        nprobes = list(range(max(1, min_ef_search // 2), 6144))
        nprobes.sort()

        ht = 2048

        def is_not_acceptable_speed(rank_nprobe):

            nprobe = nprobes[rank_nprobe]
            param_str = f"nprobe={nprobe},efSearch={2*nprobe},ht={ht}"
            set_search_hyperparameters(index, param_str, use_gpu)
            speed = get_speed(index)

            return speed >= max_speed

        with Timeit("starting initial binary search"):
            rank_best_nprobe = discrete_binary_search(is_not_acceptable_speed, len(nprobes)) - 1

        # make sure that the query time is respected
        with Timeit("starting final reconstruct"):
            query = index.reconstruct_n(0, min(index.ntotal, 50000))

        get_speed = partial(
            speed_test_ms_per_query,
            query=query,
            ksearch=40,
            timout_s=min(max_timeout_per_iteration, 200 * max_speed / 1000),
        )

        with Timeit("final linear search"):
            while is_not_acceptable_speed(rank_best_nprobe) and rank_best_nprobe > 1:
                rank_best_nprobe -= 1

        rank_best_nprobe = max(0, min(rank_best_nprobe, len(nprobes) - 1))

        nprobe = nprobes[rank_best_nprobe]

        param_str = f"nprobe={nprobe},efSearch={2*nprobe},ht={ht}"

        return param_str

    if any(re.findall(r"HNSW\d+", index_key)):

        query = index.reconstruct_n(0, min(index.ntotal, 4000))
        get_speed = partial(
            speed_test_ms_per_query,
            query=query,
            ksearch=40,
            timout_s=min(max_timeout_per_iteration, 15 * max_speed / 1000),
        )

        ef_searchs = list(range(16, 2 ** 14))

        def is_not_acceptable_speed_ef_search(rank_efsearch):

            ef_search = ef_searchs[rank_efsearch]
            param_str = f"efSearch={ef_search}"
            set_search_hyperparameters(index, param_str, use_gpu)
            speed = get_speed(index)

            return speed >= max_speed

        rank_best_ef_search = discrete_binary_search(is_not_acceptable_speed_ef_search, len(ef_searchs)) - 1

        # make sure that the query time is respected
        query = index.reconstruct_n(0, min(index.ntotal, 50000))
        get_speed = partial(
            speed_test_ms_per_query,
            query=query,
            ksearch=40,
            timout_s=min(max_timeout_per_iteration, 100 * max_speed / 1000),
        )

        while is_not_acceptable_speed_ef_search(rank_best_ef_search) and rank_best_ef_search > 1:
            rank_best_ef_search -= 1

        rank_best_ef_search = max(0, min(rank_best_ef_search, len(ef_searchs) - 1))

        ef_search = ef_searchs[rank_best_ef_search]

        param_str = f"efSearch={ef_search}"

        return param_str

    if index_key == "Flat":
        return ""

    raise NotImplementedError("we don't have heuristics for that kind or index")
