import logging

import faiss
import numpy as np
import pytest
from autofaiss.external.optimize import get_optimal_hyperparameters, get_optimal_index_keys_v2
from autofaiss.indices.index_factory import index_factory
from autofaiss.indices.index_utils import set_search_hyperparameters, speed_test_ms_per_query

LOGGER = logging.getLogger(__name__)


@pytest.mark.parametrize("nb_vectors", [10, 900, 9_000, 90_000, 900_000, 9_000_000])
@pytest.mark.parametrize("dim_vector", [10, 100])
@pytest.mark.parametrize("max_index_memory_usage", ["1K", "1M", "1G"])
def test_get_optimal_index_keys_v2(nb_vectors: int, dim_vector: int, max_index_memory_usage: str) -> None:

    # Check that should_be_memory_mappable returns only ivf indices
    for index_key in get_optimal_index_keys_v2(
        nb_vectors, dim_vector, max_index_memory_usage, should_be_memory_mappable=True
    ):
        # LOGGER.debug(f"nb_vectors={nb_vectors}, max_mem={max_index_memory_usage} -> {index_key}")
        assert "IVF" in index_key


@pytest.mark.parametrize(
    "nb_vectors, use_gpu, expected",
    [
        (999_999, False, "IVF4096,Flat"),
        (1_000_000, False, "OPQ256_768,IVF16384_HNSW32,PQ256x8"),
        (1_000_000, True, "IVF16384,Flat"),
    ],
)
def test_get_optimal_index_keys_v2_with_large_nb_vectors(nb_vectors: int, use_gpu, expected: str):
    assert (
        get_optimal_index_keys_v2(
            nb_vectors=nb_vectors,
            dim_vector=512,
            max_index_memory_usage="50G",
            should_be_memory_mappable=True,
            ivf_flat_threshold=1_000_000,
            use_gpu=use_gpu,
        )[0]
        == expected
    )


# @pytest.mark.skip(reason="This test takes too long to run (11m)")
@pytest.mark.parametrize(
    "index_key", ["OPQ64_128,IVF1024_HNSW32,PQ64x8", "OPQ64_128,IVF1024,PQ64x8", "IVF256,Flat", "HNSW15",],
)
@pytest.mark.parametrize("d", [100])
def test_get_optimal_hyperparameters(index_key: str, d: int) -> None:
    """
    Check that get_optimal_hyperparameters returns an hyperparameter string that
    match with the speed constraint of the index.
    """

    nb_vectors_list = [1000, 100000]
    target_speed_ms_list = [0.5, 1, 10, 50]
    min_ef_search = 32
    use_gpu = False

    embeddings = np.float32(np.random.rand(max(nb_vectors_list), d))
    index = index_factory(d, index_key, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings[:10000])

    for nb_vec_in, target_nb_vec in zip([0] + nb_vectors_list, nb_vectors_list):

        index.add(embeddings[nb_vec_in:target_nb_vec])
        assert index.ntotal == target_nb_vec

        for target_speed_ms in target_speed_ms_list:

            hyperparameters_str = get_optimal_hyperparameters(
                index,
                index_key,
                target_speed_ms,
                use_gpu,
                max_timeout_per_iteration_s=1.0,
                min_ef_search=min_ef_search,
            )

            set_search_hyperparameters(index, hyperparameters_str, use_gpu)

            avg_query_time_ms = speed_test_ms_per_query(index,)

            LOGGER.debug(
                f"nb_vectors={target_nb_vec}, max_mem={index_key}, target_speed_ms {target_speed_ms} -> avg_query_time_ms: {avg_query_time_ms}, {hyperparameters_str}"
            )

            if (
                "nprobe=1" == hyperparameters_str
                or "nprobe=1," in hyperparameters_str
                or "efSearch=1" == hyperparameters_str
                or "efSearch=1," in hyperparameters_str
                or f"efSearch={min_ef_search}," in hyperparameters_str
                or f"efSearch={min_ef_search}" == hyperparameters_str
            ):
                # Target_speed is too constraining
                assert avg_query_time_ms >= target_speed_ms * 0.90 - 0.25
                continue

            assert avg_query_time_ms <= 1.05 * target_speed_ms + 0.25  # ms
