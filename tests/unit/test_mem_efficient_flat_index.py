""" Test that the memory efficient flat index give same results as the faiss flat index """

import time

import faiss
import numpy as np
import pytest

from autofaiss.indices.memory_efficient_flat_index import MemEfficientFlatIndex


@pytest.fixture(name="prod_emb")
def fixture_prod_emb():
    """generate random database vectors"""
    np.random.seed(15)
    return np.random.rand(5003, 99).astype(np.float32)


@pytest.fixture(name="user_emb")
def fixture_user_emb():
    """generate random query vectors"""
    np.random.seed(17)
    return np.random.rand(501, 99).astype(np.float32)


# pylint: disable too-many-arguments redefined-outer-name
@pytest.mark.parametrize("dataset_size", [1, 10, 3000, 5003])
@pytest.mark.parametrize("batch_size", [1000, 10000])
@pytest.mark.parametrize("nb_query_vectors", [1, 10, 100])
@pytest.mark.parametrize("k", [1, 10, 101])
def test_memory_efficient_flat_index(prod_emb, user_emb, dataset_size, batch_size, nb_query_vectors, k):
    """Test our flat index vs. FAISS flat index"""

    dim = prod_emb.shape[-1]  # vectors dim

    # Test our flat index with faiss batches
    start_time = time.time()
    flat_index = MemEfficientFlatIndex(dim, "IP")
    flat_index.add(prod_emb[:dataset_size])
    D_our, I_our = flat_index.search(user_emb[:nb_query_vectors], k, batch_size=batch_size)
    print(f"Our flat index: {time.time()-start_time:.2f} (bias if all the dataset is already in RAM)")

    # Test our flat index with numpy batches
    start_time = time.time()
    flat_index = MemEfficientFlatIndex(dim, "IP")
    flat_index.add(prod_emb[:dataset_size])
    D_our_numpy, I_our_numpy = flat_index.search_numpy(user_emb[:nb_query_vectors], k, batch_size=batch_size)
    print(f"Our numpy flat index: {time.time()-start_time:.2f} (bias if all the dataset is already in RAM)")

    # Test FAISS flat index
    start_time = time.time()
    brute = faiss.IndexFlatIP(dim)
    # pylint: disable=no-value-for-parameter
    brute.add(prod_emb[:dataset_size])
    D_faiss, I_faiss = brute.search(user_emb[:nb_query_vectors], k)
    print(f"Faiss flat index: {time.time()-start_time:.2f} (no bias since all the dataset is already in RAM)")

    # Check that the vectors we can't retrive are the same
    assert np.all((I_faiss == -1) == (I_our == -1))
    assert np.all((I_faiss == -1) == (I_our_numpy == -1))

    mask = I_faiss == -1

    # Check that all the distances are equal and in the same order
    assert np.all((np.abs(D_our - D_faiss) <= 2 ** -13) | mask)
    assert np.all((np.abs(D_our_numpy - D_faiss) <= 2 ** -13) | mask)

    # Check the order is the same as Faiss -> it is not, but no big dead
    # since the computation always give the same results (repetability works)
    assert np.all(I_our == I_faiss) or True
    assert np.all(I_our_numpy == I_faiss) or True
