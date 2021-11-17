import logging
import os
import random
import shutil

from autofaiss.datasets.readers.embeddings_iterators import (
    read_first_file_shape,
    read_embeddings,
    read_total_nb_vectors_and_dim,
)
import numpy as np
import pandas as pd
import py

logging.basicConfig(level=logging.DEBUG)


def build_test_collection_numpy(tmpdir: py.path, min_size=2, max_size=10000, dim=512, nb_files=5):
    tmp_path = tmpdir.mkdir("autofaiss_numpy")
    sizes = [random.randint(min_size, max_size) for _ in range(nb_files)]
    dim = dim
    all_arrays = []
    for i, size in enumerate(sizes):
        arr = np.random.rand(size, dim).astype("float32")
        all_arrays.append(arr)
        np.save(os.path.join(tmp_path, f"{str(i)}.npy"), arr)
    all_arrays = np.vstack(all_arrays)
    return str(tmp_path), sizes, dim, all_arrays


def build_test_collection_parquet(tmpdir: py.path, min_size=2, max_size=10000, dim=512, nb_files=5):
    tmp_path = tmpdir.mkdir("autofaiss_parquet")
    print(tmp_path)
    sizes = [random.randint(min_size, max_size) for _ in range(nb_files)]
    dim = dim
    all_dfs = []
    for i, size in enumerate(sizes):
        arr = np.random.rand(size, dim).astype("float32")
        ids = np.random.randint(max_size * nb_files * 10, size=size)
        df = pd.DataFrame({"embedding": list(arr), "id": ids})
        all_dfs.append(df)
        df.to_parquet(os.path.join(tmp_path, f"{str(i)}.parquet"))
    all_dfs = pd.concat(all_dfs)
    return str(tmp_path), sizes, dim, all_dfs


def test_read_first_file_shape(tmpdir):
    expected_size = 12547
    expected_dim = 512
    tmp_dir, sizes, dim, _ = build_test_collection_numpy(
        tmpdir, min_size=expected_size, max_size=expected_size, dim=expected_dim, nb_files=5
    )
    num_rows, dim = read_first_file_shape(tmp_dir, file_format="npy")
    assert num_rows == expected_size
    assert dim == expected_dim

    tmp_dir, sizes, dim, _ = build_test_collection_parquet(
        tmpdir, min_size=expected_size, max_size=expected_size, dim=expected_dim, nb_files=5
    )
    num_rows, dim = read_first_file_shape(tmp_dir, file_format="parquet", embedding_column_name="embedding")
    assert num_rows == expected_size
    assert dim == expected_dim


def test_read_embeddings(tmpdir):
    min_size = 2
    max_size = 2048
    dim = 512
    nb_files = 5

    tmp_dir, sizes, dim, expected_array = build_test_collection_numpy(
        tmpdir, min_size=min_size, max_size=max_size, dim=dim, nb_files=nb_files
    )
    batch_size = random.randint(min_size, max_size)
    it = read_embeddings(tmp_dir, file_format="npy", batch_size=batch_size)
    all_batches = list(it)
    all_shapes = [x.shape for x in all_batches]
    actual_array = np.vstack(all_batches)

    assert all(s[0] == batch_size and s[1] == 512 for s in all_shapes[:-1])
    assert all_shapes[-1][0] <= batch_size and all_shapes[-1][1] == 512
    np.testing.assert_almost_equal(actual_array, expected_array)

    tmp_dir, sizes, dim, expected_df = build_test_collection_parquet(
        tmpdir, min_size=min_size, max_size=max_size, dim=dim, nb_files=nb_files
    )
    expected_array = np.vstack(expected_df["embedding"])
    batch_size = random.randint(min_size, max_size)
    it = read_embeddings(tmp_dir, file_format="parquet", batch_size=batch_size, embedding_column_name="embedding")
    all_batches = list(it)
    all_shapes = [x.shape for x in all_batches]
    actual_array = np.vstack(all_batches)

    assert all(s[0] == batch_size and s[1] == 512 for s in all_shapes[:-1])
    assert all_shapes[-1][0] <= batch_size and all_shapes[-1][1] == 512
    np.testing.assert_almost_equal(actual_array, expected_array)


def test_read_total_nb_vectors(tmpdir):
    min_size = random.randint(1, 100)
    max_size = random.randint(min_size, 10240)
    dim = random.randint(1, 1000)
    nb_files = random.randint(1, 10)

    tmp_dir, sizes, dim, expected_array = build_test_collection_numpy(
        tmpdir, min_size=min_size, max_size=max_size, dim=dim, nb_files=nb_files
    )
    expected_count = len(expected_array)
    actual_count, actual_dim = read_total_nb_vectors_and_dim(tmp_dir, file_format="npy")

    assert actual_count == expected_count
    assert actual_dim == dim

    tmp_dir, sizes, dim, expected_df = build_test_collection_parquet(
        tmpdir, min_size=min_size, max_size=max_size, dim=dim, nb_files=nb_files
    )
    expected_count = len(expected_df)
    actual_count, actual_dim = read_total_nb_vectors_and_dim(
        tmp_dir, file_format="parquet", embedding_column_name="embedding"
    )

    assert actual_count == expected_count
    assert actual_dim == dim
