import logging
import os
import random
from tempfile import TemporaryDirectory, NamedTemporaryFile

from fsspec.implementations.local import LocalFileSystem

from autofaiss.readers.embeddings_iterators import (
    read_first_file_shape,
    read_embeddings,
    read_total_nb_vectors_and_dim,
    get_file_list,
)
import numpy as np
import pandas as pd
import py

logging.basicConfig(level=logging.DEBUG)


def list_file_paths(dir_path):
    return [os.path.join(dir_path, filename) for filename in os.listdir(dir_path)]


def build_test_collection_numpy(tmpdir: py.path, min_size=2, max_size=10000, dim=512, nb_files=5):
    tmp_path = tmpdir.mkdir("autofaiss_numpy")
    sizes = [random.randint(min_size, max_size) for _ in range(nb_files)]
    dim = dim
    all_arrays = []
    file_paths = []
    for i, size in enumerate(sizes):
        arr = np.random.rand(size, dim).astype("float32")
        all_arrays.append(arr)
        file_path = os.path.join(tmp_path, f"{str(i)}.npy")
        file_paths.append(file_path)
        np.save(file_path, arr)
    all_arrays = np.vstack(all_arrays)
    return str(tmp_path), sizes, dim, all_arrays, file_paths


def build_test_collection_parquet(
    tmpdir: py.path, min_size=2, max_size=10000, dim=512, nb_files=5, tmpdir_name: str = "autofaiss_parquet"
):
    tmp_path = tmpdir.mkdir(tmpdir_name)
    print(tmp_path)
    sizes = [random.randint(min_size, max_size) for _ in range(nb_files)]
    dim = dim
    all_dfs = []
    file_paths = []
    for i, size in enumerate(sizes):
        arr = np.random.rand(size, dim).astype("float32")
        ids = np.random.randint(max_size * nb_files * 10, size=size)
        df = pd.DataFrame({"embedding": list(arr), "id": ids})
        all_dfs.append(df)
        file_path = os.path.join(tmp_path, f"{str(i)}.parquet")
        df.to_parquet(file_path)
        file_paths.append(file_path)
    all_dfs = pd.concat(all_dfs)
    return str(tmp_path), sizes, dim, all_dfs, file_paths


def test_read_first_file_shape(tmpdir):
    expected_size = 12547
    expected_dim = 512
    tmp_dir, sizes, dim, _, _ = build_test_collection_numpy(
        tmpdir, min_size=expected_size, max_size=expected_size, dim=expected_dim, nb_files=5
    )
    num_rows, dim = read_first_file_shape(list_file_paths(tmp_dir), file_format="npy")
    assert num_rows == expected_size
    assert dim == expected_dim

    tmp_dir, sizes, dim, _, _ = build_test_collection_parquet(
        tmpdir, min_size=expected_size, max_size=expected_size, dim=expected_dim, nb_files=5
    )
    num_rows, dim = read_first_file_shape(
        list_file_paths(tmp_dir), file_format="parquet", embedding_column_name="embedding"
    )
    assert num_rows == expected_size
    assert dim == expected_dim


def test_read_embeddings(tmpdir):
    min_size = 2
    max_size = 2048
    dim = 512
    nb_files = 5

    tmp_dir, sizes, dim, expected_array, tmp_paths = build_test_collection_numpy(
        tmpdir, min_size=min_size, max_size=max_size, dim=dim, nb_files=nb_files
    )
    batch_size = random.randint(min_size, max_size)
    it = read_embeddings(tmp_paths, file_format="npy", batch_size=batch_size)
    all_batches = list(it)
    all_shapes = [x[0].shape for x in all_batches]
    actual_array = np.vstack([x[0] for x in all_batches])

    assert all(s[0] == batch_size and s[1] == 512 for s in all_shapes[:-1])
    assert all_shapes[-1][0] <= batch_size and all_shapes[-1][1] == 512
    np.testing.assert_almost_equal(actual_array, expected_array)

    tmp_dir, sizes, dim, expected_df, tmp_paths = build_test_collection_parquet(
        tmpdir, min_size=min_size, max_size=max_size, dim=dim, nb_files=nb_files
    )
    expected_array = np.vstack(expected_df["embedding"])
    batch_size = random.randint(min_size, max_size)
    it = read_embeddings(tmp_paths, file_format="parquet", batch_size=batch_size, embedding_column_name="embedding")
    all_batches = list(it)
    all_shapes = [x[0].shape for x in all_batches]
    actual_array = np.vstack([x[0] for x in all_batches])

    assert all(s[0] == batch_size and s[1] == 512 for s in all_shapes[:-1])
    assert all_shapes[-1][0] <= batch_size and all_shapes[-1][1] == 512
    np.testing.assert_almost_equal(actual_array, expected_array)


def test_read_embeddings_with_ids(tmpdir):
    min_size = 2
    max_size = 2048
    dim = 512
    nb_files = 5

    tmp_dir, sizes, dim, expected_df, tmp_paths = build_test_collection_parquet(
        tmpdir, min_size=min_size, max_size=max_size, dim=dim, nb_files=nb_files
    )
    expected_array = np.vstack(expected_df["embedding"])
    batch_size = random.randint(min_size, max_size)
    it = read_embeddings(
        tmp_paths, file_format="parquet", batch_size=batch_size, embedding_column_name="embedding", id_columns=["id"],
    )
    all_batches = list(it)
    all_shapes = [x[0].shape for x in all_batches]
    actual_array = np.vstack([x[0] for x in all_batches])
    actual_ids = pd.concat([x[1] for x in all_batches])

    expected_df["i"] = np.arange(start=0, stop=len(expected_df))

    assert all(s[0] == batch_size and s[1] == 512 for s in all_shapes[:-1])
    assert all_shapes[-1][0] <= batch_size and all_shapes[-1][1] == 512
    np.testing.assert_almost_equal(actual_array, expected_array)
    pd.testing.assert_frame_equal(actual_ids.reset_index(drop=True), expected_df[["id", "i"]].reset_index(drop=True))


def test_read_total_nb_vectors(tmpdir):
    min_size = random.randint(1, 100)
    max_size = random.randint(min_size, 10240)
    dim = random.randint(1, 1000)
    nb_files = random.randint(1, 10)
    tmp_dir, sizes, dim, expected_array, tmp_paths = build_test_collection_numpy(
        tmpdir, min_size=min_size, max_size=max_size, dim=dim, nb_files=nb_files
    )
    expected_count = len(expected_array)
    actual_count, actual_dim, file_counts = read_total_nb_vectors_and_dim(tmp_paths, file_format="npy")

    assert actual_count == expected_count
    assert actual_dim == dim
    assert sum(file_counts) == actual_count

    tmp_dir, sizes, dim, expected_df, tmp_paths = build_test_collection_parquet(
        tmpdir, min_size=min_size, max_size=max_size, dim=dim, nb_files=nb_files
    )
    expected_count = len(expected_df)
    actual_count, actual_dim, file_counts = read_total_nb_vectors_and_dim(
        tmp_paths, file_format="parquet", embedding_column_name="embedding"
    )

    assert actual_count == expected_count
    assert actual_dim == dim
    assert sum(file_counts) == actual_count


def test_test_read_total_nb_vectors_with_empty_file():
    with TemporaryDirectory() as tmp_empty_dir:
        with NamedTemporaryFile() as tmp_file:
            df = pd.DataFrame({"embedding": [], "id": []})
            tmp_path = os.path.join(tmp_empty_dir, tmp_file.name)
            df.to_parquet(tmp_path)
            actual_count, actual_dim, file_counts = read_total_nb_vectors_and_dim(
                [tmp_path], file_format="parquet", embedding_column_name="embedding"
            )
            assert actual_count == 0
            assert actual_dim is None
            assert file_counts == [0]


def test_get_file_list_with_single_input(tmpdir):
    tmp_dir, _, _, _, _ = build_test_collection_parquet(tmpdir, tmpdir_name="a", nb_files=2)
    fs, paths = get_file_list(path=tmp_dir, file_format="parquet")
    assert isinstance(fs, LocalFileSystem)
    assert len(paths) == 2


def test_get_file_list_with_multiple_inputs(tmpdir):
    tmp_dir1, _, _, _, _ = build_test_collection_parquet(tmpdir, tmpdir_name="a", nb_files=2)
    tmp_dir2, _, _, _, _ = build_test_collection_parquet(tmpdir, tmpdir_name="b", nb_files=2)
    fs, paths = get_file_list(path=[tmp_dir1, tmp_dir2], file_format="parquet")
    assert isinstance(fs, LocalFileSystem)
    assert len(paths) == 4


def test_get_file_list_with_multiple_multiple_levels_input(tmpdir):
    tmp_dir1, _, _, _, _ = build_test_collection_parquet(tmpdir, tmpdir_name="a", nb_files=2)
    _, _, _, _, _ = build_test_collection_parquet(tmpdir, tmpdir_name="a/1", nb_files=2)
    _, _, _, _, _ = build_test_collection_parquet(tmpdir, tmpdir_name="a/1/2", nb_files=2)
    fs, paths = get_file_list(path=tmp_dir1, file_format="parquet")
    assert isinstance(fs, LocalFileSystem)
    assert len(paths) == 6
