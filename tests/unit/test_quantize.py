import logging
import os
import py
import random
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import Tuple, List

import faiss
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from numpy.testing import assert_array_equal

LOGGER = logging.getLogger(__name__)

from autofaiss import build_index, build_partitioned_indexes

logging.basicConfig(level=logging.DEBUG)

# hide py4j DEBUG from pyspark, otherwise, there will be too many useless debugging output
# https://stackoverflow.com/questions/37252527/how-to-hide-py4j-java-gatewayreceived-command-c-on-object-id-p0
logging.getLogger("py4j").setLevel(logging.ERROR)


def build_test_collection_numpy(
    tmpdir: py.path, min_size=2, max_size=10000, dim=512, nb_files=5, tmpdir_name: str = "autofaiss_numpy"
):
    tmp_path = tmpdir.mkdir(tmpdir_name)
    sizes = [random.randint(min_size, max_size) for _ in range(nb_files)]
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
    tmpdir: py.path,
    min_size=2,
    max_size=10000,
    dim=512,
    nb_files=5,
    tmpdir_name: str = "autofaiss_parquet",
    consecutive_ids=False,
):
    tmp_path = tmpdir.mkdir(tmpdir_name)
    sizes = [random.randint(min_size, max_size) for _ in range(nb_files)]
    dim = dim
    all_dfs = []
    file_paths = []
    n = 0
    for i, size in enumerate(sizes):
        arr = np.random.rand(size, dim).astype("float32")
        if consecutive_ids:
            # ids would be consecutive from 0 to N-1
            ids = list(range(n, n + size))
        else:
            ids = np.random.randint(max_size * nb_files * 10, size=size)
        df = pd.DataFrame({"embedding": list(arr), "id": ids})
        all_dfs.append(df)
        file_path = os.path.join(tmp_path, f"{str(i)}.parquet")
        df.to_parquet(file_path)
        file_paths.append(file_path)
        n += len(df)
    all_dfs = pd.concat(all_dfs)
    return str(tmp_path), sizes, dim, all_dfs, file_paths


def test_quantize(tmpdir):
    min_size = random.randint(1, 100)
    max_size = random.randint(min_size, 10240)
    dim = random.randint(1, 100)
    nb_files = random.randint(1, 5)

    tmp_dir, sizes, dim, expected_array, _ = build_test_collection_numpy(
        tmpdir, min_size=min_size, max_size=max_size, dim=dim, nb_files=nb_files
    )

    output_numpy_index = os.path.join(tmpdir.strpath, "numpy_knn.index")
    output_numpy_index_infos = os.path.join(tmpdir.strpath, "numpy_knn_infos.json")

    build_index(
        embeddings=tmp_dir,
        file_format="npy",
        index_path=output_numpy_index,
        index_infos_path=output_numpy_index_infos,
        max_index_query_time_ms=10.0,
        max_index_memory_usage="1G",
        current_memory_available="2G",
    )
    output_numpy_index_faiss = faiss.read_index(output_numpy_index)
    assert output_numpy_index_faiss.ntotal == len(expected_array)

    tmp_dir, sizes, dim, expected_df, _ = build_test_collection_parquet(
        tmpdir, min_size=min_size, max_size=max_size, dim=dim, nb_files=nb_files
    )

    index_path = os.path.join(tmpdir.strpath, "parquet_knn.index")
    index_infos_path = os.path.join(tmpdir.strpath, "infos.json")

    build_index(
        embeddings=tmp_dir,
        file_format="parquet",
        embedding_column_name="embedding",
        index_path=index_path,
        index_infos_path=index_infos_path,
        max_index_query_time_ms=10.0,
        max_index_memory_usage="1G",
        current_memory_available="2G",
    )
    output_parquet_index_faiss = faiss.read_index(index_path)
    assert output_parquet_index_faiss.ntotal == len(expected_df)


def test_quantize_with_ids(tmpdir):
    min_size = random.randint(1, 100)
    max_size = random.randint(min_size, 10240)
    dim = random.randint(1, 100)
    nb_files = random.randint(1, 5)

    tmp_dir, sizes, dim, expected_df, _ = build_test_collection_parquet(
        tmpdir, min_size=min_size, max_size=max_size, dim=dim, nb_files=nb_files
    )

    index_path = os.path.join(tmpdir.strpath, "parquet_knn.index")
    index_infos_path = os.path.join(tmpdir.strpath, "infos.json")
    ids_path = os.path.join(tmpdir.strpath, "ids")

    build_index(
        embeddings=tmp_dir,
        file_format="parquet",
        embedding_column_name="embedding",
        index_path=index_path,
        index_infos_path=index_infos_path,
        ids_path=ids_path,
        max_index_query_time_ms=10.0,
        max_index_memory_usage="1G",
        current_memory_available="2G",
        id_columns=["id"],
    )
    output_parquet_index_faiss = faiss.read_index(index_path)
    output_parquet_ids = pq.read_table(ids_path).to_pandas()
    assert output_parquet_index_faiss.ntotal == len(expected_df)

    expected_df["i"] = np.arange(start=0, stop=len(expected_df))
    pd.testing.assert_frame_equal(
        output_parquet_ids.reset_index(drop=True), expected_df[["id", "i"]].reset_index(drop=True)
    )


def test_quantize_with_pyspark(tmpdir):
    min_size = random.randint(1, 100)
    max_size = random.randint(min_size, 10240)
    dim = random.randint(1, 100)
    nb_files = random.randint(1, 5)
    tmp_dir, _, _, expected_df, _ = build_test_collection_parquet(
        tmpdir, min_size=min_size, max_size=max_size, dim=dim, nb_files=nb_files
    )
    index_parquet_path = os.path.join(tmpdir.strpath, "parquet_knn.index")
    output_parquet_index_infos = os.path.join(tmpdir.strpath, "infos.json")
    ids_path = os.path.join(tmpdir.strpath, "ids")
    temporary_indices_folder = os.path.join(tmpdir.strpath, "distributed_autofaiss_indices")
    build_index(
        embeddings=tmp_dir,
        distributed="pyspark",
        file_format="parquet",
        temporary_indices_folder=temporary_indices_folder,
        index_infos_path=output_parquet_index_infos,
        ids_path=ids_path,
        max_index_memory_usage="1G",
        current_memory_available="2G",
        id_columns=["id"],
        embedding_column_name="embedding",
        index_path=index_parquet_path,
    )
    output_parquet_index_faiss = faiss.read_index(index_parquet_path)
    output_parquet_ids = pq.read_table(ids_path).to_pandas()
    assert output_parquet_index_faiss.ntotal == len(expected_df)
    pd.testing.assert_frame_equal(
        output_parquet_ids[["id"]].reset_index(drop=True), expected_df[["id"]].reset_index(drop=True)
    )

    tmp_dir, _, _, expected_array, _ = build_test_collection_numpy(
        tmpdir, min_size=min_size, max_size=max_size, dim=dim, nb_files=nb_files
    )
    output_numpy_index = os.path.join(tmpdir.strpath, "numpy_knn.index")
    output_numpy_index_infos = os.path.join(tmpdir.strpath, "numpy_knn_infos.json")

    build_index(
        embeddings=tmp_dir,
        distributed="pyspark",
        file_format="npy",
        temporary_indices_folder=temporary_indices_folder,
        index_infos_path=output_numpy_index_infos,
        max_index_memory_usage="1G",
        current_memory_available="2G",
        embedding_column_name="embedding",
        index_path=output_numpy_index,
    )

    output_numpy_index_faiss = faiss.read_index(output_numpy_index)
    assert output_numpy_index_faiss.ntotal == len(expected_array)


def test_quantize_with_multiple_inputs(tmpdir):
    min_size = random.randint(1, 100)
    max_size = random.randint(min_size, 10240)
    dim = random.randint(1, 100)
    nb_files = random.randint(1, 5)
    tmp_dir1, _, _, expected_df1, _ = build_test_collection_parquet(
        tmpdir, min_size=min_size, max_size=max_size, dim=dim, nb_files=nb_files, tmpdir_name="autofaiss_parquet1"
    )
    tmp_dir2, _, _, expected_df2, _ = build_test_collection_parquet(
        tmpdir, min_size=min_size, max_size=max_size, dim=dim, nb_files=nb_files, tmpdir_name="autofaiss_parquet2"
    )
    expected_df = pd.concat([expected_df1, expected_df2])
    index_parquet_path = os.path.join(tmpdir.strpath, "parquet_knn.index")
    output_parquet_index_infos = os.path.join(tmpdir.strpath, "infos.json")
    build_index(
        embeddings=[tmp_dir1, tmp_dir2],
        file_format="parquet",
        embedding_column_name="embedding",
        index_path=index_parquet_path,
        index_infos_path=output_parquet_index_infos,
        max_index_query_time_ms=10.0,
        max_index_memory_usage="1G",
        current_memory_available="2G",
    )
    output_parquet_index_faiss = faiss.read_index(index_parquet_path)
    assert output_parquet_index_faiss.ntotal == len(expected_df)


def test_quantize_with_empty_file():
    with TemporaryDirectory() as tmp_dir:
        with NamedTemporaryFile() as tmp_file:
            df = pd.DataFrame({"embedding": [], "id": []})
            df.to_parquet(os.path.join(tmp_dir, tmp_file.name))
            with pytest.raises(ValueError):
                build_index(embeddings=tmp_dir, file_format="parquet", embedding_column_name="embedding")


def test_quantize_with_empty_and_non_empty_files(tmpdir):
    with TemporaryDirectory() as tmp_empty_dir:
        with NamedTemporaryFile() as tmp_file:
            df = pd.DataFrame({"embedding": [], "id": []})
            df.to_parquet(os.path.join(tmp_empty_dir, tmp_file.name))
            min_size = random.randint(1, 100)
            max_size = random.randint(min_size, 10240)
            dim = random.randint(1, 100)
            nb_files = random.randint(1, 5)
            tmp_non_empty_dir, _, _, expected_df, _ = build_test_collection_parquet(
                tmpdir,
                min_size=min_size,
                max_size=max_size,
                dim=dim,
                nb_files=nb_files,
                tmpdir_name="autofaiss_parquet1",
            )
            index_parquet_path = os.path.join(tmpdir.strpath, "parquet_knn.index")
            output_parquet_index_infos = os.path.join(tmpdir.strpath, "infos.json")
            build_index(
                embeddings=[tmp_empty_dir, tmp_non_empty_dir],
                file_format="parquet",
                embedding_column_name="embedding",
                index_path=index_parquet_path,
                index_infos_path=output_parquet_index_infos,
                max_index_query_time_ms=10.0,
                max_index_memory_usage="1G",
                current_memory_available="2G",
            )
            output_parquet_index_faiss = faiss.read_index(index_parquet_path)
            assert output_parquet_index_faiss.ntotal == len(expected_df)


def test_index_correctness_in_distributed_mode(tmpdir):
    min_size = 8000
    max_size = 10240
    dim = 512
    nb_files = 5

    # parquet
    tmp_dir, _, _, expected_df, _ = build_test_collection_parquet(
        tmpdir, min_size=min_size, max_size=max_size, dim=dim, nb_files=nb_files, consecutive_ids=True
    )
    temporary_indices_folder = os.path.join(tmpdir.strpath, "distributed_autofaiss_indices")
    ids_path = os.path.join(tmpdir.strpath, "ids")
    index, _ = build_index(
        embeddings=tmp_dir,
        distributed="pyspark",
        file_format="parquet",
        temporary_indices_folder=temporary_indices_folder,
        max_index_memory_usage="600MB",
        current_memory_available="700MB",
        embedding_column_name="embedding",
        index_key="IVF1,Flat",
        should_be_memory_mappable=True,
        metric_type="l2",
        ids_path=ids_path,
        save_on_disk=True,
        id_columns=["id"],
    )
    query = faiss.rand((1, dim))
    distances, ids = index.search(query, k=9)
    ground_truth_index = faiss.index_factory(dim, "IVF1,Flat")
    expected_array = np.stack(expected_df["embedding"])
    ground_truth_index.train(expected_array)
    ground_truth_index.add(expected_array)
    ground_truth_distances, ground_truth_ids = ground_truth_index.search(query, k=9)

    ids_mappings = pd.read_parquet(ids_path)["id"]
    assert len(ids_mappings) == len(expected_df)
    assert_array_equal(ids_mappings.iloc[ids[0, :]].to_numpy(), ids[0, :])
    assert_array_equal(ids, ground_truth_ids)

    # numpy
    tmp_dir, _, _, expected_array, _ = build_test_collection_numpy(
        tmpdir, min_size=min_size, max_size=max_size, dim=dim, nb_files=nb_files
    )
    index, _ = build_index(
        embeddings=tmp_dir,
        distributed="pyspark",
        file_format="npy",
        temporary_indices_folder=temporary_indices_folder,
        max_index_memory_usage="400MB",
        current_memory_available="500MB",
        embedding_column_name="embedding",
        index_key="IVF1,Flat",
        should_be_memory_mappable=True,
        metric_type="l2",
    )
    query = faiss.rand((1, dim))
    distances, ids = index.search(query, k=9)
    ground_truth_index = faiss.index_factory(dim, "IVF1,Flat")
    ground_truth_index.train(expected_array)
    ground_truth_index.add(expected_array)
    ground_truth_distances, ground_truth_ids = ground_truth_index.search(query, k=9)
    assert_array_equal(ids, ground_truth_ids)


def _search_from_multiple_indices(index_paths, query, k):
    all_distances, all_ids, NB_QUERIES = [], [], 1
    for rest_index_file in index_paths:
        index = faiss.read_index(rest_index_file)
        distances, ids = index.search(query, k=k)
        all_distances.append(distances)
        all_ids.append(ids)

    dists_arr = np.stack(all_distances, axis=1).reshape(NB_QUERIES, -1)
    knn_ids_arr = np.stack(all_ids, axis=1).reshape(NB_QUERIES, -1)

    sorted_k_indices = np.argsort(-dists_arr)[:, :k]
    sorted_k_dists = np.take_along_axis(dists_arr, sorted_k_indices, axis=1)
    sorted_k_ids = np.take_along_axis(knn_ids_arr, sorted_k_indices, axis=1)
    return sorted_k_dists, sorted_k_ids


def _merge_indices(index_paths):
    merged = faiss.read_index(index_paths[0])
    for rest_index_file in index_paths[1:]:
        index = faiss.read_index(rest_index_file)
        faiss.merge_into(merged, index, shift_ids=False)
    return merged


def test_index_correctness_in_distributed_mode_with_multiple_indices(tmpdir):
    min_size = 20000
    max_size = 40000
    dim = 512
    nb_files = 5

    # parquet
    tmp_dir, _, _, expected_df, _ = build_test_collection_parquet(
        tmpdir, min_size=min_size, max_size=max_size, dim=dim, nb_files=nb_files, consecutive_ids=True
    )
    temporary_indices_folder = os.path.join(tmpdir.strpath, "distributed_autofaiss_indices")
    ids_path = os.path.join(tmpdir.strpath, "ids")
    _, index_path2_metric_infos = build_index(
        embeddings=tmp_dir,
        distributed="pyspark",
        file_format="parquet",
        temporary_indices_folder=temporary_indices_folder,
        max_index_memory_usage="2GB",
        current_memory_available="500MB",
        embedding_column_name="embedding",
        index_key="IVF1,Flat",
        should_be_memory_mappable=True,
        ids_path=ids_path,
        nb_indices_to_keep=2,
        save_on_disk=True,
        id_columns=["id"],
    )
    index_paths = sorted(index_path2_metric_infos.keys())
    K, NB_QUERIES = 5, 1
    query = faiss.rand((NB_QUERIES, dim))

    ground_truth_index = faiss.index_factory(dim, "IVF1,Flat", faiss.METRIC_INNER_PRODUCT)
    expected_array = np.stack(expected_df["embedding"])
    ground_truth_index.train(expected_array)
    ground_truth_index.add(expected_array)
    _, ground_truth_ids = ground_truth_index.search(query, k=K)

    ids_mappings = pd.read_parquet(ids_path)["id"]
    assert len(ids_mappings) == len(expected_df)
    assert_array_equal(ids_mappings.iloc[ground_truth_ids[0, :]].to_numpy(), ground_truth_ids[0, :])

    _, sorted_k_ids = _search_from_multiple_indices(index_paths=index_paths, query=query, k=K)
    merged = _merge_indices(index_paths)
    _, ids = merged.search(query, k=K)
    assert_array_equal(ids, ground_truth_ids)
    assert_array_equal(sorted_k_ids, ground_truth_ids)

    # numpy
    tmp_dir, _, _, expected_array, _ = build_test_collection_numpy(
        tmpdir, min_size=min_size, max_size=max_size, dim=dim, nb_files=nb_files
    )

    temporary_indices_folder = os.path.join(tmpdir.strpath, "distributed_autofaiss_indices")
    _, index_path2_metric_infos = build_index(
        embeddings=tmp_dir,
        distributed="pyspark",
        file_format="npy",
        temporary_indices_folder=temporary_indices_folder,
        max_index_memory_usage="2GB",
        current_memory_available="500MB",
        embedding_column_name="embedding",
        index_key="IVF1,Flat",
        should_be_memory_mappable=True,
        nb_indices_to_keep=2,
    )

    ground_truth_index = faiss.index_factory(dim, "IVF1,Flat", faiss.METRIC_INNER_PRODUCT)
    ground_truth_index.train(expected_array)
    ground_truth_index.add(expected_array)
    _, ground_truth_ids = ground_truth_index.search(query, k=K)

    index_paths = sorted(index_path2_metric_infos.keys())
    _, sorted_k_ids = _search_from_multiple_indices(index_paths=index_paths, query=query, k=K)

    merged = _merge_indices(index_paths)
    _, ids = merged.search(query, k=K)
    assert_array_equal(ids, ground_truth_ids)
    assert_array_equal(sorted_k_ids, ground_truth_ids)


def test_build_partitioned_indexes(tmpdir):
    embedding_root_dir = tmpdir.mkdir("embeddings")
    output_root_dir = tmpdir.mkdir("outputs")
    temp_root_dir = tmpdir.strpath
    small_partitions = [("partnerId=123", 1), ("partnerId=44", 2)]
    big_partitions = [("partnerId=22", 3)]
    all_partitions = small_partitions + big_partitions
    expected_embeddings, partitions = _create_partitioned_parquet_embedding_dataset(
        embedding_root_dir, all_partitions, n_dimensions=3
    )

    nb_splits_per_big_index = 2
    metrics = build_partitioned_indexes(
        partitions=partitions,
        output_root_dir=str(output_root_dir),
        embedding_column_name="embedding",
        id_columns=["id"],
        temp_root_dir=str(temp_root_dir),
        nb_splits_per_big_index=nb_splits_per_big_index,
        big_index_threshold=3,
        should_be_memory_mappable=True,
    )

    assert len(all_partitions) == len(metrics)

    all_ids = []
    for partition_name, partition_size in small_partitions:
        index_path = os.path.join(output_root_dir, partition_name, "knn.index")
        index = faiss.read_index(index_path)
        assert partition_size == index.ntotal
        ids_path = os.path.join(output_root_dir, partition_name, "ids")
        ids = pq.read_table(ids_path).to_pandas()
        all_ids.append(ids)

    for partition_name, partition_size in big_partitions:
        n_embeddings = 0
        for i in range(nb_splits_per_big_index):
            index_path = os.path.join(output_root_dir, partition_name, f"knn.index{i}")
            index = faiss.read_index(index_path)
            n_embeddings += index.ntotal
        assert partition_size == n_embeddings
        ids_path = os.path.join(output_root_dir, partition_name, "ids")
        ids = pq.read_table(ids_path).to_pandas()
        all_ids.append(ids)

    all_ids = pd.concat(all_ids)
    pd.testing.assert_frame_equal(
        all_ids[["id"]].reset_index(drop=True), expected_embeddings[["id"]].reset_index(drop=True)
    )


def _create_partitioned_parquet_embedding_dataset(
    embedding_root_dir: str, partition_sizes: List[Tuple[str, int]], n_dimensions: int = 512
):
    partition_embeddings = []
    partitions = []
    n = 0
    for i, (partition_name, partition_size) in enumerate(partition_sizes):
        embeddings = np.random.rand(partition_size, n_dimensions).astype("float32")
        ids = list(range(n, n + partition_size))
        df = pd.DataFrame({"embedding": list(embeddings), "id": ids})
        partition_embeddings.append(df)
        partition_dir = os.path.join(embedding_root_dir, partition_name)
        os.mkdir(partition_dir)
        partitions.append(partition_dir)
        file_path = os.path.join(partition_dir, f"{str(i)}.parquet")
        df.to_parquet(file_path)
        n += len(df)
    all_embeddings = pd.concat(partition_embeddings)
    return all_embeddings, partitions
