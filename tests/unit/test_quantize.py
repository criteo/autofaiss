import logging
import os
import random
from tempfile import TemporaryDirectory, NamedTemporaryFile

import faiss
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from numpy.testing import assert_array_equal

LOGGER = logging.getLogger(__name__)

from autofaiss import build_index
from tests.unit.test_embeddings_iterators import build_test_collection_numpy, build_test_collection_parquet

logging.basicConfig(level=logging.DEBUG)

# hide py4j DEBUG from pyspark, otherwise, there will be too many useless debugging output
# https://stackoverflow.com/questions/37252527/how-to-hide-py4j-java-gatewayreceived-command-c-on-object-id-p0
logging.getLogger("py4j").setLevel(logging.ERROR)


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
    pd.testing.assert_frame_equal(output_parquet_ids.reset_index(drop=True), expected_df[["id"]].reset_index(drop=True))

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
                build_index(
                    embeddings=tmp_dir, file_format="parquet", embedding_column_name="embedding",
                )


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
        max_index_memory_usage="100MB",
        current_memory_available="120MB",
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
        max_index_memory_usage="100MB",
        current_memory_available="120MB",
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
    n = 0
    all_distances, all_ids, NB_QUERIES = [], [], 1
    for rest_index_file in index_paths:
        index = faiss.read_index(rest_index_file)
        distances, ids = index.search(query, k=k)
        all_distances.append(distances)
        all_ids.append(ids + n)
        n += index.ntotal

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
        faiss.merge_into(merged, index, shift_ids=True)
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
    index_path2_metric_infos = build_index(
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
    )
    index_paths = sorted(index_path2_metric_infos.keys())
    K, all_distances, all_ids, NB_QUERIES = 5, [], [], 1
    query = faiss.rand((NB_QUERIES, dim))

    #
    ground_truth_index = faiss.index_factory(dim, "IVF1,Flat", faiss.METRIC_INNER_PRODUCT)
    expected_array = np.stack(expected_df["embedding"])
    ground_truth_index.train(expected_array)
    ground_truth_index.add(expected_array)
    ground_truth_distances, ground_truth_ids = ground_truth_index.search(query, k=K)

    ids_mappings = pd.read_parquet(ids_path)["id"]
    assert len(ids_mappings) == len(expected_df)
    assert_array_equal(ids_mappings.iloc[ground_truth_ids[0, :]].to_numpy(), ground_truth_ids[0, :])

    _, sorted_k_ids = _search_from_multiple_indices(index_paths=index_paths, query=query, k=K)

    merged = _merge_indices(index_paths)
    distances, ids = merged.search(query, k=K)
    assert ground_truth_index.nprobe == merged.nprobe
    assert_array_equal(ids, ground_truth_ids)
    assert_array_equal(sorted_k_ids, ground_truth_ids)

    # numpy
    tmp_dir, _, _, expected_array, _ = build_test_collection_numpy(
        tmpdir, min_size=min_size, max_size=max_size, dim=dim, nb_files=nb_files
    )

    temporary_indices_folder = os.path.join(tmpdir.strpath, "distributed_autofaiss_indices")
    index_path2_metric_infos = build_index(
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
    ground_truth_distances, ground_truth_ids = ground_truth_index.search(query, k=K)

    index_paths = sorted(index_path2_metric_infos.keys())
    _, sorted_k_ids = _search_from_multiple_indices(index_paths=index_paths, query=query, k=K)

    merged = _merge_indices(index_paths)
    distances, ids = merged.search(query, k=K)
    assert_array_equal(ids, ground_truth_ids)
    assert_array_equal(sorted_k_ids, ground_truth_ids)
