import logging
import os
import random

import faiss
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)

from autofaiss import build_index
from tests.unit.test_embeddings_iterators import build_test_collection_numpy, build_test_collection_parquet

logging.basicConfig(level=logging.DEBUG)


def test_quantize(tmpdir):
    min_size = random.randint(1, 100)
    max_size = random.randint(min_size, 10240)
    dim = random.randint(1, 100)
    nb_files = random.randint(1, 5)

    tmp_dir, sizes, dim, expected_array = build_test_collection_numpy(
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

    tmp_dir, sizes, dim, expected_df = build_test_collection_parquet(
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

    tmp_dir, sizes, dim, expected_df = build_test_collection_parquet(
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
