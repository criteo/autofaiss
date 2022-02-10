"""
Building the index with pyspark.
"""
import math
import multiprocessing
import os
from functools import partial
from multiprocessing.pool import ThreadPool
from tempfile import TemporaryDirectory
from typing import List, Optional, Iterator, Tuple, Callable, Any

import faiss
import fsspec
import numpy as np
import pandas as pd
from fsspec import get_filesystem_class
from fsspec.implementations.hdfs import PyArrowHDFS
from tqdm import tqdm

from autofaiss.indices.index_utils import _get_index_from_bytes, _get_bytes_from_index
from autofaiss.readers.embeddings_iterators import get_matrix_reader, make_path_absolute
from autofaiss.utils.cast import cast_memory_to_bytes
from autofaiss.utils.decorators import Timeit


def _yield_embeddings_batch(
    embeddings_paths: List[str],
    chunk_sizes: List[int],
    file_system: fsspec.AbstractFileSystem,
    start: int,
    end: int,
    embedding_column_name: str,
    file_format: str,
    id_columns: Optional[List[str]] = None,
):
    """Lazy load a batch of embeddings."""
    if isinstance(file_system, PyArrowHDFS):
        file_system = get_filesystem_class("hdfs")()
    cur_start = cur_end = 0
    for chunk_size, file_path in zip(chunk_sizes, embeddings_paths):
        cur_end += chunk_size
        if cur_end < start:
            cur_start += chunk_size
            continue
        slice_start = max(0, start - cur_start)
        slice_end = min(chunk_size, end - cur_start)
        with get_matrix_reader(file_format, file_system, file_path, embedding_column_name, id_columns) as matrix_reader:
            yield matrix_reader.get_lazy_array().get_rows(start=slice_start, end=slice_end)
        if cur_end > end:
            break
        cur_start += chunk_size


def _generate_small_index_file_name(batch_id: int) -> str:
    return f"index_{batch_id}"


def _save_small_index(index: faiss.Index, batch_id: int, small_indices_folder: str) -> None:
    """Save index for one batch."""
    fs = _get_file_system(small_indices_folder)
    fs.mkdirs(small_indices_folder, exist_ok=True)
    small_index_filename = _generate_small_index_file_name(batch_id)
    with fsspec.open(small_index_filename, "wb").open() as f:
        faiss.write_index(index, faiss.PyCallbackIOWriter(f.write))
    dest_filepath = os.path.join(small_indices_folder, small_index_filename)
    fs.put(small_index_filename, dest_filepath)


def _add_index(
    start: int,
    end: int,
    broadcast_trained_index_bytes,
    file_system: fsspec.AbstractFileSystem,
    chunk_sizes: List[int],
    embeddings_file_paths: List[str],
    embedding_column_name: str,
    batch_id: int,
    small_indices_folder: str,
    file_format: str,
    id_columns: Optional[List[str]] = None,
    num_cores: Optional[int] = None,
    embedding_ids_df_handler: Optional[Callable[[pd.DataFrame, int], Any]] = None,
):
    """
    Add a batch of embeddings on trained index and save this index.

    Parameters
    ----------
    start: int
        Start position of this batch
    end: int
        End position of this batch
    broadcast_trained_index_bytes: pyspark.Broadcast
        Trained yet empty index
    chunk_sizes: List[int]
        A list of number of vectors in each embedding files
    embeddings_file_paths: List[str]
        A list of embeddings file paths
    embedding_column_name: str
        Embeddings column name for parquet ; default "embedding"
    batch_id: int
        Batch id
    small_indices_folder: str
        The folder where we save all the small indices
    num_cores: int
        Number of CPU cores (not Vcores)
    file_format: str
        Embedding file format, "npy" or "parquet"
    id_columns: Optional[List[str]]
        Names of the columns containing the Ids of the vectors, only used when file_format is "parquet"
    embedding_ids_df_handler: Optional[Callable[[pd.DataFrame, int], Any]]
        The function that handles the embeddings Ids when id_columns is given
    """
    total_chunk_sizes = sum(chunk_sizes)
    if end > total_chunk_sizes:
        end = total_chunk_sizes
    if len(chunk_sizes) != len(embeddings_file_paths):
        raise ValueError(
            f"The length of chunk_sizes should be equal to the number of embeddings_file_paths.\n"
            f"Got len(chunk_sizes)={len(chunk_sizes)},"
            f"len(embeddings_file_paths)={len(embeddings_file_paths)}.\n"
            f"chunk_sizes={chunk_sizes}\n"
            f"embeddings_file_paths={embeddings_file_paths}"
        )
    batch_vectors_gen, batch_ids_gen = zip(
        *_yield_embeddings_batch(
            embeddings_paths=embeddings_file_paths,
            chunk_sizes=chunk_sizes,
            file_system=file_system,
            start=start,
            end=end,
            embedding_column_name=embedding_column_name,
            id_columns=id_columns,
            file_format=file_format,
        )
    )

    embeddings_to_add = np.vstack(batch_vectors_gen).astype(np.float32)  # faiss requires float32 type

    if embedding_ids_df_handler is not None:
        embedding_ids_df_handler(pd.concat(batch_ids_gen), batch_id)

    if num_cores is None:
        num_cores = multiprocessing.cpu_count()

    faiss.omp_set_num_threads(num_cores)

    # load empty trained index
    empty_index = _get_index_from_bytes(broadcast_trained_index_bytes.value)

    empty_index.add(embeddings_to_add)

    del embeddings_to_add

    _save_small_index(index=empty_index, small_indices_folder=small_indices_folder, batch_id=batch_id)


def _get_pyspark_active_session():
    """Reproduce SparkSession.getActiveSession() available since pyspark 3.0."""
    import pyspark  # pylint: disable=import-outside-toplevel

    # pylint: disable=protected-access
    ss: Optional[pyspark.sql.SparkSession] = pyspark.sql.SparkSession._instantiatedSession  # mypy: ignore
    if ss is None:
        print("No pyspark session found, creating a new one!")
        ss = (
            pyspark.sql.SparkSession.builder.config("spark.driver.memory", "16G")
            .master("local[1]")
            .appName("Distributed autofaiss")
            .config("spark.submit.deployMode", "client")
            .getOrCreate()
        )
    return ss


def _batch_loader(batch_size: int, nb_batches: int) -> Iterator[Tuple[int, int, int]]:
    """Yield [batch id, batch start position, batch end position]"""
    for batch_id in range(nb_batches):
        start = batch_size * batch_id
        end = batch_size * (batch_id + 1)
        yield batch_id, start, end


def _download_one(src_dst_path: Tuple[str, str], fs: fsspec.AbstractFileSystem):
    src_path, dst_path = src_dst_path
    fs.get(src_path, dst_path)


def _parallel_download_indices_from_remote(
    fs: fsspec.AbstractFileSystem, indices_file_paths: List[str], dst_folder: str
):
    """Download small indices in parallel."""
    if len(indices_file_paths) == 0:
        return
    os.makedirs(dst_folder, exist_ok=True)
    dst_paths = [os.path.join(dst_folder, os.path.split(p)[-1]) for p in indices_file_paths]
    src_dest_paths = zip(indices_file_paths, dst_paths)
    with ThreadPool(min(16, len(indices_file_paths))) as pool:
        for _ in pool.imap_unordered(partial(_download_one, fs=fs), src_dest_paths):
            pass


def _get_stage2_folder(tmp_folder: str) -> str:
    """Get the temporary folder path used for the 2nd stage of merging"""
    return tmp_folder.rstrip("/") + "/stage-2/"


def _merge_index(
    small_indices_folder: str,
    batch_id: Optional[int] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    max_size_on_disk: str = "100GB",
) -> faiss.Index:
    """Merge all the indices in `small_indices_folder` into single one return the merged index."""
    fs = _get_file_system(small_indices_folder)
    small_indices_files = fs.ls(small_indices_folder, detail=False)
    small_indices_files = small_indices_files[start:end]

    if len(small_indices_files) == 0:
        raise ValueError(f"No small index is saved in {small_indices_folder}")

    def _merge_from_local(merged: Optional[faiss.Index] = None) -> faiss.Index:
        local_file_paths = [
            os.path.join(local_indices_folder, filename) for filename in os.listdir(local_indices_folder)
        ]
        if merged is None:
            merged = faiss.read_index(local_file_paths[0])
            start_index = 1
        else:
            start_index = 0

        for rest_index_file in tqdm(local_file_paths[start_index:]):
            index = faiss.read_index(rest_index_file)
            faiss.merge_into(merged, index, shift_ids=True)
        return merged

    # estimate index size by taking the first index
    first_index_file = small_indices_files[0]
    first_index_size = fs.size(first_index_file)
    max_sizes_in_bytes = cast_memory_to_bytes(max_size_on_disk)
    nb_files_each_time = math.floor(max_sizes_in_bytes / first_index_size)
    merged_index = None
    n = len(small_indices_files)
    nb_iterations = max(math.ceil(n / nb_files_each_time), 1)
    with Timeit("-> Merging small indices", indent=4):
        with tqdm(total=nb_iterations) as pbar:
            for i in range(nb_iterations):
                to_downloads = small_indices_files[i * nb_files_each_time : min(n, (i + 1) * nb_files_each_time)]
                with TemporaryDirectory() as local_indices_folder:
                    _parallel_download_indices_from_remote(
                        fs=fs, indices_file_paths=to_downloads, dst_folder=local_indices_folder
                    )
                    merged_index = _merge_from_local(merged_index)
                pbar.update(1)

    tmp_stage2 = _get_stage2_folder(small_indices_folder)
    if batch_id is not None:
        _save_small_index(index=merged_index, batch_id=batch_id, small_indices_folder=tmp_stage2)
    return merged_index


def _get_file_system(path: str) -> fsspec.AbstractFileSystem:
    return fsspec.core.url_to_fs(path)[0]


def run(
    faiss_index: faiss.Index,
    embeddings_file_paths: List[str],
    file_counts: List[int],
    batch_size: int,
    embedding_column_name: str = "embedding",
    num_cores_per_executor: Optional[int] = None,
    temporary_indices_folder="hdfs://root/tmp/distributed_autofaiss_indices",
    file_format: str = "npy",
    id_columns: Optional[List[str]] = None,
    embedding_ids_df_handler: Optional[Callable[[pd.DataFrame, int], Any]] = None,
) -> faiss.Index:
    """
    Create indices by pyspark.

    Parameters
    ----------
    faiss_index: faiss.Index
        Trained faiss index
    embeddings_file_paths: List[str]
        List of embeddings file in numpy or parquet format.
    file_counts: List[str]
        Number of lines for each file
    batch_size: int
        Number of vectors handled per worker
    embedding_column_name: str
        Embeddings column name for parquet; default "embedding"
    num_cores_per_executor: int
        Number of CPU cores per executor
    temporary_indices_folder: str
        Folder to save the temporary small indices
    file_format: str
        Embeddings file format; default "npy"
        "npy" or "parquet"
    id_columns: Optional[List[str]]
        Names of the columns containing the Ids of the vectors, only used when file_format is "parquet"
    embedding_ids_df_handler: Optional[Callable[[pd.DataFrame, int], Any]]
        The function that handles the embeddings Ids when id_columns is given
    """
    temporary_indices_folder = make_path_absolute(temporary_indices_folder)

    ss = _get_pyspark_active_session()
    # broadcast the index bytes
    trained_index_bytes = _get_bytes_from_index(faiss_index)
    broadcast_trained_index_bytes = ss.sparkContext.broadcast(trained_index_bytes)
    nb_vectors = sum(file_counts)
    nb_batches = math.ceil(nb_vectors / batch_size)  # use math.ceil to make sure that we cover every vector
    batches = _batch_loader(batch_size=batch_size, nb_batches=nb_batches)
    rdd = ss.sparkContext.parallelize(batches, nb_batches)
    file_system = _get_file_system(embeddings_file_paths[0])
    with Timeit("-> Adding indices", indent=2):
        rdd.foreach(
            lambda x: _add_index(
                batch_id=x[0],
                start=x[1],
                end=x[2],
                broadcast_trained_index_bytes=broadcast_trained_index_bytes,
                embeddings_file_paths=embeddings_file_paths,
                file_system=file_system,
                chunk_sizes=file_counts,
                embedding_column_name=embedding_column_name,
                id_columns=id_columns,
                small_indices_folder=temporary_indices_folder,
                num_cores=num_cores_per_executor,
                embedding_ids_df_handler=embedding_ids_df_handler,
                file_format=file_format,
            )
        )

    with Timeit("-> Merging indices", indent=2):
        fs = _get_file_system(temporary_indices_folder)
        small_indices_files = fs.ls(temporary_indices_folder, detail=False)
        batch_size = 100
        nb_batches = math.ceil(len(small_indices_files) / batch_size)
        merge_batches = _batch_loader(batch_size=batch_size, nb_batches=nb_batches)
        rdd = ss.sparkContext.parallelize(merge_batches, nb_batches)
        # Merge indices in two stages
        # stage1: each executor merges a batch of indices and saves the merged index to stage2 folder
        rdd.foreach(
            lambda x: _merge_index(small_indices_folder=temporary_indices_folder, batch_id=x[0], start=x[1], end=x[2],)  # type: ignore
        )
        # stage2: driver merges the indices generated from stage1
        merged_index = _merge_index(_get_stage2_folder(temporary_indices_folder))
        fs.rm(temporary_indices_folder, recursive=True)

    return merged_index
