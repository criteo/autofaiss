"""
Building the index with pyspark.
"""
import math
import multiprocessing
import os
import logging
from tempfile import TemporaryDirectory
from typing import List, Optional, Iterator, Tuple, Callable, Any

import faiss
import fsspec
import numpy as np
import pandas as pd
from fsspec import get_filesystem_class
from fsspec.implementations.hdfs import PyArrowHDFS
from tqdm import tqdm

from autofaiss.indices.index_utils import (
    get_index_from_bytes,
    get_bytes_from_index,
    parallel_download_indices_from_remote,
)
from autofaiss.readers.embeddings_iterators import get_matrix_reader, make_path_absolute
from autofaiss.utils.cast import cast_memory_to_bytes
from autofaiss.utils.decorators import Timeit

logger = logging.getLogger("autofaiss")


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
        if cur_end <= start:
            cur_start += chunk_size
            continue
        slice_start = max(0, start - cur_start)
        slice_end = min(chunk_size, end - cur_start)
        with get_matrix_reader(file_format, file_system, file_path, embedding_column_name, id_columns) as matrix_reader:
            yield matrix_reader.get_lazy_array().get_rows(start=slice_start, end=slice_end)
        if cur_end >= end:
            break
        cur_start += chunk_size


def _generate_small_index_file_name(batch_id: int, nb_batches: int) -> str:
    suffix_width = int(math.log10(nb_batches)) + 1
    return "index_" + str(batch_id).zfill(suffix_width)


def _save_small_index(index: faiss.Index, batch_id: int, small_indices_folder: str, nb_batches: int) -> None:
    """Save index for one batch."""
    fs = _get_file_system(small_indices_folder)
    fs.mkdirs(small_indices_folder, exist_ok=True)
    small_index_filename = _generate_small_index_file_name(batch_id, nb_batches)
    dest_filepath = os.path.join(small_indices_folder, small_index_filename)
    with fsspec.open(dest_filepath, "wb").open() as f:
        faiss.write_index(index, faiss.PyCallbackIOWriter(f.write))


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
    nb_batches: int,
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
    empty_index = get_index_from_bytes(broadcast_trained_index_bytes.value)

    empty_index.add(embeddings_to_add)

    del embeddings_to_add

    _save_small_index(
        index=empty_index, small_indices_folder=small_indices_folder, batch_id=batch_id, nb_batches=nb_batches
    )


def _get_pyspark_active_session():
    """Reproduce SparkSession.getActiveSession() available since pyspark 3.0."""
    import pyspark  # pylint: disable=import-outside-toplevel

    # pylint: disable=protected-access
    ss: Optional[pyspark.sql.SparkSession] = pyspark.sql.SparkSession._instantiatedSession  # mypy: ignore
    if ss is None:
        logger.info("No pyspark session found, creating a new one!")
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


def _get_stage2_folder(tmp_folder: str) -> str:
    """Get the temporary folder path used for the 2nd stage of merging"""
    return tmp_folder.rstrip("/") + "/stage-2/"


def _merge_index(
    small_indices_folder: str,
    nb_batches: int,
    batch_id: Optional[int] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    max_size_on_disk: str = "100GB",
    tmp_output_folder: Optional[str] = None,
) -> faiss.Index:
    """Merge all the indices in `small_indices_folder` into single one return the merged index."""
    fs = _get_file_system(small_indices_folder)
    small_indices_files = sorted(fs.ls(small_indices_folder, detail=False))
    small_indices_files = small_indices_files[start:end]

    if len(small_indices_files) == 0:
        raise ValueError(f"No small index is saved in {small_indices_folder}")

    def _merge_from_local(merged: Optional[faiss.Index] = None) -> faiss.Index:
        local_file_paths = [
            os.path.join(local_indices_folder, filename) for filename in sorted(os.listdir(local_indices_folder))
        ]
        if merged is None:
            merged = faiss.read_index(local_file_paths[0])
            start_index = 1
        else:
            start_index = 0

        for rest_index_file in tqdm(local_file_paths[start_index:]):
            # if master and executor are the same machine, rest_index_file could be the folder for stage2
            # so, we have to check whether it is file or not
            if os.path.isfile(rest_index_file):
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
                    parallel_download_indices_from_remote(
                        fs=fs, indices_file_paths=to_downloads, dst_folder=local_indices_folder
                    )
                    merged_index = _merge_from_local(merged_index)
                pbar.update(1)

    if batch_id is not None and tmp_output_folder is not None:
        _save_small_index(
            index=merged_index, batch_id=batch_id, small_indices_folder=tmp_output_folder, nb_batches=nb_batches
        )
    return merged_index


def _get_file_system(path: str) -> fsspec.AbstractFileSystem:
    return fsspec.core.url_to_fs(path)[0]


def _merge_to_n_indices(spark_session, n: int, src_folder: str):
    """Merge all the indices from src_folder into n indices, and return the folder for the next stage"""
    fs = _get_file_system(src_folder)
    nb_indices_on_src_folder = len(fs.ls(src_folder, detail=False))
    if nb_indices_on_src_folder <= n:
        # merging does not happen, return the source folder
        return src_folder
    dst_folder = _get_stage2_folder(src_folder)
    batch_size = math.ceil(nb_indices_on_src_folder / n)
    merge_batches = _batch_loader(batch_size=batch_size, nb_batches=n)
    rdd = spark_session.sparkContext.parallelize(merge_batches, n)
    rdd.foreach(
        lambda x: _merge_index(
            small_indices_folder=src_folder,
            nb_batches=n,
            batch_id=x[0],
            start=x[1],
            end=x[2],
            tmp_output_folder=dst_folder,
        )  # type: ignore
    )
    fs = _get_file_system(src_folder)
    for file in fs.ls(src_folder, detail=False):
        if fs.isfile(file):
            fs.rm(file)
    return dst_folder


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
    nb_indices_to_keep: int = 1,
) -> Tuple[Optional[faiss.Index], Optional[str]]:
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
    nb_indices_to_keep: int
        Number of indices to keep at most after the merging step
    """
    temporary_indices_folder = make_path_absolute(temporary_indices_folder)
    fs = _get_file_system(temporary_indices_folder)
    if fs.exists(temporary_indices_folder):
        fs.rm(temporary_indices_folder, recursive=True)

    ss = _get_pyspark_active_session()
    # broadcast the index bytes
    trained_index_bytes = get_bytes_from_index(faiss_index)
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
                nb_batches=nb_batches,
            )
        )

    with Timeit("-> Merging indices", indent=2):
        next_stage_folder = _merge_to_n_indices(spark_session=ss, n=100, src_folder=temporary_indices_folder,)
        if nb_indices_to_keep == 1:
            merged_index = _merge_index(small_indices_folder=next_stage_folder, nb_batches=1)
            fs.rm(temporary_indices_folder, recursive=True)
            return merged_index, None
        else:
            next_stage_folder = _merge_to_n_indices(
                spark_session=ss, n=nb_indices_to_keep, src_folder=next_stage_folder,
            )
            return None, next_stage_folder
