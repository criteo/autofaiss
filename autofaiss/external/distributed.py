import math
import multiprocessing
import os
import shutil
from itertools import repeat
from multiprocessing.pool import ThreadPool
from tempfile import NamedTemporaryFile
from typing import List, Optional, Union, Iterator, Tuple

import faiss
import fsspec
import numpy as np
from tqdm import tqdm

from autofaiss.readers.embeddings_iterators import get_file_list, ParquetLazyNdArray, ParquetMatrixReader
from autofaiss.utils.decorators import Timeit


def _yield_embeddings_batch(
    embeddings_paths: List[str],
    chunk_sizes: List[int],
    start: int,
    end: int,
    embedding_column_name: str
):
    """Lazy load a batch of embeddings."""
    cur_start = cur_end = 0
    for chunk_size, file_path in zip(chunk_sizes, embeddings_paths):
        cur_end += chunk_size
        if cur_end < start:
            cur_start += chunk_size
            continue
        parquet_array = ParquetLazyNdArray(f=file_path, embedding_column_name=embedding_column_name)
        slice_start = max(0, start - cur_start)
        slice_end = min(chunk_size, end - cur_start)
        yield from parquet_array.get_rows(start=slice_start, end=slice_end)[0]
        if cur_end > end:
            break
        cur_start += chunk_size


def _get_index_from_bytes(index_bytes: Union[bytearray, bytes]) -> faiss.Index:
    """Transforms a bytearray containing a faiss index into the corresponding object."""

    with NamedTemporaryFile(delete=False) as output_file:
        output_file.write(index_bytes)
        tmp_name = output_file.name

    b = faiss.read_index(tmp_name)
    os.remove(tmp_name)
    return b


def _get_bytes_from_index(index: faiss.Index) -> bytearray:
    """Transforms a faiss index into a bytearray."""

    with NamedTemporaryFile(delete=False) as output_file:
        faiss.write_index(index, output_file.name)
        tmp_name = output_file.name

    with open(tmp_name, "rb") as index_file:
        b = index_file.read()
        os.remove(tmp_name)
        return bytearray(b)


def _generate_small_index_file_name(batch_id: int) -> str:
    return f"index_{batch_id}"


def _save_small_index(
    index: faiss.Index,
    batch_id: int,
    small_indices_folder: str
):
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
    chunk_sizes: List[int],
    embeddings_file_paths: List[str],
    embedding_column_name: str,
    batch_id: int,
    small_indices_folder: str,
    num_cores: Optional[int] = None,
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
    """
    if end > sum(chunk_sizes):
        end = sum(chunk_sizes)
    if len(chunk_sizes) != len(embeddings_file_paths):
        raise ValueError("The length of chunk_sizes should be equal to the number of embeddings_file_paths")

    embeddings_to_add = np.vstack(_yield_embeddings_batch(
        embeddings_paths=embeddings_file_paths,
        chunk_sizes=chunk_sizes,
        start=start,
        end=end,
        embedding_column_name=embedding_column_name
    )).astype(np.float32)  # faiss requires float32 type

    if num_cores is None:
        num_cores = multiprocessing.cpu_count()

    faiss.omp_set_num_threads(num_cores)

    # load empty trained index
    empty_index = _get_index_from_bytes(broadcast_trained_index_bytes.value)

    empty_index.add(embeddings_to_add)

    del embeddings_to_add

    _save_small_index(
        index=empty_index,
        small_indices_folder=small_indices_folder,
        batch_id=batch_id
    )


def _get_pyspark_active_session():
    import pyspark
    """Reproduce SparkSession.getActiveSession() available since pyspark 3.0"""
    # pylint: disable=protected-access
    ss: Optional[
        pyspark.sql.SparkSession
    ] = pyspark.sql.SparkSession._instantiatedSession  # mypy: ignore
    if ss is None:
        print("No pyspark session found, creating a new one!")
        ss = pyspark.sql.SparkSession.builder.config("spark.driver.memory", "16G") \
            .master('local[1]') \
            .appName('Distributed autofaiss') \
            .getOrCreate()
    return ss


def _batch_loader(
    batch_size: int,
    nb_batches: int
) -> Iterator[Tuple[int, int, int]]:
    """Yield [batch id, batch start position, batch end position]"""
    for batch_id in range(nb_batches):
        start = batch_size * batch_id
        end = batch_size * (batch_id + 1)
        yield batch_id, start, end


def _download_one(src_dst_path: Tuple[str, str]):
    src_path, dst_path = src_dst_path
    os.system(f"hdfs dfs -copyToLocal {src_path} {dst_path}")


def _parallel_download_indices_from_hdfs(indices_file_paths: List[str], dst_folder: str):
    """Download small indices in parallel."""
    if len(indices_file_paths) == 0:
        return
    os.makedirs(dst_folder, exist_ok=True)
    nb_files = len(indices_file_paths)
    src_paths = ("hdfs://root" + filepath for filepath in indices_file_paths)
    src_dest_paths = zip(src_paths, repeat(dst_folder))
    with tqdm(total=nb_files) as pbar:
        with multiprocessing.Pool(processes=min(8, multiprocessing.cpu_count())) as pool:
            for _ in pool.imap_unordered(_download_one, src_dest_paths):
                pbar.update(1)


def _merge_index(small_indices_folder: str, ):
    """Merge all the indices in `small_indices_folder` into single one return the merged index."""
    fs = _get_file_system(small_indices_folder)
    small_indices_files = fs.ls(small_indices_folder, detail=False)
    if len(small_indices_files) == 0:
        raise ValueError(f"No small index is saved in {small_indices_folder}")

    def _get_index_from_file(filepath: str):
        with open(filepath, "rb") as f:
            index_bytes = f.read()
        return _get_index_from_bytes(index_bytes)

    def _merge_from_local():
        local_file_paths = [os.path.join(local_indices_folder, filename) for filename in
                            os.listdir(local_indices_folder)]
        with Timeit("-> Load first index", indent=4):
            merged = _get_index_from_file(local_file_paths[0])
        with Timeit("-> Merge the rest of indices", indent=4):
            for rest_index_file in tqdm(local_file_paths[1:]):
                index = _get_index_from_file(rest_index_file)
                faiss.merge_into(merged, index, shift_ids=True)
        return merged

    local_indices_folder = "/tmp/distributed_autofaiss_indices"
    with Timeit("-> Download small indices from HDFS to local", indent=4):
        _parallel_download_indices_from_hdfs(indices_file_paths=small_indices_files, dst_folder=local_indices_folder)
    try:
        merged_index = _merge_from_local()
    finally:
        if os.path.exists(local_indices_folder):
            shutil.rmtree(local_indices_folder)
    return merged_index


def _get_chunk_sizes(embed_paths: List[str], embedding_column_name: str) -> List[int]:
    """Get chunk sizes from a list of embeddings files."""
    def _get_parquet_row_count(embed_path: str) -> int:
        fs = _get_file_system(embed_path)
        return ParquetMatrixReader(fs, embed_path, embedding_column_name).get_row_count()
    chunk_sizes = []
    # use ThreadPool instead of multiprocessing.Pool to avoid having problem in _get_parquet_row_count
    with ThreadPool(50) as pool:
        for row_count in pool.imap(_get_parquet_row_count, embed_paths):
            chunk_sizes.append(row_count)
    return chunk_sizes


def run(
    faiss_index: faiss.Index,
    embeddings_path: str,
    batch_size: int,
    embedding_column_name: str = "embedding",
    num_cores_per_executor: Optional[int] = None,
    small_indices_folder="hdfs://root/tmp/distributed_autofaiss_indices"
) -> faiss.Index:
    """
    Create indices by pyspark.

    Parameters
    ----------
    faiss_index: faiss.Index
        Trained faiss index.
    embeddings_path: str
        Embeddings folder.
    embedding_column_name: str
        Embeddings column name for parquet ; default "embedding"
    batch_size: int
        Number of vectors handled per worker
    num_cores_per_executor: int
        Number of CPU cores per executor
    small_indices_folder: str
        Folder to save the temporary small indices.
    """
    ss = _get_pyspark_active_session()

    # broadcast the index bytes
    trained_index_bytes = _get_bytes_from_index(faiss_index)
    broadcast_trained_index_bytes = ss.sparkContext.broadcast(trained_index_bytes)

    _, filenames = get_file_list(path=embeddings_path, file_format="parquet")

    embed_paths = [os.path.join(embeddings_path, filename) for filename in filenames]

    chunk_sizes = _get_chunk_sizes(embed_paths, embedding_column_name)
    embed_paths = [p.replace("hdfs", "viewfs") for p in embed_paths]

    nb_vectors = sum(chunk_sizes)
    nb_batches = math.ceil(nb_vectors / batch_size)  # use math.ceil to make sure that we cover every vector

    batches = _batch_loader(batch_size=batch_size, nb_batches=nb_batches)
    rdd = ss.sparkContext.parallelize(batches, nb_batches)

    with Timeit("-> Adding indices", indent=2):
        rdd.foreach(lambda x: _add_index(
            batch_id=x[0],
            start=x[1],
            end=x[2],
            broadcast_trained_index_bytes=broadcast_trained_index_bytes,
            embeddings_file_paths=embed_paths,
            chunk_sizes=chunk_sizes,
            embedding_column_name=embedding_column_name,
            small_indices_folder=small_indices_folder,
            num_cores=num_cores_per_executor))

    with Timeit("-> Merging indices", indent=2):
        merged_index = _merge_index(small_indices_folder)

        fs = _get_file_system(small_indices_folder)
        fs.rm(small_indices_folder, recursive=True)

    return merged_index


def _get_file_system(small_indices_folder: str) -> fsspec.AbstractFileSystem:
    return fsspec.core.url_to_fs(small_indices_folder)[0]
