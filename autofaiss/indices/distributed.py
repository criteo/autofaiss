"""
Building the index with pyspark.
"""
import math
import multiprocessing
import os
import logging
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Iterator, Tuple, Callable, Any

import faiss
import fsspec
import pandas as pd
from embedding_reader import EmbeddingReader
from tqdm import tqdm

from autofaiss.external.optimize import get_optimal_batch_size
from autofaiss.indices.index_utils import (
    get_index_from_bytes,
    get_bytes_from_index,
    parallel_download_indices_from_remote,
)
from autofaiss.utils.path import make_path_absolute
from autofaiss.utils.cast import cast_memory_to_bytes
from autofaiss.utils.decorators import Timeit

logger = logging.getLogger("autofaiss")


def _generate_suffix(batch_id: int, nb_batches: int) -> str:
    suffix_width = int(math.log10(nb_batches)) + 1
    return str(batch_id).zfill(suffix_width)


def _generate_small_index_file_name(batch_id: int, nb_batches: int) -> str:
    return "index_" + _generate_suffix(batch_id, nb_batches)


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
    memory_available_for_adding: str,
    embedding_reader: EmbeddingReader,
    batch_id: int,
    small_indices_folder: str,
    nb_batches: int,
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
    memory_available_for_adding: str
        Memory available for adding embeddings
    embedding_reader: EmbeddingReader
        Embedding reader
    batch_id: int
        Batch id
    small_indices_folder: str
        The folder where we save all the small indices
    num_cores: int
        Number of CPU cores (not Vcores)
    embedding_ids_df_handler: Optional[Callable[[pd.DataFrame, int], Any]]
        The function that handles the embeddings Ids when id_columns is given
    """
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()

    faiss.omp_set_num_threads(num_cores)

    # load empty trained index
    empty_index = get_index_from_bytes(broadcast_trained_index_bytes.value)

    batch_size = get_optimal_batch_size(embedding_reader.dimension, memory_available_for_adding)

    ids_total = []
    for (vec_batch, ids_batch) in embedding_reader(batch_size=batch_size, start=start, end=end):
        empty_index.add(vec_batch)
        if embedding_ids_df_handler:
            ids_total.append(ids_batch)

    if embedding_ids_df_handler:
        embedding_ids_df_handler(pd.concat(ids_total), batch_id)

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


def _merge_index(
    small_indices_folder: str,
    nb_batches: int,
    batch_id: Optional[int] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    max_size_on_disk: str = "50GB",
    tmp_output_folder: Optional[str] = None,
    index_optimizer: Callable = None,
) -> Tuple[faiss.Index, Dict[str, str]]:
    """
    Merge all the indices in `small_indices_folder` into single one.
    Also run optimization when `index_optimizer` is given.
    Returns the merged index and the metric
    """
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
    nb_files_each_time = max(1, int(max_sizes_in_bytes / first_index_size))
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
        if index_optimizer is not None:
            metric_infos = index_optimizer(merged_index, index_suffix=_generate_suffix(batch_id, nb_batches))
        else:
            metric_infos = None
            _save_small_index(
                index=merged_index, batch_id=batch_id, small_indices_folder=tmp_output_folder, nb_batches=nb_batches
            )
    else:
        metric_infos = None
    return merged_index, metric_infos


def _get_file_system(path: str) -> fsspec.AbstractFileSystem:
    return fsspec.core.url_to_fs(path, use_listings_cache=False)[0]


def _merge_to_n_indices(spark_session, n: int, src_folder: str, dst_folder: str, index_optimizer: Callable = None):
    """Merge all the indices from src_folder into n indices, and return the folder for the next stage, as well as the metrics"""
    fs = _get_file_system(src_folder)
    nb_indices_on_src_folder = len(fs.ls(src_folder, detail=False))

    if nb_indices_on_src_folder <= n and index_optimizer is None:
        # no need to merge
        return src_folder, None

    batch_size = max(1, int(nb_indices_on_src_folder / n))
    n = math.ceil(nb_indices_on_src_folder / batch_size)
    merge_batches = _batch_loader(batch_size=batch_size, nb_batches=n)

    rdd = spark_session.sparkContext.parallelize(merge_batches, n)

    def merge(x):
        _, metrics = _merge_index(
            small_indices_folder=src_folder,
            nb_batches=n,
            batch_id=x[0],
            start=x[1],
            end=x[2],
            tmp_output_folder=dst_folder,
            index_optimizer=index_optimizer,
        )  # type: ignore
        return metrics

    metrics_rdd = rdd.map(merge)
    metrics = list(metrics_rdd.collect())
    if index_optimizer is not None:
        metrics_dict = {metric_info["index_path"]: metric_info for metric_info in metrics}  # type: ignore
    else:
        metrics_dict = None  # type: ignore
    for file in fs.ls(src_folder, detail=False):
        if fs.isfile(file):
            fs.rm(file)
    return dst_folder, metrics_dict


def run(
    faiss_index: faiss.Index,
    embedding_reader: EmbeddingReader,
    memory_available_for_adding: str,
    num_cores_per_executor: Optional[int] = None,
    temporary_indices_folder="hdfs://root/tmp/distributed_autofaiss_indices",
    embedding_ids_df_handler: Optional[Callable[[pd.DataFrame, int], Any]] = None,
    nb_indices_to_keep: int = 1,
    index_optimizer: Optional[Callable] = None,
) -> Tuple[Optional[faiss.Index], Optional[Dict[str, str]]]:
    """
    Create indices by pyspark.

    Parameters
    ----------
    faiss_index: faiss.Index
        Trained faiss index
    embedding_reader: EmbeddingReader
        Embedding reader.
    memory_available_for_adding: str
        Memory available for adding embeddings.
    num_cores_per_executor: int
        Number of CPU cores per executor
    temporary_indices_folder: str
        Folder to save the temporary small indices
    embedding_ids_df_handler: Optional[Callable[[pd.DataFrame, int], Any]]
        The function that handles the embeddings Ids when id_columns is given
    nb_indices_to_keep: int
        Number of indices to keep at most after the merging step
    index_optimizer: Optional[Callable]
        The function that optimizes the index
    """
    temporary_indices_folder = make_path_absolute(temporary_indices_folder)
    fs = _get_file_system(temporary_indices_folder)
    if fs.exists(temporary_indices_folder):
        fs.rm(temporary_indices_folder, recursive=True)
    stage1_folder = temporary_indices_folder.rstrip("/") + "/stage-1"
    ss = _get_pyspark_active_session()
    # broadcast the index bytes
    trained_index_bytes = get_bytes_from_index(faiss_index)
    broadcast_trained_index_bytes = ss.sparkContext.broadcast(trained_index_bytes)
    sc = ss._jsc.sc()  # pylint: disable=protected-access
    n_workers = len(sc.statusTracker().getExecutorInfos()) - 1

    # maximum between the number of spark workers, 100M embeddings per task and the number of indices to keep
    estimated_nb_batches = max(n_workers, int(embedding_reader.count / (10 ** 7)), nb_indices_to_keep)
    batch_size = max(1, int(embedding_reader.count / estimated_nb_batches))
    nb_batches = math.ceil(embedding_reader.count / batch_size)

    batches = _batch_loader(batch_size=batch_size, nb_batches=nb_batches)
    rdd = ss.sparkContext.parallelize(batches, nb_batches)
    with Timeit("-> Adding indices", indent=2):
        rdd.foreach(
            lambda x: _add_index(
                batch_id=x[0],
                start=x[1],
                end=x[2],
                memory_available_for_adding=memory_available_for_adding,
                broadcast_trained_index_bytes=broadcast_trained_index_bytes,
                embedding_reader=embedding_reader,
                small_indices_folder=stage1_folder,
                num_cores=num_cores_per_executor,
                embedding_ids_df_handler=embedding_ids_df_handler,
                nb_batches=nb_batches,
            )
        )

    with Timeit("-> Merging indices", indent=2):
        stage2_folder = temporary_indices_folder.rstrip("/") + "/stage-2"
        next_stage_folder, _ = _merge_to_n_indices(
            spark_session=ss, n=100, src_folder=stage1_folder, dst_folder=stage2_folder, index_optimizer=None
        )
        if nb_indices_to_keep == 1:
            merged_index, _ = _merge_index(small_indices_folder=next_stage_folder, nb_batches=1)
            if fs.exists(temporary_indices_folder):
                fs.rm(temporary_indices_folder, recursive=True)
            metrics = index_optimizer(merged_index, "")  # type: ignore
            return merged_index, metrics
        else:
            final_folder = temporary_indices_folder.rstrip("/") + "/final"
            next_stage_folder, metrics = _merge_to_n_indices(
                spark_session=ss,
                n=nb_indices_to_keep,
                src_folder=next_stage_folder,
                dst_folder=final_folder,
                index_optimizer=index_optimizer,
            )
            if fs.exists(temporary_indices_folder):
                fs.rm(temporary_indices_folder, recursive=True)
            return None, metrics
