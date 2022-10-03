"""
Building the index with pyspark.
"""

import math
import multiprocessing
import os
import logging
from tempfile import TemporaryDirectory
import tempfile
from typing import Dict, Optional, Iterator, Tuple, Callable, Any, Union, List
from functools import partial
from multiprocessing.pool import ThreadPool

import faiss
import fsspec
import pandas as pd
from embedding_reader import EmbeddingReader
from tqdm import tqdm

from autofaiss.external.metadata import IndexMetadata
from autofaiss.external.optimize import get_optimal_batch_size
from autofaiss.indices.build import get_write_ids_df_to_parquet_fn, get_optimize_index_fn, add_embeddings_to_index_local
from autofaiss.indices.index_utils import (
    get_index_from_bytes,
    get_bytes_from_index,
    parallel_download_indices_from_remote,
    load_index,
    save_index,
)
from autofaiss.utils.path import make_path_absolute, extract_partition_name_from_path
from autofaiss.utils.cast import cast_memory_to_bytes, cast_bytes_to_memory_string
from autofaiss.utils.decorators import Timeit
from autofaiss.indices.training import create_and_train_index_from_embedding_dir, TrainedIndex


logger = logging.getLogger("autofaiss")


def _generate_suffix(batch_id: int, nb_batches: int) -> str:
    suffix_width = int(math.log10(nb_batches)) + 1
    return str(batch_id).zfill(suffix_width)


def _generate_small_index_file_name(batch_id: int, nb_batches: int) -> str:
    return "index_" + _generate_suffix(batch_id, nb_batches)


def _add_index(
    start: int,
    end: int,
    broadcasted_trained_index_or_path,
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
    broadcasted_trained_index_or_path: pyspark.Broadcast or str
        Broadcasted trained index or path to a trained index
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

    with tempfile.TemporaryDirectory() as tmp_dir:
        # load empty trained index
        if isinstance(broadcasted_trained_index_or_path, str):
            local_index_path = os.path.join(tmp_dir, "index")
            empty_index = load_index(broadcasted_trained_index_or_path, local_index_path)
        else:
            empty_index = get_index_from_bytes(broadcasted_trained_index_or_path.value)

        batch_size = get_optimal_batch_size(embedding_reader.dimension, memory_available_for_adding)

        ids_total = []
        for (vec_batch, ids_batch) in embedding_reader(batch_size=batch_size, start=start, end=end):
            consecutive_ids = ids_batch["i"].to_numpy()
            # using add_with_ids makes it possible to have consecutive and unique ids over all the N indices
            empty_index.add_with_ids(vec_batch, consecutive_ids)
            if embedding_ids_df_handler:
                ids_total.append(ids_batch)

        if embedding_ids_df_handler:
            embedding_ids_df_handler(pd.concat(ids_total), batch_id)

        save_index(empty_index, small_indices_folder, _generate_small_index_file_name(batch_id, nb_batches))


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


def _batch_loader(nb_batches: int, total_size: int) -> Iterator[Tuple[int, int, int]]:
    """Yield [batch id, batch start position, batch end position (excluded)]"""
    # Thanks to https://stackoverflow.com/a/2135920
    batch_size, mod = divmod(total_size, nb_batches)
    for batch_id in range(nb_batches):
        start = batch_size * batch_id + min(batch_id, mod)
        end = batch_size * (batch_id + 1) + min(batch_id + 1, mod)
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
                faiss.merge_into(merged, index, shift_ids=False)
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
            save_index(merged_index, tmp_output_folder, _generate_small_index_file_name(batch_id, nb_batches))
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

    merge_batches = _batch_loader(nb_batches=n, total_size=nb_indices_on_src_folder)

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


def add_embeddings_to_index_distributed(
    trained_index_or_path: Union[faiss.Index, str],
    embedding_reader: EmbeddingReader,
    memory_available_for_adding: str,
    nb_cores: Optional[int] = None,
    temporary_indices_folder="hdfs://root/tmp/distributed_autofaiss_indices",
    embedding_ids_df_handler: Optional[Callable[[pd.DataFrame, int], Any]] = None,
    nb_indices_to_keep: int = 1,
    index_optimizer: Optional[Callable] = None,
) -> Tuple[Optional[faiss.Index], Optional[Dict[str, str]]]:
    """
    Create indices by pyspark.

    Parameters
    ----------
    trained_index_or_path: trained faiss.Index or path to a trained faiss index
        Trained faiss index
    embedding_reader: EmbeddingReader
        Embedding reader.
    memory_available_for_adding: str
        Memory available for adding embeddings.
    nb_cores: int
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

    # Broadcast index
    broadcasted_trained_index_or_path = (
        trained_index_or_path
        if isinstance(trained_index_or_path, str)
        else ss.sparkContext.broadcast(get_bytes_from_index(trained_index_or_path))
    )

    sc = ss._jsc.sc()  # pylint: disable=protected-access
    n_workers = len(sc.statusTracker().getExecutorInfos()) - 1

    # maximum between the number of spark workers, 10M embeddings per task and the number of indices to keep
    n_batches = min(
        embedding_reader.count, max(n_workers, math.ceil(embedding_reader.count / (10 ** 7)), nb_indices_to_keep)
    )
    nb_indices_to_keep = min(nb_indices_to_keep, n_batches)
    batches = _batch_loader(total_size=embedding_reader.count, nb_batches=n_batches)
    rdd = ss.sparkContext.parallelize(batches, n_batches)
    with Timeit("-> Adding indices", indent=2):
        rdd.foreach(
            lambda x: _add_index(
                batch_id=x[0],
                start=x[1],
                end=x[2],
                memory_available_for_adding=memory_available_for_adding,
                broadcasted_trained_index_or_path=broadcasted_trained_index_or_path,
                embedding_reader=embedding_reader,
                small_indices_folder=stage1_folder,
                num_cores=nb_cores,
                embedding_ids_df_handler=embedding_ids_df_handler,
                nb_batches=n_batches,
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


def _add_embeddings_to_index(
    add_embeddings_fn: Callable,
    embedding_reader: EmbeddingReader,
    output_root_dir: str,
    index_key: str,
    current_memory_available: str,
    id_columns: Optional[List[str]],
    max_index_query_time_ms: float,
    min_nearest_neighbors_to_retrieve: int,
    use_gpu: bool,
    make_direct_map: bool,
) -> Tuple[Optional[faiss.Index], Optional[Dict[str, str]]]:
    """Add embeddings to index"""

    # Define output folders
    partition = extract_partition_name_from_path(embedding_reader.embeddings_folder)
    output_dir = os.path.join(output_root_dir, partition)
    index_dest_path = os.path.join(output_dir, "knn.index")
    ids_dest_dir = os.path.join(output_dir, "ids")
    index_infos_dest_path = os.path.join(output_dir, "index_infos.json")

    # Compute memory available for adding embeddings to index
    metadata = IndexMetadata(index_key, embedding_reader.count, embedding_reader.dimension, make_direct_map)
    index_size = metadata.estimated_index_size_in_bytes()
    memory_available_for_adding = cast_bytes_to_memory_string(
        cast_memory_to_bytes(current_memory_available) - index_size
    )

    write_ids_df_to_parquet_fn = get_write_ids_df_to_parquet_fn(ids_root_dir=ids_dest_dir) if id_columns else None

    optimize_index_fn = get_optimize_index_fn(
        embedding_reader=embedding_reader,
        index_key=index_key,
        index_path=index_dest_path,
        index_infos_path=index_infos_dest_path,
        use_gpu=use_gpu,
        save_on_disk=True,
        max_index_query_time_ms=max_index_query_time_ms,
        min_nearest_neighbors_to_retrieve=min_nearest_neighbors_to_retrieve,
        make_direct_map=make_direct_map,
        index_param=None,
    )

    # Add embeddings to index
    return add_embeddings_fn(
        embedding_reader=embedding_reader,
        memory_available_for_adding=memory_available_for_adding,
        embedding_ids_df_handler=write_ids_df_to_parquet_fn,
        index_optimizer=optimize_index_fn,
    )


def _add_embeddings_from_dir_to_index(
    add_embeddings_fn: Callable,
    embedding_root_dir: str,
    output_root_dir: str,
    index_key: str,
    embedding_column_name: str,
    current_memory_available: str,
    id_columns: Optional[List[str]],
    max_index_query_time_ms: float,
    min_nearest_neighbors_to_retrieve: int,
    use_gpu: bool,
    make_direct_map: bool,
) -> Tuple[Optional[faiss.Index], Optional[Dict[str, str]]]:
    """Add embeddings from directory to index"""

    # Read embeddings
    with Timeit("-> Reading embeddings", indent=2):
        embedding_reader = EmbeddingReader(
            embedding_root_dir, file_format="parquet", embedding_column=embedding_column_name, meta_columns=id_columns
        )

    # Add embeddings to index
    return _add_embeddings_to_index(
        add_embeddings_fn=add_embeddings_fn,
        embedding_reader=embedding_reader,
        output_root_dir=output_root_dir,
        index_key=index_key,
        current_memory_available=current_memory_available,
        id_columns=id_columns,
        max_index_query_time_ms=max_index_query_time_ms,
        min_nearest_neighbors_to_retrieve=min_nearest_neighbors_to_retrieve,
        use_gpu=use_gpu,
        make_direct_map=make_direct_map,
    )


def create_big_index(
    embedding_root_dir: str,
    ss,
    output_root_dir: str,
    id_columns: Optional[List[str]],
    should_be_memory_mappable: bool,
    max_index_query_time_ms: float,
    max_index_memory_usage: str,
    min_nearest_neighbors_to_retrieve: int,
    embedding_column_name: str,
    index_key: str,
    current_memory_available: str,
    nb_cores: Optional[int],
    use_gpu: bool,
    metric_type: str,
    nb_splits_per_big_index: int,
    make_direct_map: bool,
    temp_root_dir: str,
) -> Optional[Dict[str, str]]:
    """
    Create a big index
    """

    def _create_and_train_index_from_embedding_dir() -> TrainedIndex:
        trained_index = create_and_train_index_from_embedding_dir(
            embedding_root_dir=embedding_root_dir,
            embedding_column_name=embedding_column_name,
            index_key=index_key,
            max_index_memory_usage=max_index_memory_usage,
            make_direct_map=make_direct_map,
            should_be_memory_mappable=should_be_memory_mappable,
            use_gpu=use_gpu,
            metric_type=metric_type,
            nb_cores=nb_cores,
            current_memory_available=current_memory_available,
            id_columns=id_columns,
        )

        index_output_root_dir = os.path.join(temp_root_dir, "training", partition)
        output_index_path = save_index(trained_index.index_or_path, index_output_root_dir, "trained_index")
        return TrainedIndex(output_index_path, trained_index.index_key, embedding_root_dir)

    partition = extract_partition_name_from_path(embedding_root_dir)

    # Train index
    rdd = ss.sparkContext.parallelize([embedding_root_dir], 1)
    trained_index_path, trained_index_key, _, = rdd.map(
        lambda _: _create_and_train_index_from_embedding_dir()
    ).collect()[0]

    # Add embeddings to index and compute metrics
    partition_temp_root_dir = os.path.join(temp_root_dir, "add_embeddings", partition)
    index, metrics = _add_embeddings_from_dir_to_index(
        add_embeddings_fn=partial(
            add_embeddings_to_index_distributed,
            trained_index_or_path=trained_index_path,
            nb_cores=nb_cores,
            temporary_indices_folder=partition_temp_root_dir,
            nb_indices_to_keep=nb_splits_per_big_index,
        ),
        embedding_root_dir=embedding_root_dir,
        output_root_dir=output_root_dir,
        index_key=trained_index_key,
        embedding_column_name=embedding_column_name,
        current_memory_available=current_memory_available,
        id_columns=id_columns,
        max_index_query_time_ms=max_index_query_time_ms,
        min_nearest_neighbors_to_retrieve=min_nearest_neighbors_to_retrieve,
        use_gpu=use_gpu,
        make_direct_map=make_direct_map,
    )

    # Only metrics are returned to save memory on driver
    if index:
        del index

    return metrics


def create_small_index(
    embedding_root_dir: str,
    output_root_dir: str,
    id_columns: Optional[List[str]] = None,
    should_be_memory_mappable: bool = False,
    max_index_query_time_ms: float = 10.0,
    max_index_memory_usage: str = "16G",
    min_nearest_neighbors_to_retrieve: int = 20,
    embedding_column_name: str = "embedding",
    index_key: Optional[str] = None,
    current_memory_available: str = "32G",
    use_gpu: bool = False,
    metric_type: str = "ip",
    nb_cores: Optional[int] = None,
    make_direct_map: bool = False,
) -> Tuple[Optional[faiss.Index], Optional[Dict[str, str]]]:
    """
    Create a small index
    """

    # Train index
    trained_index = create_and_train_index_from_embedding_dir(
        embedding_root_dir=embedding_root_dir,
        embedding_column_name=embedding_column_name,
        index_key=index_key,
        max_index_memory_usage=max_index_memory_usage,
        make_direct_map=make_direct_map,
        should_be_memory_mappable=should_be_memory_mappable,
        use_gpu=use_gpu,
        metric_type=metric_type,
        nb_cores=nb_cores,
        current_memory_available=current_memory_available,
        id_columns=id_columns,
    )

    # Add embeddings to index and compute metrics
    return _add_embeddings_to_index(
        add_embeddings_fn=partial(
            add_embeddings_to_index_local,
            trained_index_or_path=trained_index.index_or_path,
            add_embeddings_with_ids=True,
        ),
        embedding_reader=trained_index.embedding_reader_or_path,
        output_root_dir=output_root_dir,
        index_key=trained_index.index_key,
        current_memory_available=current_memory_available,
        id_columns=id_columns,
        max_index_query_time_ms=max_index_query_time_ms,
        min_nearest_neighbors_to_retrieve=min_nearest_neighbors_to_retrieve,
        use_gpu=use_gpu,
        make_direct_map=make_direct_map,
    )


def create_partitioned_indexes(
    partitions: List[str],
    big_index_threshold: int,
    output_root_dir: str,
    nb_cores: Optional[int],
    nb_splits_per_big_index: int,
    id_columns: Optional[List[str]] = None,
    max_index_query_time_ms: float = 10.0,
    min_nearest_neighbors_to_retrieve: int = 20,
    embedding_column_name: str = "embedding",
    index_key: Optional[str] = None,
    max_index_memory_usage: str = "16G",
    current_memory_available: str = "32G",
    use_gpu: bool = False,
    metric_type: str = "ip",
    make_direct_map: bool = False,
    should_be_memory_mappable: bool = False,
    temp_root_dir: str = "hdfs://root/tmp/distributed_autofaiss_indices",
    maximum_nb_threads: int = 256,
) -> List[Optional[Dict[str, str]]]:
    """
    Create partitioned indexes from a list of parquet partitions,
    i.e. create and train one index per parquet partition
    """

    def _create_small_indexes(embedding_root_dirs: List[str]) -> List[Optional[Dict[str, str]]]:
        rdd = ss.sparkContext.parallelize(embedding_root_dirs, len(embedding_root_dirs))
        return rdd.map(
            lambda embedding_root_dir: create_small_index(
                embedding_root_dir=embedding_root_dir,
                output_root_dir=output_root_dir,
                id_columns=id_columns,
                should_be_memory_mappable=should_be_memory_mappable,
                max_index_query_time_ms=max_index_query_time_ms,
                max_index_memory_usage=max_index_memory_usage,
                min_nearest_neighbors_to_retrieve=min_nearest_neighbors_to_retrieve,
                embedding_column_name=embedding_column_name,
                index_key=index_key,
                current_memory_available=current_memory_available,
                use_gpu=use_gpu,
                metric_type=metric_type,
                nb_cores=nb_cores,
                make_direct_map=make_direct_map,
            )[1]
        ).collect()

    ss = _get_pyspark_active_session()

    create_big_index_fn = partial(
        create_big_index,
        ss=ss,
        output_root_dir=output_root_dir,
        id_columns=id_columns,
        should_be_memory_mappable=should_be_memory_mappable,
        max_index_query_time_ms=max_index_query_time_ms,
        max_index_memory_usage=max_index_memory_usage,
        min_nearest_neighbors_to_retrieve=min_nearest_neighbors_to_retrieve,
        embedding_column_name=embedding_column_name,
        index_key=index_key,
        current_memory_available=current_memory_available,
        nb_cores=nb_cores,
        use_gpu=use_gpu,
        metric_type=metric_type,
        nb_splits_per_big_index=nb_splits_per_big_index,
        make_direct_map=make_direct_map,
        temp_root_dir=temp_root_dir,
    )

    # Compute number of embeddings for each partition
    rdd = ss.sparkContext.parallelize(partitions, len(partitions))
    partition_sizes = rdd.map(
        lambda partition: (
            partition,
            EmbeddingReader(partition, file_format="parquet", embedding_column=embedding_column_name).count,
        )
    ).collect()

    # Group partitions in two categories, small and big indexes
    small_partitions = []
    big_partitions = []
    for partition, size in partition_sizes:
        if size < big_index_threshold:
            small_partitions.append(partition)
        else:
            big_partitions.append(partition)

    # Create small and big indexes
    all_metrics = []
    n_threads = min(maximum_nb_threads, len(big_partitions) + int(len(small_partitions) > 0))
    with ThreadPool(n_threads) as p:
        small_index_metrics_future = (
            p.apply_async(_create_small_indexes, (small_partitions,)) if small_partitions else None
        )
        for metrics in p.starmap(create_big_index_fn, [(p,) for p in big_partitions]):
            all_metrics.append(metrics)
        if small_index_metrics_future:
            all_metrics.extend(small_index_metrics_future.get())

    return all_metrics
