""" main file to create an index from the the begining """

import json
import logging
import logging.config
import multiprocessing
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Union

from embedding_reader import EmbeddingReader

import faiss
import fire
import fsspec
import numpy as np
from autofaiss.indices.build import get_write_ids_df_to_parquet_fn, get_optimize_index_fn
from autofaiss.external.build import (
    create_index,
    create_partitioned_indexes,
    estimate_memory_required_for_index_creation,
    get_estimated_construction_time_infos,
)
from autofaiss.indices.training import create_empty_index
from autofaiss.external.optimize import get_optimal_hyperparameters, get_optimal_index_keys_v2
from autofaiss.external.scores import compute_fast_metrics, compute_medium_metrics
from autofaiss.indices.index_utils import set_search_hyperparameters
from autofaiss.utils.path import make_path_absolute
from autofaiss.utils.cast import cast_bytes_to_memory_string, cast_memory_to_bytes
from autofaiss.utils.decorators import Timeit

logger = logging.getLogger("autofaiss")


def _log_output_dict(infos: Dict):
    logger.info("{")
    for key, value in infos.items():
        logger.info(f"\t{key}: {value}")
    logger.info("}")


def setup_logging(logging_level: int):
    """Setup the logging."""
    logging.config.dictConfig(dict(version=1, disable_existing_loggers=False))
    logging_format = "%(asctime)s [%(levelname)s]: %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)


def build_index(
    embeddings: Union[str, np.ndarray, List[str]],
    index_path: Optional[str] = "knn.index",
    index_infos_path: Optional[str] = "index_infos.json",
    ids_path: Optional[str] = None,
    save_on_disk: bool = True,
    file_format: str = "npy",
    embedding_column_name: str = "embedding",
    id_columns: Optional[List[str]] = None,
    index_key: Optional[str] = None,
    index_param: Optional[str] = None,
    max_index_query_time_ms: float = 10.0,
    max_index_memory_usage: str = "16G",
    min_nearest_neighbors_to_retrieve: int = 20,
    current_memory_available: str = "32G",
    use_gpu: bool = False,
    metric_type: str = "ip",
    nb_cores: Optional[int] = None,
    make_direct_map: bool = False,
    should_be_memory_mappable: bool = False,
    distributed: Optional[str] = None,
    temporary_indices_folder: str = "hdfs://root/tmp/distributed_autofaiss_indices",
    verbose: int = logging.INFO,
    nb_indices_to_keep: int = 1,
) -> Tuple[Optional[faiss.Index], Optional[Dict[str, str]]]:
    """
    Reads embeddings and creates a quantized index from them.
    The index is stored on the current machine at the given output path.

    Parameters
    ----------
    embeddings : Union[str, np.ndarray, List[str]]
        Local path containing all preprocessed vectors and cached files.
        This could be a single directory or multiple directories.
        Files will be added if empty.
        Or directly the Numpy array of embeddings
    index_path: Optional(str)
        Destination path of the quantized model.
    index_infos_path: Optional(str)
        Destination path of the metadata file.
    ids_path: Optional(str)
        Only useful when id_columns is not None and file_format=`parquet`. T
        his will be the path (in any filesystem)
        where the mapping files Ids->vector index will be store in parquet format
    save_on_disk: bool
        Whether to save the index on disk, default to True.
    file_format: Optional(str)
        npy or parquet ; default npy
    embedding_column_name: Optional(str)
        embeddings column name for parquet ; default embedding
    id_columns: Optional(List[str])
        Can only be used when file_format=`parquet`.
        In this case these are the names of the columns containing the Ids of the vectors,
        and separate files will be generated to map these ids to indices in the KNN index ;
        default None
    index_key: Optional(str)
        Optional string to give to the index factory in order to create the index.
        If None, an index is chosen based on an heuristic.
    index_param: Optional(str)
        Optional string with hyperparameters to set to the index.
        If None, the hyper-parameters are chosen based on an heuristic.
    max_index_query_time_ms: float
        Bound on the query time for KNN search, this bound is approximative
    max_index_memory_usage: str
        Maximum size allowed for the index, this bound is strict
    min_nearest_neighbors_to_retrieve: int
        Minimum number of nearest neighbors to retrieve when querying the index.
        Parameter used only during index hyperparameter finetuning step, it is
        not taken into account to select the indexing algorithm.
        This parameter has the priority over the max_index_query_time_ms constraint.
    current_memory_available: str
        Memory available on the machine creating the index, having more memory is a boost
        because it reduces the swipe between RAM and disk.
    use_gpu: bool
        Experimental, gpu training is faster, not tested so far
    metric_type: str
        Similarity function used for query:
            - "ip" for inner product
            - "l2" for euclidian distance
    nb_cores: Optional[int]
        Number of cores to use. Will try to guess the right number if not provided
    make_direct_map: bool
        Create a direct map allowing reconstruction of embeddings. This is only needed for IVF indices.
        Note that might increase the RAM usage (approximately 8GB for 1 billion embeddings)
    should_be_memory_mappable: bool
        If set to true, the created index will be selected only among the indices that can be memory-mapped on disk.
        This makes it possible to use 50GB indices on a machine with only 1GB of RAM. Default to False
    distributed: Optional[str]
        If "pyspark", create the indices using pyspark.
        Only "parquet" file format is supported.
    temporary_indices_folder: str
        Folder to save the temporary small indices that are generated by each spark executor.
        Only used when distributed = "pyspark".
    verbose: int
        set verbosity of outputs via logging level, default is `logging.INFO`
    nb_indices_to_keep: int
        Number of indices to keep at most when distributed is "pyspark".
        It allows you to build an index larger than `current_memory_available`
        If it is not equal to 1,
            - You are expected to have at most `nb_indices_to_keep` indices with the following names:
                "{index_path}i" where i ranges from 1 to `nb_indices_to_keep`
            - `build_index` returns a mapping from index path to metrics
        Default to 1.
    """
    setup_logging(verbose)
    # if using distributed mode, it doesn't make sense to use indices that are not memory mappable
    if distributed == "pyspark":
        should_be_memory_mappable = True
    if index_path:
        index_path = make_path_absolute(index_path)
    elif save_on_disk:
        logger.error("Please specify a index_path if you set save_on_disk as True")
        return None, None
    if index_infos_path:
        index_infos_path = make_path_absolute(index_infos_path)
    elif save_on_disk:
        logger.error("Please specify a index_infos_path if you set save_on_disk as True")
        return None, None
    if ids_path:
        ids_path = make_path_absolute(ids_path)

    if nb_indices_to_keep < 1:
        logger.error("Please specify nb_indices_to_keep an integer value larger or equal to 1")
        return None, None
    elif nb_indices_to_keep > 1:
        if distributed is None:
            logger.error('nb_indices_to_keep can only be larger than 1 when distributed is "pyspark"')
            return None, None
        if not save_on_disk:
            logger.error("Please set save_on_disk to True when nb_indices_to_keep is larger than 1")
            return None, None
    current_bytes = cast_memory_to_bytes(current_memory_available)
    max_index_bytes = cast_memory_to_bytes(max_index_memory_usage)
    memory_left = current_bytes - max_index_bytes

    if nb_indices_to_keep == 1 and memory_left < current_bytes * 0.1:
        logger.error(
            "You do not have enough memory to build this index, "
            "please increase current_memory_available or decrease max_index_memory_usage"
        )
        return None, None

    if nb_cores is None:
        nb_cores = multiprocessing.cpu_count()
    logger.info(f"Using {nb_cores} omp threads (processes), consider increasing --nb_cores if you have more")
    faiss.omp_set_num_threads(nb_cores)

    if isinstance(embeddings, np.ndarray):
        tmp_dir_embeddings = tempfile.TemporaryDirectory()
        np.save(os.path.join(tmp_dir_embeddings.name, "emb.npy"), embeddings)
        embeddings_path = tmp_dir_embeddings.name
    else:
        embeddings_path = embeddings  # type: ignore

    with Timeit("Launching the whole pipeline"):
        with Timeit("Reading total number of vectors and dimension"):
            embedding_reader = EmbeddingReader(
                embeddings_path,
                file_format=file_format,
                embedding_column=embedding_column_name,
                meta_columns=id_columns,
            )
            nb_vectors = embedding_reader.count
            vec_dim = embedding_reader.dimension
            logger.info(f"There are {nb_vectors} embeddings of dim {vec_dim}")

        with Timeit("Compute estimated construction time of the index", indent=1):
            for log_lines in get_estimated_construction_time_infos(nb_vectors, vec_dim, indent=2).split("\n"):
                logger.info(log_lines)

        with Timeit("Checking that your have enough memory available to create the index", indent=1):
            necessary_mem, index_key_used = estimate_memory_required_for_index_creation(
                nb_vectors, vec_dim, index_key, max_index_memory_usage, make_direct_map, nb_indices_to_keep
            )
            logger.info(
                f"{cast_bytes_to_memory_string(necessary_mem)} of memory "
                "will be needed to build the index (more might be used if you have more)"
            )

            prefix = "(default) " if index_key is None else ""

            if necessary_mem > cast_memory_to_bytes(current_memory_available):
                r = (
                    f"The current memory available on your machine ({current_memory_available}) is not "
                    f"enough to create the {prefix}index {index_key_used} that requires "
                    f"{cast_bytes_to_memory_string(necessary_mem)} to train. "
                    "You can decrease the number of clusters of you index since the Kmeans algorithm "
                    "used for clusterisation is responsible for this high memory usage."
                    "Consider increasing the options current_memory_available or decreasing max_index_memory_usage"
                )
                logger.error(r)
                return None, None

        if index_key is None:
            with Timeit("Selecting most promising index types given data characteristics", indent=1):
                best_index_keys = get_optimal_index_keys_v2(
                    nb_vectors,
                    vec_dim,
                    max_index_memory_usage,
                    make_direct_map=make_direct_map,
                    should_be_memory_mappable=should_be_memory_mappable,
                    use_gpu=use_gpu,
                )
                if not best_index_keys:
                    return None, None
                index_key = best_index_keys[0]

        if id_columns is not None:
            logger.info(f"Id columns provided {id_columns} - will be reading the corresponding columns")
            if ids_path is not None:
                logger.info(f"\tWill be writing the Ids DataFrame in parquet format to {ids_path}")
                fs, _ = fsspec.core.url_to_fs(ids_path, use_listings_cache=False)
                if fs.exists(ids_path):
                    fs.rm(ids_path, recursive=True)
                fs.mkdirs(ids_path)
            else:
                logger.error(
                    "\tAs ids_path=None - the Ids DataFrame will not be written and will be ignored subsequently"
                )
                logger.error("\tPlease provide a value ids_path for the Ids to be written")

        write_ids_df_to_parquet_fn = get_write_ids_df_to_parquet_fn(ids_path) if ids_path and id_columns else None

        optimize_index_fn = get_optimize_index_fn(
            embedding_reader=embedding_reader,
            index_key=index_key,
            index_path=index_path,
            index_infos_path=index_infos_path,
            use_gpu=use_gpu,
            save_on_disk=save_on_disk,
            max_index_query_time_ms=max_index_query_time_ms,
            min_nearest_neighbors_to_retrieve=min_nearest_neighbors_to_retrieve,
            index_param=index_param,
            make_direct_map=make_direct_map,
        )

        with Timeit("Creating the index", indent=1):
            index, metric_infos = create_index(
                embedding_reader,
                index_key,
                metric_type,
                current_memory_available,
                use_gpu=use_gpu,
                embedding_ids_df_handler=write_ids_df_to_parquet_fn,
                make_direct_map=make_direct_map,
                distributed_engine=distributed,
                temporary_indices_folder=temporary_indices_folder,
                nb_indices_to_keep=nb_indices_to_keep,
                index_optimizer=optimize_index_fn,
            )
            if metric_infos:
                _log_output_dict(metric_infos)
            return index, metric_infos


def build_partitioned_indexes(
    partitions: List[str],
    output_root_dir: str,
    embedding_column_name: str = "embedding",
    index_key: Optional[str] = None,
    id_columns: Optional[List[str]] = None,
    max_index_query_time_ms: float = 10.0,
    max_index_memory_usage: str = "16G",
    min_nearest_neighbors_to_retrieve: int = 20,
    current_memory_available: str = "32G",
    use_gpu: bool = False,
    metric_type: str = "ip",
    nb_cores: Optional[int] = None,
    make_direct_map: bool = False,
    should_be_memory_mappable: bool = False,
    temp_root_dir: str = "hdfs://root/tmp/distributed_autofaiss_indices",
    verbose: int = logging.INFO,
    nb_splits_per_big_index: int = 1,
    big_index_threshold: int = 5_000_000,
    maximum_nb_threads: int = 256,
) -> List[Optional[Dict[str, str]]]:
    """
    Create partitioned indexes from a partitioned parquet dataset,
    i.e. create one index per parquet partition

    Only supported with PySpark. A PySpark session must be active before calling this function

    Parameters
    ----------
    partitions : str
        List of partitions containing embeddings
    output_root_dir: str
        Output root directory where indexes, metrics and ids will be written
    embedding_column_name: str
        Parquet dataset column name containing embeddings
    index_key: Optional(str)
        Optional string to give to the index factory in order to create the index.
        If None, an index is chosen based on an heuristic.
    id_columns: Optional(List[str])
        Parquet dataset column name(s) that are used as IDs for embeddings.
        A mapping from these IDs to faiss indices will be written in separate files.
    max_index_query_time_ms: float
        Bound on the query time for KNN search, this bound is approximative
    max_index_memory_usage: str
        Maximum size allowed for the index, this bound is strict
    min_nearest_neighbors_to_retrieve: int
        Minimum number of nearest neighbors to retrieve when querying the index.
        Parameter used only during index hyperparameter finetuning step, it is
        not taken into account to select the indexing algorithm.
        This parameter has the priority over the max_index_query_time_ms constraint.
    current_memory_available: str
        Memory available on the machine creating the index, having more memory is a boost
        because it reduces the swipe between RAM and disk.
    use_gpu: bool
        Experimental, gpu training is faster, not tested so far
    metric_type: str
        Similarity function used for query:
            - "ip" for inner product
            - "l2" for euclidean distance
    nb_cores: Optional[int]
        Number of cores to use. Will try to guess the right number if not provided
    make_direct_map: bool
        Create a direct map allowing reconstruction of embeddings. This is only needed for IVF indices.
        Note that might increase the RAM usage (approximately 8GB for 1 billion embeddings)
    should_be_memory_mappable: bool
        If set to true, the created index will be selected only among the indices that can be memory-mapped on disk.
        This makes it possible to use 50GB indices on a machine with only 1GB of RAM. Default to False
    temp_root_dir: str
        Temporary directory that will be used to store intermediate results/computation
    verbose: int
        set verbosity of outputs via logging level, default is `logging.INFO`
    nb_splits_per_big_index: int
        Number of indices to split a big index into.
        This allows you building indices bigger than `current_memory_available`.
    big_index_threshold: int
        Threshold used to define big indexes.
        Indexes with more `than big_index_threshold` embeddings are considered big indexes.
    maximum_nb_threads: int
        Maximum number of threads to parallelize index creation
    """
    setup_logging(verbose)

    # Sanity checks
    if not partitions:
        raise ValueError("partitions can't be empty")
    check_not_null_not_empty("output_root_dir", output_root_dir)
    check_not_null_not_empty("embedding_column_name", embedding_column_name)
    if nb_splits_per_big_index < 1:
        raise ValueError(f"nb_indices_to_keep must be > 0; Got {nb_splits_per_big_index}")
    if big_index_threshold < 1:
        raise ValueError(f"big_index_threshold must be > 0; Got {big_index_threshold}")
    if index_key:
        n_dimensions = EmbeddingReader(
            partitions[0], file_format="parquet", embedding_column=embedding_column_name
        ).dimension
        # Create an empty index to validate the index key
        create_empty_index(n_dimensions, index_key=index_key, metric_type=metric_type)

    # Create partitioned indexes
    return create_partitioned_indexes(
        partitions=partitions,
        output_root_dir=output_root_dir,
        embedding_column_name=embedding_column_name,
        index_key=index_key,
        id_columns=id_columns,
        should_be_memory_mappable=should_be_memory_mappable,
        max_index_query_time_ms=max_index_query_time_ms,
        max_index_memory_usage=max_index_memory_usage,
        min_nearest_neighbors_to_retrieve=min_nearest_neighbors_to_retrieve,
        current_memory_available=current_memory_available,
        use_gpu=use_gpu,
        metric_type=metric_type,
        nb_cores=nb_cores,
        make_direct_map=make_direct_map,
        temp_root_dir=temp_root_dir,
        nb_splits_per_big_index=nb_splits_per_big_index,
        big_index_threshold=big_index_threshold,
        maximum_nb_threads=maximum_nb_threads,
    )


def check_not_null_not_empty(name: str, value: str):
    if not value:
        raise ValueError(f"{name} can't be None or empty; Got {value}")


def tune_index(
    index_path: Union[str, faiss.Index],
    index_key: str,
    index_param: Optional[str] = None,
    output_index_path: Optional[str] = "tuned_knn.index",
    save_on_disk: bool = True,
    min_nearest_neighbors_to_retrieve: int = 20,
    max_index_query_time_ms: float = 10.0,
    use_gpu: bool = False,
    verbose: int = logging.INFO,
) -> faiss.Index:
    """
    Set hyperparameters to the given index.

    If an index_param is given, set this hyperparameters to the index,
    otherwise perform a greedy heusistic to make the best out or the max_index_query_time_ms constraint

    Parameters
    ----------
    index_path : Union[str, faiss.Index]
        Path to .index file
        Can also be an index
    index_key: str
        String to give to the index factory in order to create the index.
    index_param: Optional(str)
        Optional string with hyperparameters to set to the index.
        If None, the hyper-parameters are chosen based on an heuristic.
    output_index_path: str
        Path to the newly created .index file
    save_on_disk: bool
        Whether to save the index on disk, default to True.
    min_nearest_neighbors_to_retrieve: int
        Minimum number of nearest neighbors to retrieve when querying the index.
    max_index_query_time_ms: float
        Query speed constraint for the index to create.
    use_gpu: bool
        Experimental, gpu training is faster, not tested so far.
    verbose: int
        set verbosity of outputs via logging level, default is `logging.INFO`

    Returns
    -------
    index
        The faiss index
    """
    setup_logging(verbose)

    if isinstance(index_path, str):
        index_path = make_path_absolute(index_path)
        with fsspec.open(index_path, "rb").open() as f:
            index = faiss.read_index(faiss.PyCallbackIOReader(f.read))
    else:
        index = index_path

    if index_param is None:
        with Timeit("Compute best hyperparameters"):
            index_param = get_optimal_hyperparameters(
                index,
                index_key,
                max_speed_ms=max_index_query_time_ms,
                min_nearest_neighbors_to_retrieve=min_nearest_neighbors_to_retrieve,
            )

    with Timeit("Set search hyperparameters for the index"):
        set_search_hyperparameters(index, index_param, use_gpu)

    logger.info(f"The optimal hyperparameters are {index_param}.")

    if save_on_disk:
        with fsspec.open(output_index_path, "wb").open() as f:
            faiss.write_index(index, faiss.PyCallbackIOWriter(f.write))
        logger.info("The index with these parameters has been saved on disk.")

    return index


def score_index(
    index_path: Union[str, faiss.Index],
    embeddings: Union[str, np.ndarray],
    save_on_disk: bool = True,
    output_index_info_path: str = "infos.json",
    current_memory_available: str = "32G",
    verbose: int = logging.INFO,
) -> Optional[Dict[str, Union[str, float, int]]]:
    """
    Compute metrics on a given index, use cached ground truth for fast scoring the next times.

    Parameters
    ----------
    index_path : Union[str, faiss.Index]
        Path to .index file. Or in memory index
    embeddings: Union[str, np.ndarray]
        Path containing all preprocessed vectors and cached files. Can also be an in memory array.
    save_on_disk: bool
        Whether to save on disk
    output_index_info_path : str
        Path to index infos .json
    current_memory_available: str
        Memory available on the current machine, having more memory is a boost
        because it reduces the swipe between RAM and disk.
    verbose: int
        set verbosity of outputs via logging level, default is `logging.INFO`

    Returns
    -------
    metric_infos: Optional[Dict[str, Union[str, float, int]]]
        Metric infos of the index.
    """
    setup_logging(verbose)
    faiss.omp_set_num_threads(multiprocessing.cpu_count())

    if isinstance(index_path, str):
        index_path = make_path_absolute(index_path)
        with fsspec.open(index_path, "rb").open() as f:
            index = faiss.read_index(faiss.PyCallbackIOReader(f.read))
        fs, path_in_fs = fsspec.core.url_to_fs(index_path, use_listings_cache=False)
        index_memory = fs.size(path_in_fs)
    else:
        index = index_path
        with tempfile.NamedTemporaryFile("wb") as f:
            faiss.write_index(index, faiss.PyCallbackIOWriter(f.write))
            fs, path_in_fs = fsspec.core.url_to_fs(f.name, use_listings_cache=False)
            index_memory = fs.size(path_in_fs)

    if isinstance(embeddings, np.ndarray):
        tmp_dir_embeddings = tempfile.TemporaryDirectory()
        np.save(os.path.join(tmp_dir_embeddings.name, "emb.npy"), embeddings)
        embeddings_path = tmp_dir_embeddings.name
    else:
        embeddings_path = embeddings

    embedding_reader = EmbeddingReader(embeddings_path, file_format="npy")

    infos: Dict[str, Union[str, float, int]] = {}

    with Timeit("Compute fast metrics"):
        infos.update(compute_fast_metrics(embedding_reader, index))

    logger.info("Intermediate recap:")
    _log_output_dict(infos)

    current_in_bytes = cast_memory_to_bytes(current_memory_available)
    memory_left = current_in_bytes - index_memory

    if memory_left < current_in_bytes * 0.1:
        logger.info(
            f"Not enough memory, at least {cast_bytes_to_memory_string(index_memory * 1.1)}"
            "is needed, please increase current_memory_available"
        )
        return None

    with Timeit("Compute medium metrics"):
        infos.update(compute_medium_metrics(embedding_reader, index, memory_left))

    logger.info("Performances recap:")
    _log_output_dict(infos)

    if save_on_disk:
        with fsspec.open(output_index_info_path, "w").open() as f:
            json.dump(infos, f)

    return infos


def main():
    """Main entry point"""
    fire.Fire(
        {
            "build_index": build_index,
            "tune_index": tune_index,
            "score_index": score_index,
            "build_partitioned_indexes": build_partitioned_indexes,
        }
    )


if __name__ == "__main__":
    main()
