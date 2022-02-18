""" main file to create an index from the the begining """

import json
import logging
import logging.config
import multiprocessing
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import fire
import fsspec
import numpy as np
import pandas as pd
from autofaiss.external.build import (
    create_index,
    estimate_memory_required_for_index_creation,
    get_estimated_construction_time_infos,
)
from autofaiss.external.optimize import get_optimal_hyperparameters, get_optimal_index_keys_v2
from autofaiss.external.scores import compute_fast_metrics, compute_medium_metrics
from autofaiss.indices.index_utils import set_search_hyperparameters
from autofaiss.readers.embeddings_iterators import get_file_list, make_path_absolute, read_total_nb_vectors_and_dim
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
    current_memory_available: str = "32G",
    use_gpu: bool = False,
    metric_type: str = "ip",
    nb_cores: Optional[int] = None,
    make_direct_map: bool = False,
    should_be_memory_mappable: bool = False,
    distributed: Optional[str] = None,
    temporary_indices_folder: str = "hdfs://root/tmp/distributed_autofaiss_indices",
    verbose: int = logging.INFO,
) -> Tuple[Optional[Any], Optional[Dict[str, Union[str, float, int]]]]:
    """
    Reads embeddings and creates a quantized index from them.
    The index is stored on the current machine at the given ouput path.

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
    """
    setup_logging(verbose)
    if index_path is not None:
        index_path = make_path_absolute(index_path)
    elif save_on_disk:
        logger.error("Please specify a index_path if you set save_on_disk as True")
        return None, None
    if index_infos_path is not None:
        index_infos_path = make_path_absolute(index_infos_path)
    elif save_on_disk:
        logger.error("Please specify a index_infos_path if you set save_on_disk as True")
        return None, None
    if ids_path is not None:
        ids_path = make_path_absolute(ids_path)

    current_bytes = cast_memory_to_bytes(current_memory_available)
    max_index_bytes = cast_memory_to_bytes(max_index_memory_usage)
    memory_left = current_bytes - max_index_bytes

    if memory_left < current_bytes * 0.1:
        logger.error(
            "You do not have enough memory to build this index"
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
            _, embeddings_file_paths = get_file_list(path=embeddings_path, file_format=file_format)
            nb_vectors, vec_dim, file_counts = read_total_nb_vectors_and_dim(
                embeddings_file_paths, file_format=file_format, embedding_column_name=embedding_column_name
            )
            embeddings_file_paths, file_counts = zip(  # type: ignore
                *((fp, count) for fp, count in zip(embeddings_file_paths, file_counts) if count > 0)
            )
            embeddings_file_paths = list(embeddings_file_paths)
            file_counts = list(file_counts)
            logger.info(f"There are {nb_vectors} embeddings of dim {vec_dim}")

        with Timeit("Compute estimated construction time of the index", indent=1):
            for log_lines in get_estimated_construction_time_infos(nb_vectors, vec_dim, indent=2).split("\n"):
                logger.info(log_lines)

        with Timeit("Checking that your have enough memory available to create the index", indent=1):
            necessary_mem, index_key_used = estimate_memory_required_for_index_creation(
                nb_vectors, vec_dim, index_key, max_index_memory_usage, make_direct_map
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
                fs, _ = fsspec.core.url_to_fs(ids_path)
                if fs.exists(ids_path):
                    fs.rm(ids_path, recursive=True)
                fs.mkdirs(ids_path)
            else:
                logger.error(
                    "\tAs ids_path=None - the Ids DataFrame will not be written and will be ignored subsequently"
                )
                logger.error("\tPlease provide a value ids_path for the Ids to be written")

        def write_ids_df_to_parquet(ids: pd.DataFrame, batch_id: int):
            filename = f"part-{batch_id:08d}-{uuid.uuid1()}.parquet"
            output_file = os.path.join(ids_path, filename)  # type: ignore
            with fsspec.open(output_file, "wb") as f:
                logger.debug(f"Writing id DataFrame to file {output_file}")
                ids.to_parquet(f)

        with Timeit("Creating the index", indent=1):
            index = create_index(
                embeddings_file_paths,
                index_key,
                metric_type,
                nb_vectors,
                current_memory_available,
                use_gpu=use_gpu,
                file_format=file_format,
                embedding_column_name=embedding_column_name,
                id_columns=id_columns,
                embedding_ids_df_handler=write_ids_df_to_parquet if ids_path and id_columns else None,
                make_direct_map=make_direct_map,
                distributed=distributed,
                temporary_indices_folder=temporary_indices_folder,
                file_counts=file_counts if distributed is not None else None,
            )

        if index_param is None:
            with Timeit("Computing best hyperparameters", indent=1):
                index_param = get_optimal_hyperparameters(index, index_key, max_speed_ms=max_index_query_time_ms)

        # Set search hyperparameters for the index
        set_search_hyperparameters(index, index_param, use_gpu)
        logger.info(f"The best hyperparameters are: {index_param}")

        metric_infos: Dict[str, Union[str, float, int]] = {}
        metric_infos["index_key"] = index_key
        metric_infos["index_param"] = index_param

        with Timeit("Compute fast metrics", indent=1):
            metric_infos.update(
                compute_fast_metrics(
                    embeddings_file_paths, index, file_format=file_format, embedding_column_name=embedding_column_name
                )
            )

        if save_on_disk:
            with Timeit("Saving the index on local disk", indent=1):
                with fsspec.open(index_path, "wb").open() as f:
                    faiss.write_index(index, faiss.PyCallbackIOWriter(f.write))
                with fsspec.open(index_infos_path, "w").open() as f:
                    json.dump(metric_infos, f)

        logger.info("Recap:")
        _log_output_dict(metric_infos)

    return index, metric_infos


def tune_index(
    index_path: Union[str, Any],
    index_key: str,
    index_param: Optional[str] = None,
    output_index_path: Optional[str] = None,
    save_on_disk: bool = True,
    max_index_query_time_ms: float = 10.0,
    use_gpu: bool = False,
    verbose: int = logging.INFO,
) -> Tuple[Optional[Any], Optional[Dict[str, Union[str, float, int]]]]:
    """
    Set hyperparameters to the given index.

    If an index_param is given, set this hyperparameters to the index,
    otherwise perform a greedy heusistic to make the best out or the max_index_query_time_ms constraint

    Parameters
    ----------
    index_path : Union[str, Any]
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
        with fsspec.open(index_path, "r").open() as f:
            index = faiss.read_index(faiss.PyCallbackIOReader(f.read))
    else:
        index = index_path

    if index_param is None:
        with Timeit("Compute best hyperparameters"):
            index_param = get_optimal_hyperparameters(index, index_key, max_speed_ms=max_index_query_time_ms)

    with Timeit("Set search hyperparameters for the index"):
        set_search_hyperparameters(index, index_param, use_gpu)

    if save_on_disk:
        with fsspec.open(output_index_path, "wb").open() as f:
            faiss.write_index(index, faiss.PyCallbackIOWriter(f.write))

    logger.info(f"The optimal hyperparameters are {index_param}, the index with these parameters has been saved.")

    return index


def score_index(
    index_path: Union[str, Any],
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
    index_path : Union[str, Any]
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
    """
    setup_logging(verbose)
    faiss.omp_set_num_threads(multiprocessing.cpu_count())

    if isinstance(index_path, str):
        index_path = make_path_absolute(index_path)
        with fsspec.open(index_path, "r").open() as f:
            index = faiss.read_index(faiss.PyCallbackIOReader(f.read))
        fs, path_in_fs = fsspec.core.url_to_fs(index_path)
        index_memory = fs.size(path_in_fs)
    else:
        index = index_path
        with tempfile.NamedTemporaryFile("wb") as f:
            faiss.write_index(index, faiss.PyCallbackIOWriter(f.write))
            fs, path_in_fs = fsspec.core.url_to_fs(f.name)
            index_memory = fs.size(path_in_fs)

    if isinstance(embeddings, np.ndarray):
        tmp_dir_embeddings = tempfile.TemporaryDirectory()
        np.save(os.path.join(tmp_dir_embeddings.name, "emb.npy"), embeddings)
        embeddings_path = tmp_dir_embeddings.name
    else:
        embeddings_path = embeddings

    infos: Dict[str, Union[str, float, int]] = {}

    with Timeit("Compute fast metrics"):
        infos.update(compute_fast_metrics(embeddings_path, index))

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
        infos.update(compute_medium_metrics(embeddings_path, index, memory_left))

    logger.info("Performances recap:")
    _log_output_dict(infos)

    if save_on_disk:
        with fsspec.open(output_index_info_path, "w").open() as f:
            json.dump(infos, f)

    return infos


def main():
    """Main entry point"""
    fire.Fire({"build_index": build_index, "tune_index": tune_index, "score_index": score_index})


if __name__ == "__main__":
    main()
