""" gather functions necessary to build an index """

import logging
from typing import Dict, Optional, Tuple, Union, Callable, Any, List

import faiss
import pandas as pd
from embedding_reader import EmbeddingReader

from autofaiss.external.metadata import IndexMetadata
from autofaiss.external.optimize import check_if_index_needs_training, get_optimal_index_keys_v2, get_optimal_train_size
from autofaiss.utils.cast import cast_bytes_to_memory_string, cast_memory_to_bytes, to_readable_time
from autofaiss.utils.decorators import Timeit
from autofaiss.indices import distributed
from autofaiss.indices.index_utils import initialize_direct_map
from autofaiss.indices.training import create_and_train_new_index
from autofaiss.indices.build import add_embeddings_to_index_local


logger = logging.getLogger("autofaiss")


def estimate_memory_required_for_index_creation(
    nb_vectors: int,
    vec_dim: int,
    index_key: Optional[str] = None,
    max_index_memory_usage: Optional[str] = None,
    make_direct_map: bool = False,
    nb_indices_to_keep: int = 1,
) -> Tuple[int, str]:
    """
    Estimates the RAM necessary to create the index
    The value returned is in Bytes
    """

    if index_key is None:
        if max_index_memory_usage is not None:
            index_key = get_optimal_index_keys_v2(
                nb_vectors, vec_dim, max_index_memory_usage, make_direct_map=make_direct_map
            )[0]
        else:
            raise ValueError("you should give max_index_memory_usage value if no index_key is given")

    metadata = IndexMetadata(index_key, nb_vectors, vec_dim, make_direct_map)

    index_memory = metadata.estimated_index_size_in_bytes()
    needed_for_adding = min(index_memory * 0.1, 10 ** 9)

    index_needs_training = check_if_index_needs_training(index_key)

    if index_needs_training:
        # Compute the smallest number of vectors required to train the index given
        # the maximal memory constraint
        nb_vectors_train = get_optimal_train_size(nb_vectors, index_key, "1K", vec_dim)

        memory_for_training = metadata.compute_memory_necessary_for_training(nb_vectors_train)
    else:
        memory_for_training = 0

    # the calculation for max_index_memory_in_one_index comes from the way we split batches
    # see _batch_loader in distributed.py
    max_index_memory_in_one_index = index_memory // nb_indices_to_keep + index_memory % nb_indices_to_keep

    return int(max(max_index_memory_in_one_index + needed_for_adding, memory_for_training)), index_key


def get_estimated_construction_time_infos(nb_vectors: int, vec_dim: int, indent: int = 0) -> str:
    """
    Gives a general approximation of the construction time of the index
    """

    size = 4 * nb_vectors * vec_dim

    train = 1000  # seconds, depends on the number of points for training
    add = 450 * size / (150 * 1024 ** 3)  # seconds, Linear approx (450s for 150GB in classic conditions)

    infos = (
        f"-> Train: {to_readable_time(train, rounding=True)}\n"
        f"-> Add: {to_readable_time(add, rounding=True)}\n"
        f"Total: {to_readable_time(train + add, rounding=True)}"
    )
    tab = "\t" * indent
    infos = tab + infos.replace("\n", "\n" + tab)

    return infos


def add_embeddings_to_index(
    embedding_reader: EmbeddingReader,
    trained_index_or_path: Union[str, faiss.Index],
    metadata: IndexMetadata,
    current_memory_available: str,
    embedding_ids_df_handler: Optional[Callable[[pd.DataFrame, int], Any]] = None,
    distributed_engine: Optional[str] = None,
    temporary_indices_folder: str = "hdfs://root/tmp/distributed_autofaiss_indices",
    nb_indices_to_keep: int = 1,
    index_optimizer: Callable = None,
) -> Tuple[Optional[faiss.Index], Optional[Dict[str, str]]]:
    """Add embeddings to the index"""

    with Timeit("-> Adding the vectors to the index", indent=2):

        # Estimate memory available for adding embeddings to index
        size_per_index = metadata.estimated_index_size_in_bytes() / nb_indices_to_keep
        memory_available_for_adding = cast_bytes_to_memory_string(
            cast_memory_to_bytes(current_memory_available) - size_per_index
        )
        logger.info(
            f"The memory available for adding the vectors is {memory_available_for_adding}"
            "(total available - used by the index)"
        )

        if distributed_engine is None:
            return add_embeddings_to_index_local(
                embedding_reader=embedding_reader,
                trained_index_or_path=trained_index_or_path,
                memory_available_for_adding=memory_available_for_adding,
                embedding_ids_df_handler=embedding_ids_df_handler,
                index_optimizer=index_optimizer,
                add_embeddings_with_ids=False,
            )

        elif distributed_engine == "pyspark":
            return distributed.add_embeddings_to_index_distributed(
                trained_index_or_path=trained_index_or_path,
                embedding_reader=embedding_reader,
                memory_available_for_adding=memory_available_for_adding,
                embedding_ids_df_handler=embedding_ids_df_handler,
                temporary_indices_folder=temporary_indices_folder,
                nb_indices_to_keep=nb_indices_to_keep,
                index_optimizer=index_optimizer,
            )
        else:
            raise ValueError(f'Distributed by {distributed_engine} is not supported, only "pyspark" is supported')


def create_index(
    embedding_reader: EmbeddingReader,
    index_key: str,
    metric_type: Union[str, int],
    current_memory_available: str,
    embedding_ids_df_handler: Optional[Callable[[pd.DataFrame, int], Any]] = None,
    use_gpu: bool = False,
    make_direct_map: bool = False,
    distributed_engine: Optional[str] = None,
    temporary_indices_folder: str = "hdfs://root/tmp/distributed_autofaiss_indices",
    nb_indices_to_keep: int = 1,
    index_optimizer: Callable = None,
) -> Tuple[Optional[faiss.Index], Optional[Dict[str, str]]]:
    """
    Create an index and add embeddings to the index
    """

    metadata = IndexMetadata(index_key, embedding_reader.count, embedding_reader.dimension, make_direct_map)

    # Create and train index
    trained_index = create_and_train_new_index(
        embedding_reader, index_key, metadata, metric_type, current_memory_available, use_gpu
    )

    # Add embeddings to index
    index, metrics = add_embeddings_to_index(
        embedding_reader,
        trained_index,
        metadata,
        current_memory_available,
        embedding_ids_df_handler,
        distributed_engine,
        temporary_indices_folder,
        nb_indices_to_keep,
        index_optimizer,
    )

    if make_direct_map:
        initialize_direct_map(index)

    return index, metrics


def create_partitioned_indexes(
    partitions: List[str],
    output_root_dir: str,
    embedding_column_name: str = "embedding",
    index_key: Optional[str] = None,
    id_columns: Optional[List[str]] = None,
    should_be_memory_mappable: bool = False,
    max_index_query_time_ms: float = 10.0,
    max_index_memory_usage: str = "16G",
    min_nearest_neighbors_to_retrieve: int = 20,
    current_memory_available: str = "32G",
    use_gpu: bool = False,
    metric_type: str = "ip",
    nb_cores: Optional[int] = None,
    make_direct_map: bool = False,
    temp_root_dir: str = "hdfs://root/tmp/distributed_autofaiss_indices",
    big_index_threshold: int = 5_000_000,
    nb_splits_per_big_index: int = 1,
    maximum_nb_threads: int = 256,
) -> List[Optional[Dict[str, str]]]:
    """
    Create partitioned indexes from a list of parquet partitions, i.e. create one index per parquet partition

    Only supported with Pyspark. An active PySpark session must exist before calling this method
    """

    return distributed.create_partitioned_indexes(
        partitions=partitions,
        big_index_threshold=big_index_threshold,
        output_root_dir=output_root_dir,
        nb_cores=nb_cores,
        nb_splits_per_big_index=nb_splits_per_big_index,
        id_columns=id_columns,
        max_index_query_time_ms=max_index_query_time_ms,
        min_nearest_neighbors_to_retrieve=min_nearest_neighbors_to_retrieve,
        embedding_column_name=embedding_column_name,
        index_key=index_key,
        max_index_memory_usage=max_index_memory_usage,
        current_memory_available=current_memory_available,
        use_gpu=use_gpu,
        metric_type=metric_type,
        make_direct_map=make_direct_map,
        should_be_memory_mappable=should_be_memory_mappable,
        temp_root_dir=temp_root_dir,
        maximum_nb_threads=maximum_nb_threads,
    )
