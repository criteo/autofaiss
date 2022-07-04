""" gather functions necessary to build an index """
import logging
from typing import Dict, Optional, Tuple, Union, Callable, Any

import faiss
import pandas as pd
from faiss import extract_index_ivf
from embedding_reader import EmbeddingReader

from autofaiss.external.metadata import IndexMetadata
from autofaiss.external.optimize import (
    check_if_index_needs_training,
    get_optimal_batch_size,
    get_optimal_index_keys_v2,
    get_optimal_train_size,
)
from autofaiss.indices.index_factory import index_factory
from autofaiss.utils.cast import (
    cast_bytes_to_memory_string,
    cast_memory_to_bytes,
    to_faiss_metric_type,
    to_readable_time,
)
from autofaiss.utils.decorators import Timeit
from autofaiss.indices import distributed

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


def create_empty_index(
    vec_dim: int,
    index_key: str,
    metric_type: Union[str, int],
) -> faiss.Index:
    """Create empty index"""

    with Timeit(f"-> Instanciate the index {index_key}", indent=2):

        # Convert metric_type to faiss type
        metric_type = to_faiss_metric_type(metric_type)

        # Instanciate the index
        return index_factory(vec_dim, index_key, metric_type)


def train_index(
    embedding_reader: EmbeddingReader,
    index_key: str,
    index: faiss.Index,
    metadata: IndexMetadata,
    current_memory_available: int,
    use_gpu: bool
) -> faiss.Index:
    """Train index"""

    logger.info(
        f"The index size will be approximately {cast_bytes_to_memory_string(metadata.estimated_index_size_in_bytes())}"
    )

    # Extract training vectors
    with Timeit("-> Extract training vectors", indent=2):

        memory_available_for_training = cast_bytes_to_memory_string(cast_memory_to_bytes(current_memory_available))

        # Determine the number of vectors necessary to train the index
        train_size = get_optimal_train_size(
            embedding_reader.count, index_key, memory_available_for_training, embedding_reader.dimension
        )
        memory_needed_for_training = metadata.compute_memory_necessary_for_training(train_size)
        logger.info(
            f"Will use {train_size} vectors to train the index, "
            f"that will use {cast_bytes_to_memory_string(memory_needed_for_training)} of memory"
        )

        # Extract training vectors
        train_vectors, _ = next(embedding_reader(batch_size=train_size, start=0, end=train_size))

    # Instanciate the index and train it
    # pylint: disable=no-member
    if use_gpu:
        # if this fails, it means that the GPU version was not comp.
        assert (
            faiss.StandardGpuResources
        ), "FAISS was not compiled with GPU support, or loading _swigfaiss_gpu.so failed"
        res = faiss.StandardGpuResources()
        dev_no = 0
        # transfer to GPU (may be partial).
        index = faiss.index_cpu_to_gpu(res, dev_no, index)

    with Timeit(
        f"-> Training the index with {train_vectors.shape[0]} vectors of dim {train_vectors.shape[1]}", indent=2
    ):
        index.train(train_vectors)

    del train_vectors

    return index


def create_and_train_index(
    embedding_reader: EmbeddingReader,
    index_key: str,
    metadata: IndexMetadata,
    metric_type: Union[str, int],
    current_memory_available: str,
    use_gpu: bool = False
) -> faiss.Index:
    """Create and train index"""

    # Instanciate the index
    index = create_empty_index(embedding_reader.dimension, index_key, metric_type)

    # Train index if needed
    if check_if_index_needs_training(index_key):
        index = train_index(
            embedding_reader,
            index_key,
            index,
            metadata,
            current_memory_available,
            use_gpu
        )
    return index


def add_embeddings_to_index_local(
    embedding_reader: EmbeddingReader,
    index: faiss.Index,
    memory_available_for_adding: str,
    embedding_ids_df_handler: Optional[Callable[[pd.DataFrame, int], Any]] = None,
    index_optimizer: Callable = None
) -> Tuple[faiss.Index, Dict[str, str]]:
    """Add embeddings to index from driver"""

    vec_dim = embedding_reader.dimension
    batch_size = get_optimal_batch_size(vec_dim, memory_available_for_adding)
    logger.info(
        f"Using a batch size of {batch_size} (memory overhead {cast_bytes_to_memory_string(batch_size * vec_dim * 4)})"
    )
    for batch_id, (vec_batch, ids_batch) in enumerate(embedding_reader(batch_size=batch_size)):
        index.add(vec_batch)
        if embedding_ids_df_handler:
            embedding_ids_df_handler(ids_batch, batch_id)
    metric_infos = index_optimizer(index, "")  # type: ignore
    return index, metric_infos


def add_embeddings_to_index(
    embedding_reader: EmbeddingReader,
    index: faiss.Index,
    metadata: IndexMetadata,
    current_memory_available: str,
    embedding_ids_df_handler: Optional[Callable[[pd.DataFrame, int], Any]] = None,
    make_direct_map: bool = False,
    distributed_engine: Optional[str] = None,
    temporary_indices_folder: str = "hdfs://root/tmp/distributed_autofaiss_indices",
    nb_indices_to_keep: int = 1,
    index_optimizer: Callable = None
) -> Tuple[faiss.Index, Dict[str, str]]:
    """Add embeddings to the index"""

    with Timeit("-> Adding the vectors to the index", indent=2):

        # Memory map index
        if make_direct_map:
            # Retrieve the embedded index if we are in an IndexPreTransform state
            embedded_index = extract_index_ivf(index) if isinstance(index, faiss.swigfaiss.IndexPreTransform) else index

            # Make direct map is only implemented for IndexIVF and IndexBinaryIVF, see built file faiss/swigfaiss.py
            if isinstance(embedded_index, (faiss.swigfaiss.IndexIVF, faiss.swigfaiss.IndexBinaryIVF)):
                embedded_index.make_direct_map()

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
                embedding_reader,
                index,
                memory_available_for_adding,
                embedding_ids_df_handler,
                index_optimizer
            )
        elif distributed_engine == "pyspark":
            return distributed.add_embeddings_to_index(
                faiss_index_or_path=index,
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
) -> Tuple[faiss.Index, Dict[str, str]]:
    """
    Create an index and add embeddings to the index
    """

    metadata = IndexMetadata(index_key, embedding_reader.count, embedding_reader.dimension, make_direct_map)

    # Create and train index
    index = create_and_train_index(
        embedding_reader,
        index_key,
        metadata,
        metric_type,
        current_memory_available,
        use_gpu
    )

    # Add embeddings to index
    return add_embeddings_to_index(
        embedding_reader,
        index,
        metadata,
        current_memory_available,
        embedding_ids_df_handler,
        make_direct_map,
        distributed_engine,
        temporary_indices_folder,
        nb_indices_to_keep,
        index_optimizer
    )
