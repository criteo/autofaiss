"""Index training"""

from typing import Union
import logging

import faiss
from embedding_reader import EmbeddingReader

from autofaiss.external.metadata import IndexMetadata
from autofaiss.external.optimize import (
    check_if_index_needs_training,
    get_optimal_train_size
)
from autofaiss.indices.index_factory import index_factory
from autofaiss.utils.cast import (
    cast_bytes_to_memory_string,
    cast_memory_to_bytes,
    to_faiss_metric_type,
)
from autofaiss.utils.decorators import Timeit


logger = logging.getLogger("autofaiss")


def _create_empty_index(vec_dim: int, index_key: str, metric_type: Union[str, int]) -> faiss.Index:
    """Create empty index"""

    with Timeit(f"-> Instanciate the index {index_key}", indent=2):

        # Convert metric_type to faiss type
        metric_type = to_faiss_metric_type(metric_type)

        # Instanciate the index
        return index_factory(vec_dim, index_key, metric_type)


def _train_index(
    embedding_reader: EmbeddingReader,
    index_key: str,
    index: faiss.Index,
    metadata: IndexMetadata,
    current_memory_available: str,
    use_gpu: bool,
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


def create_and_train_new_index(
    embedding_reader: EmbeddingReader,
    index_key: str,
    metadata: IndexMetadata,
    metric_type: Union[str, int],
    current_memory_available: str,
    use_gpu: bool = False,
) -> faiss.Index:
    """Create and train new index"""

    # Instanciate the index
    index = _create_empty_index(embedding_reader.dimension, index_key, metric_type)

    # Train index if needed
    if check_if_index_needs_training(index_key):
        index = _train_index(embedding_reader, index_key, index, metadata, current_memory_available, use_gpu)
    return index
