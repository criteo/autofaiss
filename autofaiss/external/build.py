""" gather functions necessary to build an index """
import re
from typing import Optional, Tuple, Union, Callable, Any, List

import faiss
import pandas as pd
from faiss import extract_index_ivf

from autofaiss.readers.embeddings_iterators import read_first_file_shape, read_embeddings
from autofaiss.external.metadata import IndexMetadata
from autofaiss.external.optimize import (
    check_if_index_needs_training,
    compute_memory_necessary_for_training,
    get_optimal_batch_size,
    get_optimal_index_keys_v2,
    get_optimal_train_size,
    set_search_hyperparameters,
)
from autofaiss.indices.index_factory import index_factory
from autofaiss.utils.cast import (
    cast_bytes_to_memory_string,
    cast_memory_to_bytes,
    to_faiss_metric_type,
    to_readable_time,
)
from autofaiss.utils.decorators import Timeit
from autofaiss.indices.distributed import run


def estimate_memory_required_for_index_creation(
    nb_vectors: int,
    vec_dim: int,
    index_key: Optional[str] = None,
    max_index_memory_usage: Optional[str] = None,
    make_direct_map: bool = False,
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
    needed_for_adding = min(index_memory * 0.1, 10**9)
    index_overhead = index_memory * 0.1

    index_needs_training = check_if_index_needs_training(index_key)

    if index_needs_training:
        # Compute the smallest number of vectors required to train the index given
        # the maximal memory constraint
        nb_vectors_train = get_optimal_train_size(nb_vectors, index_key, "1K", vec_dim)

        memory_for_training = (
            compute_memory_necessary_for_training(nb_vectors_train, index_key, vec_dim) + index_memory * 0.25
        )
    else:
        memory_for_training = 0

    return (int(index_overhead + max(index_memory + needed_for_adding, memory_for_training))), index_key


def get_estimated_construction_time_infos(nb_vectors: int, vec_dim: int, indent: int = 0) -> str:
    """
    Gives a general approximation of the construction time of the index
    """

    size = 4 * nb_vectors * vec_dim

    train = 1000  # seconds, depends on the number of points for training
    add = 450 * size / (150 * 1024**3)  # seconds, Linear approx (450s for 150GB in classic conditions)

    infos = (
        f"-> Train: {to_readable_time(train, rounding=True)}\n"
        f"-> Add: {to_readable_time(add, rounding=True)}\n"
        f"Total: {to_readable_time(train + add, rounding=True)}"
    )
    tab = "\t" * indent
    infos = tab + infos.replace("\n", "\n" + tab)

    return infos


def create_index(
    embeddings_file_paths: List[str],
    index_key: str,
    metric_type: Union[str, int],
    nb_vectors: int,
    current_memory_available: str,
    file_format: str = "npy",
    embedding_column_name: str = "embeddings",
    id_columns: Optional[List[str]] = None,
    embedding_ids_df_handler: Optional[Callable[[pd.DataFrame, int], Any]] = None,
    use_gpu: bool = False,
    make_direct_map: bool = False,
    distributed: Optional[str] = None,
    temporary_indices_folder: str = "hdfs://root/tmp/distributed_autofaiss_indices",
    file_counts: List[int] = None,
):
    """
    Function that returns an index on the numpy arrays stored on disk in the embeddings_path path.
    """

    # Instanciate the index
    with Timeit(f"-> Instanciate the index {index_key}", indent=2):

        # Convert metric_type to faiss type
        metric_type = to_faiss_metric_type(metric_type)

        # Get information for one partition
        _, vec_dim = read_first_file_shape(
            embeddings_file_paths, file_format=file_format, embedding_column_name=embedding_column_name
        )

        # Instanciate the index
        index = index_factory(vec_dim, index_key, metric_type)

    metadata = IndexMetadata(index_key, nb_vectors, vec_dim, make_direct_map)

    print(
        f"The index size will be approximately {cast_bytes_to_memory_string(metadata.estimated_index_size_in_bytes())}"
    )

    index_needs_training = check_if_index_needs_training(index_key)

    if index_needs_training:

        # Extract training vectors
        with Timeit("-> Extract training vectors", indent=2):

            memory_available_for_training = cast_bytes_to_memory_string(
                cast_memory_to_bytes(current_memory_available) - metadata.estimated_index_size_in_bytes() * 0.25
            )

            # Determine the number of vectors necessary to train the index
            train_size = get_optimal_train_size(nb_vectors, index_key, memory_available_for_training, vec_dim)
            memory_needed_for_training = compute_memory_necessary_for_training(train_size, index_key, vec_dim)
            print(
                f"Will use {train_size} vectors to train the index, "
                f"that will use {cast_bytes_to_memory_string(memory_needed_for_training)} of memory"
            )

            # Extract training vectors
            train_vectors, _ = next(
                read_embeddings(
                    embeddings_file_paths,
                    file_format=file_format,
                    embedding_column_name=embedding_column_name,
                    batch_size=train_size,
                    verbose=True,
                )
            )

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

    memory_available_for_adding = cast_bytes_to_memory_string(
        cast_memory_to_bytes(current_memory_available) - metadata.estimated_index_size_in_bytes()
    )

    print(
        f"The memory available for adding the vectors is {memory_available_for_adding}"
        "(total available - used by the index)"
    )
    print("Will be using at most 1GB of ram for adding")
    # Add the vectors to the index.
    with Timeit("-> Adding the vectors to the index", indent=2):
        batch_size = get_optimal_batch_size(vec_dim, memory_available_for_adding)
        print(
            f"Using a batch size of {batch_size} (memory overhead {cast_bytes_to_memory_string(batch_size*vec_dim*4)})"
        )

        if make_direct_map:
            # Retrieve the embedded index if we are in an IndexPreTransform state
            if isinstance(index, faiss.swigfaiss.IndexPreTransform):
                embedded_index = extract_index_ivf(index)
            else:
                embedded_index = index

            # Make direct map is only implemented for IndexIVF and IndexBinaryIVF, see built file faiss/swigfaiss.py
            if isinstance(embedded_index, (faiss.swigfaiss.IndexIVF, faiss.swigfaiss.IndexBinaryIVF)):
                embedded_index.make_direct_map()
        if distributed is None:
            for batch_id, (vec_batch, ids_batch) in enumerate(
                read_embeddings(
                    embeddings_file_paths,
                    file_format=file_format,
                    embedding_column_name=embedding_column_name,
                    id_columns=id_columns,
                    batch_size=batch_size,
                    verbose=True,
                )
            ):
                index.add(vec_batch)
                if embedding_ids_df_handler:
                    embedding_ids_df_handler(ids_batch, batch_id)
        elif distributed == "pyspark":
            index = run(
                faiss_index=index,
                embedding_column_name=embedding_column_name,
                file_counts=file_counts,  # type: ignore
                id_columns=id_columns,
                file_format=file_format,
                embeddings_file_paths=embeddings_file_paths,
                batch_size=batch_size,
                embedding_ids_df_handler=embedding_ids_df_handler,
                temporary_indices_folder=temporary_indices_folder,
            )
        else:
            raise ValueError(f'Distributed by {distributed} is not supported, only "pyspark" is supported')
    # Give standard values for index hyperparameters if possible.
    if any(re.findall(r"OPQ\d+_\d+,IVF\d+_HNSW\d+,PQ\d+", index_key)):
        set_search_hyperparameters(index, f"nprobe={64},efSearch={128},ht={2048}", use_gpu)
    # return the index.
    return index
