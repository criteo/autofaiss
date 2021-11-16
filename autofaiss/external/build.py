""" gather functions necessary to build an index """

import re
from typing import Optional, Tuple, Union

import faiss
from autofaiss.external.metadata import IndexMetadata
from autofaiss.datasets.readers.local_iterators import read_embeddings_local, read_shapes_local
from autofaiss.datasets.readers.remote_iterators import read_embeddings_remote, read_filenames
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


def estimate_memory_required_for_index_creation(
    nb_vectors: int, vec_dim: int, index_key: Optional[str] = None, max_index_memory_usage: Optional[str] = None
) -> Tuple[int, str]:
    """
    Estimates the RAM necessary to create the index
    The value returned is in Bytes
    """

    if index_key is None:
        if max_index_memory_usage is not None:
            index_key = get_optimal_index_keys_v2(nb_vectors, vec_dim, max_index_memory_usage)[0]
        else:
            raise ValueError("you should give max_index_memory_usage value if no index_key is given")

    metadata = IndexMetadata(index_key, nb_vectors, vec_dim)

    index_memory = metadata.estimated_index_size_in_bytes()
    needed_for_adding = min(index_memory * 0.1, 10 ** 9)
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


def get_estimated_download_time_infos(
    embeddings_hdfs_path: str, bandwidth_gbytes_per_sec: float = 1.0, indent: int = 0
) -> Tuple[str, Tuple[int, int]]:
    """
    Gives a general approximation of the download time (and preprocessing time) of embeddings
    """
    nb_vectors_approx, vec_dim = get_nb_vectors_approx_and_dim_from_hdfs(embeddings_hdfs_path)

    size = 4 * nb_vectors_approx * vec_dim

    download = 1.1 * size / (bandwidth_gbytes_per_sec * 1024 ** 3)  # seconds
    preprocess = 1.6 * download  # seconds

    infos = (
        f"-> Download: {to_readable_time(download, rounding=True)}\n"
        f"-> Preprocess: {to_readable_time(preprocess, rounding=True)}\n"
        f"Total: {to_readable_time(download + preprocess, rounding=True)}"
        " (< 1 minute if files are already cached)"
    )

    tab = "\t" * indent
    infos = tab + infos.replace("\n", "\n" + tab)

    return infos, (nb_vectors_approx, vec_dim)


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


def get_nb_vectors_approx_and_dim_from_hdfs(parquet_embeddings_path: str) -> Tuple[int, int]:
    """legacy function to give the dimensions of a parquet file
    Still useful for tests"""

    # Get information for one partition
    avg_batch_length, vec_dim = next(read_embeddings_remote(parquet_embeddings_path, verbose=False)).shape

    # Count the number of files
    nb_files = len(read_filenames(parquet_embeddings_path))

    nb_vectors_approx = nb_files * avg_batch_length

    return nb_vectors_approx, vec_dim


def get_nb_vectors_and_dim(embeddings_path: str) -> Tuple[int, int]:
    """
    Function that gives the total shape of the embeddings array
    """

    tot_vec = 0
    vec_dim = -1

    for shape in read_shapes_local(embeddings_path):
        batch_length, dim = shape
        tot_vec += batch_length
        vec_dim = dim

    return tot_vec, vec_dim


def build_index(
    embeddings_path: str,
    index_key: str,
    metric_type: Union[str, int],
    nb_vectors: int,
    current_memory_available: str,
    use_gpu: bool = False,
):
    """
    Function that returns an index on the numpy arrays stored on disk in the embeddings_path path.
    """

    # Instanciate the index
    with Timeit(f"-> Instanciate the index {index_key}", indent=2):

        # Convert metric_type to faiss type
        metric_type = to_faiss_metric_type(metric_type)

        # Get information for one partition
        _, vec_dim = next(read_shapes(embeddings_path))

        # Instanciate the index
        index = index_factory(vec_dim, index_key, metric_type)

    metadata = IndexMetadata(index_key, nb_vectors, vec_dim)

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
            train_vectors = next(read_embeddings(embeddings_path, batch_size=train_size, verbose=True))

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
        for vec_batch in read_embeddings(embeddings_path, batch_size=batch_size, verbose=True):
            index.add(vec_batch)

    # Give standard values for index hyperparameters if possible.
    if any(re.findall(r"OPQ\d+_\d+,IVF\d+_HNSW\d+,PQ\d+", index_key)):
        set_search_hyperparameters(index, f"nprobe={64},efSearch={128},ht={2048}", use_gpu)

    # return the index.
    return index
