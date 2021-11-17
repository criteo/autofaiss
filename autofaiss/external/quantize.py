""" main file to create an index from the the begining """

import logging
import os
from pprint import pprint as pp
from typing import Dict, Optional, Union
import multiprocessing

import faiss

import fire
import fsspec

from autofaiss.datasets.readers.embeddings_iterators import read_total_nb_vectors_and_dim
from autofaiss.external.build import (
    build_index,
    estimate_memory_required_for_index_creation,
    get_estimated_construction_time_infos,
)
from autofaiss.external.optimize import get_optimal_hyperparameters, get_optimal_index_keys_v2
from autofaiss.external.scores import compute_fast_metrics, compute_medium_metrics
from autofaiss.indices.index_utils import set_search_hyperparameters, load_index_from_hdfs, save_index_on_hdfs
from autofaiss.utils.decorators import Timeit
from autofaiss.utils.cast import cast_memory_to_bytes, cast_bytes_to_memory_string


class Quantizer:
    """class defining the quantization pipeline"""

    def __init__(self):
        """Empty constructor"""

    def quantize(
        self,
        embeddings_path: str,
        output_path: str,
        file_format: str = "npy",
        embedding_column_name: str = "embeddings",
        index_key: Optional[str] = None,
        index_param: Optional[str] = None,
        max_index_query_time_ms: float = 10.0,
        max_index_memory_usage: str = "16G",
        current_memory_available: str = "32G",
        use_gpu: bool = False,
        metric_type: str = "ip",
        nb_cores: Optional[int] = None,
    ) -> str:
        """
        Reads embeddings and creates a quantized index from them.
        The index is stored on the current machine at the given ouput path.

        Parameters
        ----------
        embeddings_path : str
            Local path containing all preprocessed vectors and cached files.
            Files will be added if empty.
        output_path: str
            Destination path of the quantized model on local machine.
        index_key: Optinal(str)
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
        """

        current_bytes = cast_memory_to_bytes(current_memory_available)
        max_index_bytes = cast_memory_to_bytes(max_index_memory_usage)
        memory_left = current_bytes - max_index_bytes

        if memory_left < current_bytes * 0.1:
            print(
                "You do not have enough memory to build this index"
                "please increase current_memory_available or decrease max_index_memory_usage"
            )
            return "Not enough memory"

        if nb_cores is None:
            nb_cores = multiprocessing.cpu_count()
        print(f"Using {nb_cores} omp threads (processes), consider increasing --nb_cores if you have more")
        faiss.omp_set_num_threads(nb_cores)

        with Timeit("Launching the whole pipeline"):

            nb_vectors, vec_dim = read_total_nb_vectors_and_dim(
                embeddings_path, file_format=file_format, embedding_column_name=embedding_column_name
            )
            print(f"There are {nb_vectors} embeddings of dim {vec_dim}")

            with Timeit("Compute estimated construction time of the index", indent=1):
                print(get_estimated_construction_time_infos(nb_vectors, vec_dim, indent=2))

            with Timeit("Checking that your have enough memory available to create the index", indent=1):
                necessary_mem, index_key_used = estimate_memory_required_for_index_creation(
                    nb_vectors, vec_dim, index_key, max_index_memory_usage
                )
                print(
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
                    print(r)
                    return r

            if index_key is None:
                with Timeit("Selecting most promising index types given data characteristics", indent=1):
                    best_index_keys = get_optimal_index_keys_v2(nb_vectors, vec_dim, max_index_memory_usage)
                    if not best_index_keys:
                        return "Constraint on memory too high, no index can be that small"
                    index_key = best_index_keys[0]

            with Timeit("Creating the index", indent=1):
                index = build_index(
                    embeddings_path,
                    index_key,
                    metric_type,
                    nb_vectors,
                    current_memory_available,
                    use_gpu=use_gpu,
                    file_format=file_format,
                    embedding_column_name=embedding_column_name,
                )

            if index_param is None:
                with Timeit("Computing best hyperparameters", indent=1):
                    index_param = get_optimal_hyperparameters(index, index_key, max_speed=max_index_query_time_ms)

            # Set search hyperparameters for the index
            set_search_hyperparameters(index, index_param, use_gpu)
            print(f"The best hyperparameters are: {index_param}")

            with Timeit("Saving the index on local disk", indent=1):
                fs, _ = fsspec.core.url_to_fs(output_path)
                fs.makedirs(output_path, exist_ok=True)
                index_name = f"{index_key}-{index_param}.index"
                faiss.write_index(index, f"{output_path}/{index_name}")

            metric_infos: Dict[str, Union[str, float, int]] = {}

            with Timeit("Compute fast metrics", indent=1):
                metric_infos.update(
                    compute_fast_metrics(
                        embeddings_path, index, file_format=file_format, embedding_column_name=embedding_column_name
                    )
                )

            print("Recap:")
            pp(metric_infos)

        return "Done"

    def tuning(
        self,
        index_path: str,
        index_key: str,
        index_param: Optional[str] = None,
        dest_path: Optional[str] = None,
        is_local_index_path: bool = False,
        max_index_query_time_ms: float = 10.0,
        use_gpu: bool = False,
    ) -> str:
        """
        Set hyperparameters to the given index.

        If an index_param is given, set this hyperparameters to the index,
        otherwise perform a greedy heusistic to make the best out or the max_index_query_time_ms constraint

        Parameters
        ----------
        index_path : str
            Path to .index file on local disk if is_local_index_path is True,
            otherwise path on hdfs.
        index_key: str
            String to give to the index factory in order to create the index.
        index_param: Optional(str)
            Optional string with hyperparameters to set to the index.
            If None, the hyper-parameters are chosen based on an heuristic.
        dest_path: Optional[str]
            Path to the newly created .index file. On local disk if is_local_index_path is True,
            otherwise on hdfs. If None id given, index_path is the destination path.
        is_local_index_path: bool
            True if the dest_path and index_path are local path, False if there are hdfs paths.
        max_index_query_time_ms: float
            Query speed constraint for the index to create.
        use_gpu: bool
            Experimental, gpu training is faster, not tested so far.

        Returns
        -------
        infos: str
            Message describing the index created.
        """

        if dest_path is None:
            dest_path = index_path

        if is_local_index_path:
            with Timeit("Loading index from local disk"):
                index = faiss.read_index(index_path)
        else:
            with Timeit("Loading index from HDFS"):
                index, _ = load_index_from_hdfs(index_path)

        if index_param is None:

            with Timeit("Compute best hyperparameters"):
                index_param = get_optimal_hyperparameters(index, index_key, max_speed=max_index_query_time_ms)

        with Timeit("Set search hyperparameters for the index"):
            set_search_hyperparameters(index, index_param, use_gpu)

        if is_local_index_path:
            with Timeit("Saving index on local disk"):
                faiss.write_index(index, dest_path)
        else:
            with Timeit("Saving index to HDFS"):
                save_index_on_hdfs(index, dest_path)

        return f"The optimal hyperparameters are {index_param}, the index with these parameters has been saved."

    def score_index(
        self,
        index_path: str,
        embeddings_path: str,
        is_local_index_path: bool = False,
        current_memory_available: str = "32G",
    ) -> None:
        """
        Compute metrics on a given index, use cached ground truth for fast scoring the next times.

        Parameters
        ----------
        index_path : str
            Path to .index file on local disk if is_local_index_path is True,
            otherwise path on hdfs.
        embeddings_path: str
            Local path containing all preprocessed vectors and cached files.
        is_local_index_path: bool
            True if the dest_path and index_path are local path, False if there are hdfs paths.
        current_memory_available: str
            Memory available on the current machine, having more memory is a boost
            because it reduces the swipe between RAM and disk.
        """
        faiss.omp_set_num_threads(multiprocessing.cpu_count())

        if is_local_index_path:
            with Timeit("Loading index from local disk"):
                index_memory = os.path.getsize(index_path)
                index = faiss.read_index(index_path)
        else:
            with Timeit("Loading index from HDFS"):
                index, index_memory = load_index_from_hdfs(index_path)

        infos: Dict[str, Union[str, float, int]] = {}

        with Timeit("Compute fast metrics"):
            infos.update(compute_fast_metrics(embeddings_path, index))

        print("Intermediate recap:")
        pp(infos)

        current_in_bytes = cast_memory_to_bytes(current_memory_available)
        memory_left = current_in_bytes - index_memory

        if memory_left < current_in_bytes * 0.1:
            print(
                f"Not enough memory, at least {cast_bytes_to_memory_string(index_memory*1.1)}"
                "is needed, please increase current_memory_available"
            )
            return

        with Timeit("Compute medium metrics"):
            infos.update(compute_medium_metrics(embeddings_path, index, memory_left))

        print("Performances recap:")
        pp(infos)


def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO)

    fire.Fire(Quantizer)


if __name__ == "__main__":
    main()
