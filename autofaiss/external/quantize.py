""" main file to create an index from the the begining """

import logging
import os
from pprint import pprint as pp
from typing import Dict, Optional, Union

import faiss

import fire
from autofaiss.datasets.readers.local_iterators import read_embeddings_local
from autofaiss.external.build import (
    build_index,
    estimate_memory_required_for_index_creation,
    get_estimated_construction_time_infos,
    get_nb_vectors_and_dim,
)
from autofaiss.external.optimize import get_optimal_hyperparameters, get_optimal_index_keys_v2
from autofaiss.external.scores import compute_fast_metrics, compute_medium_metrics
from autofaiss.indices.index_utils import set_search_hyperparameters, load_index_from_hdfs
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
        index_key: Optional[str] = None,
        index_param: Optional[str] = None,
        max_index_query_time_ms: float = 10.0,
        max_index_memory_usage: str = "32G",
        current_memory_available: str = "32G",
        use_gpu: bool = False,
        metric_type: str = "ip",
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
        """

        with Timeit("Launching the whole pipeline"):

            nb_vectors, vec_dim = get_nb_vectors_and_dim(embeddings_path)

            with Timeit("Compute estimated construction time of the index", indent=1):
                print(get_estimated_construction_time_infos(nb_vectors, vec_dim, indent=2))

            with Timeit("Checking that your have enough memory available to create the index", indent=1):
                necessary_mem, index_key_used = estimate_memory_required_for_index_creation(
                    nb_vectors, vec_dim, index_key, max_index_memory_usage
                )

                prefix = "(default) " if index_key is None else ""

                if 0.9 * necessary_mem > cast_memory_to_bytes(current_memory_available):

                    return (
                        f"The current memory available on your machine ({current_memory_available}) is not "
                        f"enough to create the {prefix}index {index_key_used} that requires "
                        f"{cast_bytes_to_memory_string(necessary_mem)} to train. "
                        "You can decrease the number of clusters of you index since the Kmeans algorithm "
                        "used for clusterisation is responsible for this high memory usage."
                    )

            if index_key is None:
                with Timeit("Selecting most promising index types given data characteristics", indent=1):
                    _, vec_dim = next(read_embeddings_local(embeddings_path, verbose=False)).shape
                    best_index_keys = get_optimal_index_keys_v2(nb_vectors, vec_dim, max_index_memory_usage)
                    if not best_index_keys:
                        return "Constraint on memory too high, no index can be that small"
                    index_key = best_index_keys[0]

            with Timeit("Creating the index", indent=1):
                index = build_index(
                    embeddings_path, index_key, metric_type, nb_vectors, current_memory_available, use_gpu
                )

            if index_param is None:
                with Timeit("Computing best hyperparameters", indent=1):
                    index_param = get_optimal_hyperparameters(index, index_key, max_speed=max_index_query_time_ms)

            # Set search hyperparameters for the index
            set_search_hyperparameters(index, index_param, use_gpu)
            print(f"The best hyperparameters are: {index_param}")

            # Save the index
            index_name = f"{index_key}-{index_param}.index"

            with Timeit("Saving the index on local disk", indent=1):
                os.makedirs(output_path, exist_ok=True)
                index_name = f"{index_key}-{index_param}.index"
                faiss.write_index(index, f"{output_path}/{index_name}")

            metric_infos: Dict[str, Union[str, float, int]] = {}

            with Timeit("Compute fast metrics", indent=1):
                metric_infos.update(compute_fast_metrics(embeddings_path, index))

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
                index = load_index_from_hdfs(index_path)

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
                load_index_from_hdfs(dest_path)

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

        if is_local_index_path:
            with Timeit("Loading index from local disk"):
                index = faiss.read_index(index_path)
        else:
            with Timeit("Loading index from HDFS"):
                index = load_index_from_hdfs(index_path)

        infos: Dict[str, Union[str, float, int]] = {}

        with Timeit("Compute fast metrics"):
            infos.update(compute_fast_metrics(embeddings_path, index))

        print("Intermediate recap:")
        pp(infos)

        with Timeit("Compute medium metrics"):
            infos.update(compute_medium_metrics(embeddings_path, index, current_memory_available))

        print("Performances recap:")
        pp(infos)


def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO)

    fire.Fire(Quantizer)


if __name__ == "__main__":
    main()
