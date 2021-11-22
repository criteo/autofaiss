""" useful functions to apply on an index """

import os
import time
from itertools import chain, repeat
from pathlib import Path
from typing import Dict, Optional, Union
import random

import faiss
import numpy as np


def get_index_size(index: faiss.Index) -> int:
    """Returns the size in RAM of a given index"""

    index_file = "/tmp/tmp_index" + str(random.randrange(100000)) + ".idx"

    if os.path.exists(index_file):
        os.remove(index_file)

    faiss.write_index(index, index_file)

    size_in_bytes = Path(index_file).stat().st_size
    os.remove(index_file)

    return size_in_bytes


def speed_test_ms_per_query(
    index: faiss.Index, query: Optional[np.ndarray] = None, ksearch: int = 40, timout_s: Union[float, int] = 5.0
) -> float:
    """Evaluate the average speed in milliseconds of the index without using batch"""

    nb_samples = 2_000

    if query is None:
        query = np.random.rand(nb_samples, index.d).astype("float32")

    count = 0
    nb_repeat = 1 + (nb_samples - 1) // query.shape[0]

    start_time = time.perf_counter()

    for one_query in chain.from_iterable(repeat(query, nb_repeat)):

        _, _ = index.search(np.expand_dims(one_query, 0), ksearch)

        count += 1

        if time.perf_counter() - start_time > timout_s:
            break

    return (time.perf_counter() - start_time) / count * 1000.0


def search_speed_test(
    index: faiss.Index, query: Optional[np.ndarray] = None, ksearch: int = 40, timout_s: Union[float, int] = 10.0
) -> Dict[str, float]:
    """return the average and 99p search speed"""

    nb_samples = 2_000

    if query is None:
        query = np.random.rand(nb_samples, index.d).astype("float32")

    test_start_time_s = time.perf_counter()
    speed_list_ms = []  # in milliseconds

    nb_repeat = 1 + (nb_samples - 1) // query.shape[0]

    for one_query in chain.from_iterable(repeat(query, nb_repeat)):

        start_time_s = time.perf_counter()  # high precision
        _, _ = index.search(np.expand_dims(one_query, 0), ksearch)
        end_time_s = time.perf_counter()

        search_time_ms = 1000.0 * (end_time_s - start_time_s)
        speed_list_ms.append(search_time_ms)

        if time.perf_counter() - test_start_time_s > timout_s:
            break

    speed_list_ms = np.array(speed_list_ms)
    print(len(speed_list_ms))

    # avg2 = 1000 * (time.perf_counter() - test_start_time_s) / len(speed_list_ms)

    speed_infos = {
        "avg_search_speed_ms": np.average(speed_list_ms),
        "99p_search_speed_ms": np.quantile(speed_list_ms, 0.99),
    }

    return speed_infos


def format_speed_ms_per_query(speed: float) -> str:
    """format the speed (ms/query) into a nice string"""
    return f"{speed:.2f} ms/query"


def quantize_vec_without_modifying_index(index: faiss.Index, vecs: np.ndarray) -> np.ndarray:
    """qantize a batch of vectors"""
    quantized_vecs = index.sa_decode(index.sa_encode(vecs))
    return quantized_vecs


def set_search_hyperparameters(index: faiss.Index, param_str: str, use_gpu: bool = False) -> None:
    """set hyperparameters to an index"""
    # depends on installed faiss version # pylint: disable=no-member
    params = faiss.ParameterSpace() if not use_gpu else faiss.GpuParameterSpace()
    params.set_index_parameters(index, param_str)
