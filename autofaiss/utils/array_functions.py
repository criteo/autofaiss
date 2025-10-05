""" useful functions t apply on numpy arrays """

import numpy as np


def sanitize(x):
    return np.ascontiguousarray(x, dtype="float32")


def multi_array_split(array_list, nb_chunk):
    total_length = len(array_list[0])
    chunk_size = (total_length - 1) // nb_chunk + 1
    assert all(len(x) == total_length for x in array_list)
    for i in range(0, total_length, chunk_size):
        yield tuple(x[i : i + chunk_size] for x in array_list)
