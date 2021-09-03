""" Functions that read efficiently files stored on disk """

import os
import re
from typing import Iterator, List

import numpy as np
from tqdm import tqdm as tq


def read_embeddings_local(local_embeddings_path: str, stack_input: int = 1, verbose=True) -> Iterator[np.ndarray]:
    """
    Iterate over embeddings arrays saved on disk.
    It is possible to iterate over batchs of files and yield stacked embeddings arrays.

    Parameters
    ----------
    local_embeddings_path : str
        Path on local disk of the embedding in numpy format.
    stack_input : int (default 1)
        Number of arrays that should be stacked at each iterations.
        This parameter is useful when working with many small files.
    verbose : bool
        Print detailed informations if set to True

    Returns
    -------
    embeddings_iterator : Iterator[np.ndarray]
        An iterator over batchs of stacked embedding arrays.
    """
    return read_arrays_local(local_embeddings_path, stack_input=stack_input, verbose=verbose)


def read_arrays_local(
    local_path: str, reg_exp_pattern: str = r".+\.npy", stack_input: int = 1, verbose=True
) -> Iterator[np.ndarray]:
    """
    Iterate over numpy array files that match the reg ex pattern and yield their content.
    It is possible to iterate over the stacked content of several arrays.

    Parameters
    ----------
    local_embeddings_path : str
        Path on local disk of arrays in numpy format.
    stack_input : int (default 1)
        Number of arrays that should be stacked at each iterations.
        This parameter is useful when working with many small files.
    verbose : bool
        Print detailed informations if set to True

    Returns
    -------
    arrays_iterator : Iterator[np.ndarray]
        An iterator over batchs of stacked arrays.
    """

    assert stack_input > 0

    reg_exp = re.compile(reg_exp_pattern)

    filenames = os.walk(local_path).__next__()[2]
    filenames = [filename for filename in filenames if reg_exp.match(filename)]
    filenames.sort()
    embeddings_stack: List[np.ndarray] = []

    iterator = enumerate(filenames)
    if verbose:
        iterator = tq(list(iterator))

    for file_number, file_name in iterator:

        if embeddings_stack and (file_number % stack_input == 0):
            yield np.concatenate(embeddings_stack)
            embeddings_stack = []

        try:
            emb = np.load(f"{local_path}/{file_name}")
            if emb.dtype != "float32":
                emb = emb.astype("float32")
            embeddings_stack.append(emb)
        except Exception as e:  # pylint: disable=broad-except
            print(e)

    if embeddings_stack:
        yield np.concatenate(embeddings_stack).astype(np.float32)
