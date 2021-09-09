""" Functions that read efficiently files stored on disk """

import os
import re
from typing import Iterator, Optional

import numpy as np
from tqdm import tqdm as tq


def read_shapes_local(local_path: str, reg_exp_pattern: str = r".+\.npy") -> Iterator[np.ndarray]:
    """
    Iterate over shapes saved on disk.

    Parameters
    ----------
    local_embeddings_path : str
        Path on local disk of the embedding in numpy format.

    Returns
    -------
    shape_iterator : Iterator[np.ndarray]
        An iterator over shapes.
    """
    reg_exp = re.compile(reg_exp_pattern)
    filenames = os.walk(local_path).__next__()[2]
    filenames = [filename for filename in filenames if reg_exp.match(filename)]
    filenames.sort()
    for file_name in filenames:
        # https://stackoverflow.com/a/48592009
        shape = np.load(f"{local_path}/{file_name}", mmap_mode="r").shape
        yield shape


def read_embeddings_local(
    local_embeddings_path: str, batch_size: Optional[int] = None, verbose=True, reg_exp_pattern: str = r".+\.npy",
) -> Iterator[np.ndarray]:
    """
    Iterate over embeddings arrays saved on disk.
    It is possible to iterate over batchs of files and yield stacked embeddings arrays.

    The implementation adopted here is chosen for memory concern: it is very important
    for autofaiss to avoid using more memory than necessary.
    In particular, for the faiss training, a large embedding array is nessary.
    It is not possible to save it twice in memory.
    This implementation pre-allocate an array of size batch_size and keep it updated during the iteration over files.
    The maximum memory usage is batch_size * dim * 4

    Parameters
    ----------
    local_embeddings_path : str
        Path on local disk of the embedding in numpy format.
    batch_size : int (default None)
        Outputs a maximum of batch_size vectors, the default is the size of the first file
        This parameter is useful when working with many small files.
    verbose : bool
        Print detailed informations if set to True

    Returns
    -------
    embeddings_iterator : Iterator[np.ndarray]
        An iterator over batchs of stacked embedding arrays.
    """
    try:
        first_vector_count, dim = next(read_shapes_local(local_embeddings_path, reg_exp_pattern))
    except StopIteration as err:
        raise Exception("no files to read from") from err

    if batch_size is None:
        batch_size = first_vector_count

    reg_exp = re.compile(reg_exp_pattern)

    filenames = os.walk(local_embeddings_path).__next__()[2]
    filenames = [filename for filename in filenames if reg_exp.match(filename)]
    filenames.sort()
    embeddings_batch = None
    nb_emb_in_batch = 0

    iterator = filenames
    if verbose:
        iterator = tq(list(iterator))

    for file_name in iterator:
        emb = np.load(f"{local_embeddings_path}/{file_name}", mmap_mode="r")
        vec_size = emb.shape[0]
        current_emb_index = 0
        while True:
            left_in_emb = vec_size - current_emb_index
            remaining_to_add = max(batch_size - nb_emb_in_batch, 0)
            adding = min(remaining_to_add, left_in_emb)
            additional = max(left_in_emb - adding, 0)
            if embeddings_batch is None:
                embeddings_batch = np.empty((batch_size, dim), "float32")
            embeddings_batch[nb_emb_in_batch : (nb_emb_in_batch + adding), :] = emb[
                current_emb_index : (current_emb_index + adding), :
            ]
            nb_emb_in_batch += adding
            current_emb_index += adding
            if nb_emb_in_batch == batch_size:
                yield embeddings_batch
                nb_emb_in_batch = 0
                embeddings_batch = None
            if additional == 0:
                break

    if nb_emb_in_batch > 0 and embeddings_batch is not None:
        yield embeddings_batch[:nb_emb_in_batch]
