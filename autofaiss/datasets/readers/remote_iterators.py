""" functions that read efficiently files from hdfs without saving it on disk """

from typing import Iterator, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm as tq


def read_filenames(parquet_embeddings_path: str) -> List[str]:
    """
    Read the name of all the parquet files

    Parameters
    ----------
    parquet_embeddings_path: str
        Path to .parquet embedding files on hdfs

    Returns
    -------
    filenames: List[str]
        The list of the paths to .parquet files of the given directory sorted in
        non decreasing order.
    """

    hdfs = pa.hdfs.connect()

    all_filenames = hdfs.ls(parquet_embeddings_path)
    filenames = ["hdfs:" + path.split(":")[-1] for path in all_filenames if path.endswith(".parquet")]
    filenames.sort()

    return filenames


def read_embeddings_remote(
    embeddings_path: str, column_label: str = "embedding", stack_input: int = 1, verbose=True
) -> Iterator[np.ndarray]:
    """
    Return an iterator over embeddings from a parquet folder

    Parameters
    ----------
    embeddings_path : str
        Path on the hdfs of the embedding in parquet format.
    column_label : str (default "embeddings")
        Name of the column in which the embeddings are stored.
    stack_input : int (default 1)
        Number of arrays that should be stacked at each iterations.
        This parameter is useful when working with many small files.
    verbose : bool
        Print detailed informations if set to True

    Returns
    -------
    embeddings_iterator : Iterator[np.ndarray]
        An iterator over batchs of embedding arrays.
    """

    assert stack_input > 0

    filenames = read_filenames(embeddings_path)

    embeddings_stack: List[np.ndarray] = []

    iterator = list(enumerate(filenames))
    if verbose:
        iterator = tq(iterator)

    for file_number, file_name in iterator:

        if embeddings_stack and (file_number % stack_input == 0):
            yield np.concatenate(embeddings_stack)
            embeddings_stack = []

        small_table = pq.read_table(file_name)
        pandas_df = small_table[column_label].to_pandas()
        embeddings_stack.append(np.stack(pandas_df).astype("float32"))

    if embeddings_stack:
        yield np.concatenate(embeddings_stack)
