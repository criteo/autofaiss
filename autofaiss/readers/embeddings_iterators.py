""" Functions that read efficiently files stored on disk """
import logging
import os
from multiprocessing.pool import ThreadPool
from typing import Iterator, Optional, Tuple

import numpy as np
from tqdm import tqdm as tq
import fsspec
import pyarrow.parquet as pq
import re

LOGGER = logging.getLogger(__name__)


def get_shape_numpy(f):
    first_line = f.readline()
    result = re.search(r"'shape': \(([0-9]+), ([0-9]+)\)", str(first_line))
    return (int(result.group(1)), int(result.group(2)))


def read_first_file_shape(
    embeddings_path: str, file_format: str, embedding_column_name: Optional[str] = None
) -> Tuple[int, int]:
    """
    Read the shape of the first file in the embeddings directory.

    Parameters
    ----------
    embeddings_path : str
        Path of the embeddings directory.
    file_format : str
        Format of the embeddings file.
    embedding_column_name: str
        If file_format="parquet" - the name of the column containing the embeddings

    Returns
    -------
    shape : (int, int)
        Shape of the first embedding file.
    """
    fs, path_in_fs = fsspec.core.url_to_fs(embeddings_path)
    LOGGER.debug(f"Using filesystem {fs} for {embeddings_path}")
    filenames = fs.walk(path_in_fs).__next__()[2]
    filenames = [filename for filename in filenames if filename.endswith(f".{file_format}")]
    filenames.sort()
    LOGGER.debug(f"Found files in path {embeddings_path}: ")
    LOGGER.debug("\n".join(f"\t{f}" for f in filenames))

    first_file = filenames[0]
    first_file_path = os.path.join(embeddings_path, first_file)
    LOGGER.info(f"First file in path {embeddings_path} = {first_file_path}")
    if file_format == "npy":
        LOGGER.info(f"Opening numpy file {first_file_path}")
        with fs.open(first_file_path, "rb") as f:
            shape = get_shape_numpy(f)
    elif file_format == "parquet":
        LOGGER.info(f"Opening parquet file {first_file_path} and getting column {embedding_column_name}")
        with fs.open(first_file_path, "rb") as f:
            emb = pq.read_table(f).to_pandas()
            embeddings_raw = emb[embedding_column_name].to_numpy()
            embeddings = np.stack(embeddings_raw).astype("float32")
            shape = embeddings.shape
    else:
        raise ValueError("Unknown file format")
    return shape


def read_total_nb_vectors_and_dim(
    embeddings_path: str, file_format: str = "npy", embedding_column_name: str = "embeddings"
) -> Tuple[int, int]:
    """
        Get the count and dim of embeddings.
        Parameters
        ----------
        embeddings_path : str
            Path of the embedding in numpy or parquet format.
        file_format : str (default "npy")

        Returns
        -------
        (count, dim) : (int, int)
            count: total number of vectors in the dataset.
            dim: embedding dimension
        """
    fs, path_in_fs = fsspec.core.url_to_fs(embeddings_path)
    filenames = fs.walk(path_in_fs).__next__()[2]
    filenames = [filename for filename in filenames if filename.endswith(f".{file_format}")]
    filenames.sort()

    _, dim = read_first_file_shape(
        embeddings_path, file_format=file_format, embedding_column_name=embedding_column_name
    )

    def get_number_of_lines(filename, fs):
        file_path = os.path.join(embeddings_path, filename)
        if file_format == "npy":
            with fs.open(file_path, "rb") as f:
                shape = get_shape_numpy(f)
                return shape[0]
        elif file_format == "parquet":
            with fs.open(file_path) as file:
                parquet_file = pq.ParquetFile(file, memory_map=True)
                return parquet_file.metadata.num_rows
        else:
            raise ValueError("Unknown file format")

    count = 0
    i = 0
    with ThreadPool(50) as p:
        for c in p.imap_unordered(lambda f: get_number_of_lines(f, fs), filenames):
            count += c
            i += 1

    return count, dim


def read_embeddings(
    embeddings_path: str,
    batch_size: Optional[int] = None,
    verbose=True,
    file_format="npy",
    embedding_column_name="embeddings",
) -> Iterator[np.ndarray]:
    """
    Iterate over embeddings arrays.
    It is possible to iterate over batchs of files and yield stacked embeddings arrays.

    The implementation adopted here is chosen for memory concern: it is very important
    for autofaiss to avoid using more memory than necessary.
    In particular, for the faiss training, a large embedding array is necessary.
    It is not possible to save it twice in memory.
    This implementation pre-allocate an array of size batch_size and keep it updated during the iteration over files.
    The maximum memory usage is batch_size * dim * 4

    Parameters
    ----------
    embeddings_path : str
        Path on local disk of the embedding in numpy format.
    batch_size : int (default None)
        Outputs a maximum of batch_size vectors, the default is the size of the first file
        This parameter is useful when working with many small files.
    file_format : str (default "npy")
        Format of the embedding files.
        npy or parquet
    verbose : bool
        Print detailed informations if set to True
    embedding_column_name: str
        If file_format="parquet" - the name of the column containing the embeddings

    Returns
    -------
    embeddings_iterator : Iterator[np.ndarray]
        An iterator over batches of stacked embedding arrays.
    """
    try:
        first_vector_count, dim = read_first_file_shape(
            embeddings_path, file_format, embedding_column_name=embedding_column_name
        )
    except StopIteration as err:
        raise Exception("no files to read from") from err

    if batch_size is None:
        batch_size = first_vector_count

    fs, path_in_fs = fsspec.core.url_to_fs(embeddings_path)

    filenames = fs.walk(path_in_fs).__next__()[2]
    filenames = [filename for filename in filenames if filename.endswith(f".{file_format}")]
    filenames.sort()
    embeddings_batch = None
    nb_emb_in_batch = 0

    iterator = filenames
    if verbose:
        iterator = tq(list(iterator))

    for filename in iterator:
        file_path = os.path.join(embeddings_path, filename)
        with fs.open(file_path, "rb") as f:
            if file_format == "npy":
                emb = np.load(f)
            elif file_format == "parquet":
                emb_table = pq.read_table(f).to_pandas()
                embeddings_raw = emb_table[embedding_column_name].to_numpy()
                emb = np.stack(embeddings_raw)
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
