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
from abc import ABC


LOGGER = logging.getLogger(__name__)


class AbstractMatrixReader(ABC):
    """Read a file and provide its shape, row count and ndarray. Behaves as a context manager"""

    def __init__(self, fs, file_path):
        self.f = fs.open(file_path, "rb")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()


class NumpyEagerNdArray:
    def __init__(self, f):
        self.n = np.load(f)
        self.shape = self.n.shape
        self.num_rows = self.shape[0]

    def get_rows(self, start, end):
        return self.n[start:end, :]


def read_numpy_header(f):
    f.seek(0)
    first_line = f.readline()
    result = re.search(r"'shape': \(([0-9]+), ([0-9]+)\)", str(first_line))
    shape = (int(result.group(1)), int(result.group(2)))
    dtype = re.search(r"'descr': '([<f0-9]+)'", str(first_line)).group(1)
    return (shape, dtype, f.tell())


class NumpyLazyNdArray:
    """Reads a numpy file lazily"""

    def __init__(self, f):
        self.f = f
        (self.shape, self.dtype, self.header_offset) = read_numpy_header(f)
        self.byteperitem = np.dtype(self.dtype).itemsize * self.shape[1]
        self.num_rows = self.shape[0]

    def get_rows(self, start, end):
        length = end - start
        self.f.seek(self.header_offset + start * self.byteperitem)
        return np.frombuffer(self.f.read(length * self.byteperitem), dtype=self.dtype).reshape((length, self.shape[1]))


class NumpyMatrixReader(AbstractMatrixReader):
    """Read a numpy file and provide its shape, row count and ndarray. Behaves as a context manager"""

    def __init__(self, fs, file_path, *_):
        super().__init__(fs, file_path)

    def get_shape(self):
        shape, _, _ = read_numpy_header(self.f)
        return shape

    def get_row_count(self):
        return self.get_shape()[0]

    def get_eager_ndarray(self):
        return NumpyEagerNdArray(self.f)

    def get_lazy_ndarray(self):
        return NumpyLazyNdArray(self.f)


class ParquetEagerNdArray:
    def __init__(self, f, embedding_column_name):
        emb_table = pq.read_table(f).to_pandas()
        embeddings_raw = emb_table[embedding_column_name].to_numpy()
        self.n = np.stack(embeddings_raw)
        self.num_rows = self.self.n.shape[0]

    def get_rows(self, start, end):
        return self.n[start:end, :]


class ParquetLazyNdArray:
    def __init__(self, f, embedding_column_name):
        self.table = pq.read_table(f)
        self.num_rows = self.table.num_rows
        self.embedding_column_name = embedding_column_name

    def get_rows(self, start, end):
        embeddings_raw = self.table.slice(start, end - start)[self.embedding_column_name].to_numpy()
        return np.stack(embeddings_raw)


class ParquetMatrixReader(AbstractMatrixReader):
    """Read a parquet file and provide its shape, row count and ndarray. Behaves as a context manager"""

    def __init__(self, fs, file_path, embedding_column_name):
        super().__init__(fs, file_path)
        self.embedding_column_name = embedding_column_name

    def get_shape(self):
        parquet_file = pq.ParquetFile(self.f, memory_map=True)
        num_rows = parquet_file.metadata.num_rows
        batches = parquet_file.iter_batches(batch_size=1, columns=(self.embedding_column_name,))
        dimension = next(batches).to_pandas()[self.embedding_column_name].to_numpy()[0].shape[0]
        return (num_rows, dimension)

    def get_row_count(self):
        parquet_file = pq.ParquetFile(self.f, memory_map=True)
        return parquet_file.metadata.num_rows

    def get_eager_ndarray(self):
        return ParquetEagerNdArray(self.f, self.embedding_column_name)

    def get_lazy_ndarray(self):
        return ParquetLazyNdArray(self.f, self.embedding_column_name)


matrix_readers_registry = {"parquet": ParquetMatrixReader, "npy": NumpyMatrixReader}


def get_matrix_reader(file_format, fs, file_path, *args):
    cls = matrix_readers_registry[file_format]
    return cls(fs, file_path, *args)


def get_file_list(path, file_format):
    fs, path_in_fs = fsspec.core.url_to_fs(path)
    filenames = fs.walk(path_in_fs).__next__()[2]
    filenames = [filename for filename in filenames if filename.endswith(f".{file_format}")]
    filenames.sort()
    return fs, filenames


def read_first_file_shape(
    embeddings_path: str, file_format: str, embedding_column_name: Optional[str] = None
) -> Tuple[int, int]:
    """
    Read the shape of the first file in the embeddings directory.
    """
    fs, filenames = get_file_list(embeddings_path, file_format)

    first_file = filenames[0]
    first_file_path = os.path.join(embeddings_path, first_file)
    with get_matrix_reader(file_format, fs, first_file_path, embedding_column_name) as matrix_reader:
        return matrix_reader.get_shape()


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
    fs, filenames = get_file_list(embeddings_path, file_format)

    _, dim = read_first_file_shape(
        embeddings_path, file_format=file_format, embedding_column_name=embedding_column_name
    )

    def file_to_line_count(f):
        with get_matrix_reader(
            file_format, fs, os.path.join(embeddings_path, f), embedding_column_name
        ) as matrix_reader:
            return matrix_reader.get_row_count()

    count = 0
    i = 0
    with ThreadPool(50) as p:
        for c in p.imap_unordered(file_to_line_count, filenames):
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

    fs, filenames = get_file_list(embeddings_path, file_format)
    embeddings_batch = None
    nb_emb_in_batch = 0

    iterator = filenames
    if verbose:
        iterator = tq(list(iterator))

    for filename in iterator:
        file_path = os.path.join(embeddings_path, filename)
        with get_matrix_reader(
            file_format, fs, os.path.join(embeddings_path, file_path), embedding_column_name
        ) as matrix_reader:
            emb = matrix_reader.get_lazy_ndarray()
            vec_size = emb.num_rows
            current_emb_index = 0
            while True:
                left_in_emb = vec_size - current_emb_index
                remaining_to_add = max(batch_size - nb_emb_in_batch, 0)
                adding = min(remaining_to_add, left_in_emb)
                additional = max(left_in_emb - adding, 0)
                if embeddings_batch is None:
                    embeddings_batch = np.empty((batch_size, dim), "float32")
                embeddings_batch[nb_emb_in_batch : (nb_emb_in_batch + adding), :] = emb.get_rows(
                    current_emb_index, (current_emb_index + adding)
                )
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
