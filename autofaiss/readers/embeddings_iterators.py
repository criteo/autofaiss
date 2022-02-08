""" Functions that read efficiently files stored on disk """
import logging
from multiprocessing.pool import ThreadPool
from typing import Iterator, Optional, Tuple, List, Union
import re
from abc import ABC

import pandas as pd
import numpy as np
from tqdm import tqdm as tq
import fsspec
import os
import pyarrow.parquet as pq


LOGGER = logging.getLogger(__name__)


class AbstractArray:
    num_rows: int

    def get_rows(self, start: int, end: int) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
        pass


class AbstractMatrixReader(ABC):
    """Read a file and provide its shape, row count and ndarray. Behaves as a context manager"""

    def __init__(self, fs: fsspec.AbstractFileSystem, file_path: str):
        self.f = fs.open(file_path, "rb")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()

    def get_row_count(self) -> int:
        pass

    def get_shape(self) -> Tuple[int, int]:
        pass

    def get_lazy_array(self) -> AbstractArray:
        pass

    def get_eager_array(self) -> AbstractArray:
        pass


class NumpyEagerNdArray(AbstractArray):
    def __init__(self, f: Union[str, fsspec.core.OpenFile]):
        self.n = np.load(f)
        self.shape = self.n.shape
        self.num_rows = self.shape[0]

    def get_rows(self, start: int, end: int) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
        return self.n[start:end, :], None


def read_numpy_header(f):
    f.seek(0)
    file_size = f.size if isinstance(f.size, int) else f.size()
    first_line = f.read(min(file_size, 300)).split(b"\n")[0]
    result = re.search(r"'shape': \(([0-9]+), ([0-9]+)\)", str(first_line))
    shape = (int(result.group(1)), int(result.group(2)))
    dtype = re.search(r"'descr': '([<f0-9]+)'", str(first_line)).group(1)
    end = len(first_line) + 1  # the first line content and the endline
    f.seek(0)
    return (shape, dtype, end)


class NumpyLazyNdArray(AbstractArray):
    """Reads a numpy file lazily"""

    def __init__(self, f: fsspec.spec.AbstractBufferedFile):
        self.f = f
        (self.shape, self.dtype, self.header_offset) = read_numpy_header(f)
        self.byteperitem = np.dtype(self.dtype).itemsize * self.shape[1]
        self.num_rows = self.shape[0]

    def get_rows(self, start: int, end: int) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
        length = end - start
        self.f.seek(self.header_offset + start * self.byteperitem)
        return (
            np.frombuffer(self.f.read(length * self.byteperitem), dtype=self.dtype).reshape((length, self.shape[1])),
            None,
        )


class NumpyMatrixReader(AbstractMatrixReader):
    """Read a numpy file and provide its shape, row count and ndarray. Behaves as a context manager"""

    def __init__(self, fs: fsspec.AbstractFileSystem, file_path: str, *_):
        super().__init__(fs, file_path)

    def get_shape(self) -> Tuple[int, int]:
        shape, _, _ = read_numpy_header(self.f)
        return shape

    def get_row_count(self) -> int:
        return self.get_shape()[0]

    def get_eager_array(self) -> NumpyEagerNdArray:
        return NumpyEagerNdArray(self.f)

    def get_lazy_array(self) -> NumpyLazyNdArray:
        return NumpyLazyNdArray(self.f)


class ParquetEagerNdArray(AbstractArray):
    def __init__(self, f, embedding_column_name: str, id_columns: Optional[List[str]] = None):
        emb_table = pq.read_table(f).to_pandas()
        embeddings_raw = emb_table[embedding_column_name].to_numpy()
        self.n = np.stack(embeddings_raw)
        self.ids = emb_table[id_columns] if id_columns else None
        self.num_rows = self.n.shape[0]

    def get_rows(self, start, end) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
        return self.n[start:end, :], self.ids.iloc[start:end] if self.ids else None


class ParquetLazyNdArray(AbstractArray):
    """Reads a parquet file lazily"""

    def __init__(self, f, embedding_column_name: str, id_columns: Optional[List[str]] = None):
        self.table = pq.read_table(f)
        self.num_rows = self.table.num_rows
        self.embedding_column_name = embedding_column_name
        self.id_columns = id_columns

    def get_rows(self, start, end) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
        table_slice = self.table.slice(start, end - start)
        embeddings_raw = table_slice[self.embedding_column_name].to_numpy()
        ids = table_slice.select(self.id_columns).to_pandas() if self.id_columns else None
        return np.stack(embeddings_raw), ids


class ParquetMatrixReader(AbstractMatrixReader):
    """Read a parquet file and provide its shape, row count and ndarray. Behaves as a context manager"""

    def __init__(self, fs, file_path, embedding_column_name, id_columns=None):
        super().__init__(fs, file_path)
        self.embedding_column_name = embedding_column_name
        self.id_columns = id_columns

    def get_shape(self) -> Tuple[int, int]:
        parquet_file = pq.ParquetFile(self.f, memory_map=True)
        num_rows = parquet_file.metadata.num_rows
        batches = parquet_file.iter_batches(batch_size=1, columns=[self.embedding_column_name])
        dimension = next(batches).to_pandas()[self.embedding_column_name].to_numpy()[0].shape[0]
        return (num_rows, dimension)

    def get_row_count(self) -> int:
        parquet_file = pq.ParquetFile(self.f, memory_map=True)
        return parquet_file.metadata.num_rows

    def get_eager_array(self) -> ParquetEagerNdArray:
        return ParquetEagerNdArray(self.f, self.embedding_column_name, self.id_columns)

    def get_lazy_array(self) -> ParquetLazyNdArray:
        return ParquetLazyNdArray(self.f, self.embedding_column_name, self.id_columns)


matrix_readers_registry = {"parquet": ParquetMatrixReader, "npy": NumpyMatrixReader}


def get_matrix_reader(file_format: str, fs: fsspec.AbstractFileSystem, file_path: str, *args) -> AbstractMatrixReader:
    cls = matrix_readers_registry[file_format]
    return cls(fs, file_path, *args)


def get_file_list(path: Union[str, List[str]], file_format: str) -> Tuple[fsspec.AbstractFileSystem, List[str]]:
    """
    Get the file system and all the file paths that matches `file_format` under the given `path`.
    The `path` could a single folder or multiple folders.

    :raises ValueError: if file system is inconsistent under different folders.
    """
    if isinstance(path, str):
        return _get_file_list(path, file_format)
    all_file_paths = []
    fs = None
    for p in path:
        cur_fs, file_paths = _get_file_list(p, file_format, sort_result=False)
        if fs is None:
            fs = cur_fs
        elif fs != cur_fs:
            raise ValueError(
                f"The file system in different folder are inconsistent.\n" f"Got one {fs} and the other {cur_fs}"
            )
        all_file_paths.extend(file_paths)
    all_file_paths.sort()
    return fs, all_file_paths


def make_path_absolute(path):
    fs, p = fsspec.core.url_to_fs(path)
    if fs.protocol == "file":
        return os.path.abspath(p)
    return path


def _get_file_list(
    path: str, file_format: str, sort_result: bool = True
) -> Tuple[fsspec.AbstractFileSystem, List[str]]:
    """Get the file system and all the file paths that matches `file_format` given a single path."""
    path = make_path_absolute(path)
    fs, path_in_fs = fsspec.core.url_to_fs(path)
    prefix = path[: path.index(path_in_fs)]
    glob_pattern = path.rstrip("/") + f"**/*.{file_format}"
    file_paths = fs.glob(glob_pattern)
    if sort_result:
        file_paths.sort()
    file_paths_with_prefix = [prefix + file_path for file_path in file_paths]
    return fs, file_paths_with_prefix


def read_first_file_shape(
    embeddings_path: Union[str, List[str]], file_format: str, embedding_column_name: Optional[str] = None
) -> Tuple[int, int]:
    """
    Read the shape of the first file in the embeddings directory.
    """
    fs, file_paths = get_file_list(embeddings_path, file_format)

    first_file_path = file_paths[0]
    return get_file_shape(first_file_path, file_format, embedding_column_name, fs)


def get_file_shape(
    file_path: str, file_format: str, embedding_column_name: Optional[str], fs: fsspec.AbstractFileSystem
):
    with get_matrix_reader(file_format, fs, file_path, embedding_column_name) as matrix_reader:
        return matrix_reader.get_shape()


def read_total_nb_vectors_and_dim(
    embeddings_path: Union[str, List[str]], file_format: str = "npy", embedding_column_name: str = "embeddings"
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
    fs, file_paths = get_file_list(embeddings_path, file_format)

    _, dim = get_file_shape(file_paths[0], file_format=file_format, embedding_column_name=embedding_column_name, fs=fs)

    def file_to_line_count(f):
        with get_matrix_reader(file_format, fs, f, embedding_column_name) as matrix_reader:
            return matrix_reader.get_row_count()

    count = 0
    i = 0
    with tq(total=len(file_paths)) as pbar:
        with ThreadPool(50) as p:
            for c in p.imap_unordered(file_to_line_count, file_paths):
                count += c
                i += 1
                pbar.update(1)

    return count, dim


def read_embeddings(
    embeddings_path: Union[str, List[str]],
    batch_size: Optional[int] = None,
    verbose: bool = True,
    file_format: str = "npy",
    embedding_column_name: str = "embeddings",
    id_columns: Optional[List[str]] = None,
) -> Iterator[Tuple[np.ndarray, Optional[pd.DataFrame]]]:
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
    id_columns: List[str] (default None)
        List of columns containing the ids of the vectors,
        these will be mapped to integer ids between 0 and total_nb_embeddings

    Returns
    -------
    embeddings_iterator : Iterator[Tuple[np.ndarray, Optional[pd.DataFrame]]]
        An iterator over batches of stacked embedding arrays
        and (optionally) a DataFrame containing the mapping between id columns
        and an index in [0, total_nb_embeddings-1] store in a column "i"
    """
    try:
        first_vector_count, dim = read_first_file_shape(
            embeddings_path, file_format, embedding_column_name=embedding_column_name
        )
    except StopIteration as err:
        raise Exception("no files to read from") from err

    if batch_size is None:
        batch_size = first_vector_count

    fs, file_paths = get_file_list(embeddings_path, file_format)
    embeddings_batch = None
    ids_batch = None
    nb_emb_in_batch = 0

    iterator = file_paths
    if verbose:
        iterator = tq(list(iterator))

    total_embeddings_processed = 0
    for file_path in iterator:
        with get_matrix_reader(file_format, fs, file_path, embedding_column_name, id_columns) as matrix_reader:
            array = matrix_reader.get_lazy_array()
            vec_size = array.num_rows
            current_emb_index = 0
            while True:
                left_in_emb = vec_size - current_emb_index
                remaining_to_add = max(batch_size - nb_emb_in_batch, 0)
                adding = min(remaining_to_add, left_in_emb)
                additional = max(left_in_emb - adding, 0)
                if embeddings_batch is None:
                    embeddings_batch = np.empty((batch_size, dim), "float32")
                current_embeddings, ids_df = array.get_rows(current_emb_index, (current_emb_index + adding))
                embeddings_batch[nb_emb_in_batch : (nb_emb_in_batch + adding), :] = current_embeddings

                if id_columns is not None:
                    if ids_batch is None:
                        ids_batch = np.empty((batch_size, len(id_columns)), dtype="object")
                    ids_batch[nb_emb_in_batch : (nb_emb_in_batch + adding), :] = ids_df.to_numpy()

                nb_emb_in_batch += adding
                current_emb_index += adding
                if nb_emb_in_batch == batch_size:
                    if id_columns is not None:
                        ids_batch_df = pd.DataFrame(ids_batch, columns=id_columns).infer_objects()
                        ids_batch_df["i"] = np.arange(
                            start=total_embeddings_processed, stop=total_embeddings_processed + nb_emb_in_batch
                        )
                    else:
                        ids_batch_df = None

                    yield embeddings_batch, ids_batch_df
                    total_embeddings_processed += nb_emb_in_batch
                    nb_emb_in_batch = 0
                    embeddings_batch = None
                    ids_batch = None
                    ids_batch_df = None
                if additional == 0:
                    break

    if nb_emb_in_batch > 0 and embeddings_batch is not None:
        if id_columns is not None:
            ids_batch_df = pd.DataFrame(
                ids_batch[:nb_emb_in_batch, :], columns=id_columns  # type: ignore
            ).infer_objects()
            ids_batch_df["i"] = np.arange(
                start=total_embeddings_processed, stop=total_embeddings_processed + nb_emb_in_batch
            )
        else:
            ids_batch_df = None
        total_embeddings_processed += nb_emb_in_batch
        yield embeddings_batch[:nb_emb_in_batch], ids_batch_df
