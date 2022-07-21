""" useful functions to apply on an index """

import os
import time
from functools import partial
from itertools import chain, repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Optional, Union, List, Tuple
import logging

from faiss import extract_index_ivf
import faiss
import fsspec
import numpy as np

logger = logging.getLogger("autofaiss")


def get_index_size(index: faiss.Index) -> int:
    """Returns the size in RAM of a given index"""
    with NamedTemporaryFile() as tmp_file:
        faiss.write_index(index, tmp_file.name)
        size_in_bytes = Path(tmp_file.name).stat().st_size

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

    speed_list_ms2 = np.array(speed_list_ms)

    # avg2 = 1000 * (time.perf_counter() - test_start_time_s) / len(speed_list_ms)

    speed_infos = {
        "avg_search_speed_ms": np.average(speed_list_ms2),
        "99p_search_speed_ms": np.quantile(speed_list_ms2, 0.99),
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


def get_index_from_bytes(index_bytes: Union[bytearray, bytes]) -> faiss.Index:
    """Transforms a bytearray containing a faiss index into the corresponding object."""

    with NamedTemporaryFile(delete=False) as output_file:
        output_file.write(index_bytes)
        tmp_name = output_file.name

    b = faiss.read_index(tmp_name)
    os.remove(tmp_name)
    return b


def get_bytes_from_index(index: faiss.Index) -> bytearray:
    """Transforms a faiss index into a bytearray."""

    with NamedTemporaryFile(delete=False) as output_file:
        faiss.write_index(index, output_file.name)
        tmp_name = output_file.name

    with open(tmp_name, "rb") as index_file:
        b = index_file.read()
        os.remove(tmp_name)
        return bytearray(b)


def parallel_download_indices_from_remote(
    fs: fsspec.AbstractFileSystem, indices_file_paths: List[str], dst_folder: str
):
    """Download small indices in parallel."""

    def _download_one(src_dst_path: Tuple[str, str], fs: fsspec.AbstractFileSystem):
        src_path, dst_path = src_dst_path
        try:
            fs.get(src_path, dst_path)
        except Exception as e:
            raise Exception(f"Failed to download {src_path} to {dst_path}") from e

    if len(indices_file_paths) == 0:
        return
    os.makedirs(dst_folder, exist_ok=True)
    dst_paths = [os.path.join(dst_folder, os.path.split(p)[-1]) for p in indices_file_paths]
    src_dest_paths = zip(indices_file_paths, dst_paths)
    with ThreadPool(min(16, len(indices_file_paths))) as pool:
        for _ in pool.imap_unordered(partial(_download_one, fs=fs), src_dest_paths):
            pass


def initialize_direct_map(index: faiss.Index) -> None:
    nested_index = extract_index_ivf(index) if isinstance(index, faiss.swigfaiss.IndexPreTransform) else index

    # Make direct map is only implemented for IndexIVF and IndexBinaryIVF, see built file faiss/swigfaiss.py
    if isinstance(nested_index, (faiss.swigfaiss.IndexIVF, faiss.swigfaiss.IndexBinaryIVF)):
        nested_index.make_direct_map()


def save_index(index: faiss.Index, root_dir: str, index_filename: str) -> str:
    """Save index"""
    fs = fsspec.core.url_to_fs(root_dir, use_listings_cache=False)[0]
    fs.mkdirs(root_dir, exist_ok=True)
    output_index_path = os.path.join(root_dir, index_filename)
    with fsspec.open(output_index_path, "wb").open() as f:
        faiss.write_index(index, faiss.PyCallbackIOWriter(f.write))
    return output_index_path


def load_index(index_src_path: str, index_dst_path: str) -> faiss.Index:
    fs = fsspec.core.url_to_fs(index_src_path, use_listings_cache=False)[0]
    try:
        fs.get(index_src_path, index_dst_path)
    except Exception as e:
        raise Exception(f"Failed to download index from {index_src_path} to {index_dst_path}") from e
    return faiss.read_index(index_dst_path)
