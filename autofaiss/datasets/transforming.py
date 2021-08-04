""" functions that transform objects from the datasets """

import os
import re
from functools import partial
from multiprocessing import Pool
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm as tq

from autofaiss.datasets.readers.remote_iterators import read_filenames
from autofaiss.utils.os_tools import list_local_files


def parquet_already_transformed(remote_embeddings_path: str, local_embeddings_path: str) -> bool:
    """ Checks if embeddings were all transformed in .npy arrays or should be downloaded again """

    remote_filenames = read_filenames(remote_embeddings_path)
    nb_remote_parquet_files = len(remote_filenames)

    local_filenames = list_local_files(local_embeddings_path)
    reg_exp_pattern = r"(emb|map)_\d+\.npy"
    reg_exp = re.compile(reg_exp_pattern)
    nb_local_map_and_emb_files = len([1 for filename in local_filenames if reg_exp.match(filename)])

    return 2 * nb_remote_parquet_files == nb_local_map_and_emb_files


def convert_parquet_to_numpy(
    parquet_file: str,
    embeddings_path: str,
    embedding_column_name: str,
    keys_path: Optional[str] = None,
    key_column_name: Optional[float] = None,
) -> None:
    """ Convert one embedding parquet file to an embedding numpy file """

    emb = None
    if not os.path.exists(embeddings_path):
        emb = pq.read_table(parquet_file).to_pandas()
        embeddings_raw = emb[embedding_column_name].to_numpy()
        embeddings = np.stack(embeddings_raw).astype("float32")
        np.save(embeddings_path, embeddings)

    if keys_path is not None and not os.path.exists(keys_path):
        if emb is None:
            emb = pq.read_table(parquet_file).to_pandas()
        key_raw = emb[key_column_name].to_numpy()
        np.save(keys_path, key_raw)


def run_one(
    parquet_file: str,
    embeddings_folder: str,
    delete: bool,
    embedding_column_name: str,
    keys_folder: Optional[float] = None,
    key_column_name: Optional[float] = None,
) -> None:
    """ Convertion function to call for parallel execution """
    num = parquet_file.split("/")[-1].split("-")[1]

    convert_parquet_to_numpy(
        parquet_file,
        f"{embeddings_folder}/emb_{num}.npy",
        embedding_column_name,
        f"{keys_folder}/key_{num}.npy",
        key_column_name,
    )

    if delete:
        os.remove(parquet_file)


def convert_all_parquet_to_numpy(
    parquet_folder: str,
    embeddings_folder: str,
    n_cores: int = 32,
    delete: bool = False,
    embedding_column_name: str = "embedding",
    keys_folder: Optional[str] = None,
    key_column_name: Optional[str] = None,
) -> None:
    """ Convert embedding parquet files to an embedding numpy files
    Optionally also extract keys from parquet files
    """

    assert n_cores > 0
    os.makedirs(embeddings_folder, exist_ok=True)
    if keys_folder:
        os.makedirs(keys_folder, exist_ok=True)

    parquet_files = [f"{parquet_folder}/{x}" for x in os.listdir(parquet_folder) if x.endswith(".parquet")]
    parquet_files.sort()

    nb_files = len(parquet_files)

    func = partial(
        run_one,
        embeddings_folder=embeddings_folder,
        delete=delete,
        embedding_column_name=embedding_column_name,
        keys_folder=keys_folder,
        key_column_name=key_column_name,
    )

    with tq(total=nb_files) as pbar:
        with Pool(processes=n_cores) as pool:
            for _ in pool.imap_unordered(func, parquet_files):
                pbar.update(1)
