""" functions that transform objects from the datasets """

import os
import re
from functools import partial
from multiprocessing import Pool

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


def convert_parquet_to_numpy(parquet_file: str, embeddings_path: str, embedding_column_name: str) -> None:
    """ Convert one embedding parquet file to an embedding numpy file """

    if not os.path.exists(embeddings_path):
        emb = pq.read_table(parquet_file).to_pandas()
        embeddings_raw = emb[embedding_column_name].to_numpy()
        embeddings = np.stack(embeddings_raw).astype("float32")
        np.save(embeddings_path, embeddings)


def run_one(parquet_file: str, embeddings_folder: str, delete: bool, embedding_column_name: str) -> None:
    """ Convertion function to call for parallel execution """
    num = parquet_file.split("/")[-1].split("-")[1]

    convert_parquet_to_numpy(parquet_file, f"{embeddings_folder}/emb_{num}.npy", embedding_column_name)

    if delete:
        os.remove(parquet_file)


def convert_all_parquet_to_numpy(
    parquet_folder: str,
    embeddings_folder: str,
    n_cores: int = 32,
    delete: bool = False,
    embedding_column_name: str = "embedding",
) -> None:
    """ Convert embedding parquet files to an embedding numpy files """

    assert n_cores > 0
    os.makedirs(embeddings_folder, exist_ok=True)

    parquet_files = [f"{parquet_folder}/{x}" for x in os.listdir(parquet_folder) if x.endswith(".parquet")]
    parquet_files.sort()

    nb_files = len(parquet_files)

    func = partial(
        run_one, embeddings_folder=embeddings_folder, delete=delete, embedding_column_name=embedding_column_name
    )

    with tq(total=nb_files) as pbar:
        with Pool(processes=n_cores) as pool:
            for _ in pool.imap_unordered(func, parquet_files):
                pbar.update(1)
