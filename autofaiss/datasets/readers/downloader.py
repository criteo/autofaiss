""" Functions to download and transform the parquet files to numpy files """

import os
from itertools import repeat
from multiprocessing import Pool
from typing import Tuple

from tqdm import tqdm as tq

from autofaiss.datasets.readers.remote_iterators import read_filenames
from autofaiss.utils.os_tools import run_command


def download(parquet_embeddings_path: str, dest_path: str, n_cores: int = 32, verbose: bool = True) -> bool:
    """
    Download .parquet files from hdfs at max speed
    Parallelisation is essential to use the full bandwidth.
    """

    filenames = read_filenames(parquet_embeddings_path)

    nb_files = len(filenames)

    os.makedirs(dest_path, exist_ok=True)

    src_dest_paths = zip(filenames, repeat(dest_path))

    if n_cores == 1:

        if verbose:
            src_dest_paths = tq(list(src_dest_paths))

        for src_dest_path in src_dest_paths:
            download_one(src_dest_path)

    else:

        with tq(total=nb_files) as pbar:
            with Pool(processes=n_cores) as pool:
                for _ in pool.imap_unordered(download_one, src_dest_paths):
                    pbar.update(1)

    return True


def download_one(src_dest_path: Tuple[str, str]) -> None:
    """ Function to download one file from hdfs to local"""

    filename, dest_path = src_dest_path

    if not os.path.exists(f"{dest_path}/{filename.split('/')[-1]}"):
        cmd = f"hdfs dfs -copyToLocal {filename} {dest_path}"
        _ = run_command(cmd)
