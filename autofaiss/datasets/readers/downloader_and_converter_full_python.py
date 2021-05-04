""" functions to download and convert embedding parquet files """

import os

import numpy as np
from tqdm import tqdm as tq

from autofaiss.datasets.readers.remote_iterators import read_filenames


def save(filename: str, dest_path: str) -> None:
    """
    Function that loads and converts an hdfs embeddings .parquet file to
    numpy arrays and store it in the local desination path.
    """

    basename = filename.split("/")[-1].split(".")[0]

    embeddings_array_path = f"{dest_path}/{basename}_embeddings.npy"

    if not os.path.exists(embeddings_array_path):

        # pylint: disable=import-outside-toplevel
        import pyarrow.parquet as pq

        small_table = pq.read_table(filename, use_threads=False)
        pandas_df = small_table.to_pandas()
        embeddings = np.stack(pandas_df["embedding"].to_numpy()).astype("float32")

        with open(embeddings_array_path, "wb") as embedding_file:
            np.save(embedding_file, embeddings)


def download(parquet_embeddings_path, dest_path, n_cores=1, verbose=True) -> bool:
    """
    Download and convert all the embedding parquet file from the hdfs path
    to numpy arrays.
    Parallelisation is not possible since the connection to hdfs is unique (singleton class)
    """

    os.makedirs(dest_path, exist_ok=True)

    filenames = list(reversed(read_filenames(parquet_embeddings_path)))

    if n_cores != 1:
        raise ValueError("multiprocessing is not compatible")

    if verbose:
        filenames = tq(list(filenames))
    for filename in filenames:
        save(filename, dest_path)

    return True


if __name__ == "__main__":

    DEST_PATH = "/home/v.paltz/downloaded_vectors/image_embeddings_100_views_US"
    SRC_PATH = "/user/deepr/dev/r.beaumont/image_embeddings_100_views_US"

    download(SRC_PATH, DEST_PATH)
