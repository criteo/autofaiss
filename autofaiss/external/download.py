"""
Function that downloads a parquet file containing embeddings on the
current machine in the right format (several .npy files)
"""
import logging
from autofaiss.datasets.readers.downloader import download
from typing import Optional

from autofaiss.external.build import get_estimated_download_time_infos
from autofaiss.datasets.transforming import convert_all_parquet_to_numpy
from autofaiss.utils.decorators import Timeit


def download_from_hdfs(
    embeddings_hdfs_path: str,
    local_save_path: str,
    n_cores: int = 10,
    delete_tmp_files: bool = True,
    bandwidth_gbytes_per_sec: Optional[float] = None,
    verbose: bool = True,
    embedding_column_name: str = "embedding",
    keys_folder: Optional[str] = None,
    key_column_name: Optional[str] = None,
) -> None:
    """
    Download a file containing embeddings in parquet format on hdfs and
    convert it to several .npy files
    Optionally save keys in another folder (useful to keep track of what is each embedding)
    """
    if bandwidth_gbytes_per_sec is not None:
        infos, _ = get_estimated_download_time_infos(embeddings_hdfs_path, bandwidth_gbytes_per_sec)
        print(infos)

    with Timeit("Download the parquet files on local disk", indent=1):
        download(embeddings_hdfs_path, dest_path=local_save_path, n_cores=n_cores, verbose=verbose)

    with Timeit("Convert .parquet files to numpy arrays", indent=1):
        convert_all_parquet_to_numpy(
            local_save_path,
            local_save_path,
            delete=delete_tmp_files,
            embedding_column_name=embedding_column_name,
            n_cores=n_cores,
            keys_folder=keys_folder,
            key_column_name=key_column_name,
        )


def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO)
    fire.Fire(download_from_hdfs)


if __name__ == "__main__":
    main()
