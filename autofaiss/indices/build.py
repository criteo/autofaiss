""" Common functions to build an index """

import logging
from typing import Dict, Optional, Callable
import uuid
import re
import os

import fsspec
import faiss
import pandas as pd
from embedding_reader import EmbeddingReader

from autofaiss.external.optimize import optimize_and_measure_index
from autofaiss.indices.index_utils import set_search_hyperparameters
from autofaiss.utils.path import make_path_absolute


logger = logging.getLogger("autofaiss")


def get_write_ids_df_to_parquet_fn(ids_root_dir: str) -> Callable[[pd.DataFrame, int], None]:
    """Create function to write ids from Pandas dataframe to parquet"""

    def _write_ids_df_to_parquet_fn(ids: pd.DataFrame, batch_id: int):
        filename = f"part-{batch_id:08d}-{uuid.uuid1()}.parquet"
        output_file = os.path.join(ids_root_dir, filename)  # type: ignore
        with fsspec.open(output_file, "wb") as f:
            logger.debug(f"Writing id DataFrame to file {output_file}")
            ids.to_parquet(f, index=False)

    return _write_ids_df_to_parquet_fn


def get_optimize_index_fn(
    embedding_reader: EmbeddingReader,
    index_key: str,
    index_path: Optional[str],
    index_infos_path: Optional[str],
    use_gpu: bool,
    save_on_disk: bool,
    max_index_query_time_ms: float,
    min_nearest_neighbors_to_retrieve: int,
    index_param: Optional[str] = None,
) -> Callable[[faiss.Index, str], Dict]:
    """Create function to optimize index by choosing best hyperparameters and calculating metrics"""

    def _optimize_index_fn(index: faiss.Index, index_suffix: str):
        cur_index_path = make_path_absolute(index_path) + index_suffix if index_path else None
        cur_index_infos_path = make_path_absolute(index_infos_path) + index_suffix if index_infos_path else None
        if any(re.findall(r"OPQ\d+_\d+,IVF\d+_HNSW\d+,PQ\d+", index_key)):
            set_search_hyperparameters(index, f"nprobe={64},efSearch={128},ht={2048}", use_gpu)
        metric_infos = optimize_and_measure_index(
            embedding_reader,
            index,
            cur_index_infos_path,
            index_key,
            index_param,
            cur_index_path,
            max_index_query_time_ms=max_index_query_time_ms,
            min_nearest_neighbors_to_retrieve=min_nearest_neighbors_to_retrieve,
            save_on_disk=save_on_disk,
            use_gpu=use_gpu,
        )
        return metric_infos

    return _optimize_index_fn