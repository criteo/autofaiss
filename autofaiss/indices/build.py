""" Common functions to build an index """

import logging
from typing import Dict, Optional, Tuple, Union, Callable, Any
import uuid
import re
import os
import tempfile

import fsspec
import faiss
import pandas as pd
from embedding_reader import EmbeddingReader

from autofaiss.external.optimize import optimize_and_measure_index, get_optimal_batch_size
from autofaiss.indices.index_utils import set_search_hyperparameters, initialize_direct_map, load_index
from autofaiss.utils.path import make_path_absolute
from autofaiss.utils.cast import cast_bytes_to_memory_string


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
    make_direct_map: bool,
    index_param: Optional[str],
) -> Callable[[faiss.Index, str], Dict]:
    """Create function to optimize index by choosing best hyperparameters and calculating metrics"""

    def _optimize_index_fn(index: faiss.Index, index_suffix: str):
        if make_direct_map:
            initialize_direct_map(index)

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


def add_embeddings_to_index_local(
    embedding_reader: EmbeddingReader,
    trained_index_or_path: Union[faiss.Index, str],
    memory_available_for_adding: str,
    embedding_ids_df_handler: Optional[Callable[[pd.DataFrame, int], Any]] = None,
    index_optimizer: Callable = None,
    add_embeddings_with_ids: bool = False,
) -> Tuple[Optional[faiss.Index], Optional[Dict[str, str]]]:
    """Add embeddings to index from driver"""

    vec_dim = embedding_reader.dimension
    batch_size = get_optimal_batch_size(vec_dim, memory_available_for_adding)
    logger.info(
        f"Using a batch size of {batch_size} (memory overhead {cast_bytes_to_memory_string(batch_size * vec_dim * 4)})"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        if isinstance(trained_index_or_path, str):
            local_index_path = os.path.join(tmp_dir, "index")
            trained_index = load_index(trained_index_or_path, local_index_path)
        else:
            trained_index = trained_index_or_path
        for batch_id, (vec_batch, ids_batch) in enumerate(embedding_reader(batch_size=batch_size)):
            if add_embeddings_with_ids:
                trained_index.add_with_ids(vec_batch, ids_batch["i"].to_numpy())
            else:
                trained_index.add(vec_batch)
            if embedding_ids_df_handler:
                embedding_ids_df_handler(ids_batch, batch_id)
        metric_infos = index_optimizer(trained_index, "") if index_optimizer else None  # type: ignore
        return trained_index, metric_infos
