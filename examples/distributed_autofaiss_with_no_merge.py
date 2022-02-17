"""
An example of running autofaiss by pyspark to produce N indices.

You need to install pyspark before using the following example.
"""
from typing import Dict

from autofaiss import build_index

# You'd better create a spark session before calling build_index,
# otherwise, a spark session would be created by autofaiss with the least configuration.

index_path2_metric_infos: Dict[str, Dict] = build_index(
    embeddings="hdfs://root/path/to/your/embeddings/folder",
    distributed="pyspark",
    file_format="parquet",
    temporary_indices_folder="hdfs://root/tmp/distributed_autofaiss_indices",
    current_memory_available="10G",
    max_index_memory_usage="90G",
    nb_indices_to_keep=10,
)
