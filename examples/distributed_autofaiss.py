"""
An example of running autofaiss by pyspark.

You need to install pyspark before using the following example.
"""
from autofaiss import build_index

# You'd better create a spark session before calling build_index,
# otherwise, a spark session would be created by autofaiss with the least configuration.

index, index_infos = build_index(
    embeddings="hdfs://root/path/to/your/embeddings/folder",
    distributed="pyspark",
    file_format="parquet"
)
