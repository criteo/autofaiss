"""
Given a partitioned dataset of embeddings, create an index per partition
"""

from autofaiss import build_partitioned_indexes
from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel

from pyspark import SparkConf, SparkContext


def create_spark_session():
    # PEX file packaging your Python environment and accessible on yarn by all executors
    os.environ["PYSPARK_PYTHON"] = "/home/ubuntu/autofaiss.pex"
    spark = (
        SparkSession.builder.config("spark.submit.deployMode", "client")
        .config("spark.executorEnv.PEX_ROOT", "./.pex")
        .config("spark.task.cpus", "16")
        .config("spark.driver.port", "5678")
        .config("spark.driver.blockManager.port", "6678")
        .config("spark.driver.host", "172.31.35.188")
        .config("spark.driver.bindAddress", "172.31.35.188")
        .config("spark.executor.memory", "18G")  # make sure to increase this if you're using more cores per executor
        .config("spark.executor.memoryOverhead", "8G")
        .config("spark.task.maxFailures", "100")
        .appName("Partitioned indexes")
        .getOrCreate()
    )
    return spark


spark = create_spark_session()

partitions = [
    "/root/directory/to/partitions/A",
    "/root/directory/to/partitions/B",
    "/root/directory/to/partitions/C",
    "/root/directory/to/partitions/D",
    ...,
]

# Parameter `big_index_threshold` is used to to define the minimum size of a big index.
# Partitions with >= `big_index_threshold` embeddings will be created in a distributed
# way and resulting index will be split into `nb_splits_per_big_index` smaller indexes.
# Partitions with less than `big_index_threshold` embeddings will not be created in a
# distributed way and resulting index will be composed of only one index.
index_metrics = build_partitioned_indexes(
    partitions=partitions,
    output_root_dir="/output/root/directory",
    embedding_column_name="embedding",
    nb_splits_per_big_index=2,
    big_index_threshold=5_000_000,
)
