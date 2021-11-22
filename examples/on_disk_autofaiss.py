from autofaiss import build_index

build_index(
    embeddings_or_path="embeddings",
    index_path="knn.index",
    index_infos_path="infos.json",
    max_index_memory_usage="4G",
    current_memory_available="4G",
)
