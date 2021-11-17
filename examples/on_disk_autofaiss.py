from autofaiss import quantize

quantize(
    embeddings_path="embeddings",
    output_path="my_index_folder",
    max_index_memory_usage="4G",
    current_memory_available="4G",
)
