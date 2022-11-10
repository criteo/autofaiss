from autofaiss import build_index, update_index


index, _ = build_index(embeddings="/path/to/your/embeddings/folder", index_path="index_v1")
new_index, _ = update_index(embeddings="/path/to/your/new/embeddings/folder", trained_index_or_path=index)
# or you can pass the index path
# new_index, _ = update_index(embeddings="/path/to/your/new/embeddings/folder", trained_index_or_path="index_v1")
