from autofaiss.external.build import estimate_memory_required_for_index_creation

#
# def test_estimate_memory_required_for_index_creation():
#     needed_memory, _ = estimate_memory_required_for_index_creation(
#         nb_vectors=4_000_000_000,
#         vec_dim=512,
#         index_key="OPQ4_28,IVF131072_HNSW32,PQ4x8",
#         max_index_memory_usage="50G",
#     )
#     assert needed_memory == 100
