"""
File that contains the descriptions of the different indices features.
"""

from enum import Enum


class IndexBlock(Enum):
    FLAT = 0
    IVF = 1
    IVF_HNSW = 2
    HNSW = 3
    PQ = 4
    OPQ = 5
    PAD = 6


class TunableParam(Enum):
    EFSEARCH = 0
    NPROBE = 1
    HT = 2


INDEX_DESCRIPTION_BLOCKS = {
    IndexBlock.FLAT: """Flat Index (Simple exhaustive search)
    All the vectors are stored without compression in a 2D array.
    The search is exhaustive and gives exact results, but it can be slow when there are
    many vectors.
    This index can't be memory-mapped on disk in current faiss implementation, but and IVF
    index with just one cluster makes it possible to memory-map it.""",
    IndexBlock.IVF: """IVF (Inverted File Index):
    The vector space is divided into several groups that are represented by a cluster vector.
    The algorithm first looks for the closest cluster vectors by iterating over them one by one.
    Then, The IVF mapping is used to read the vectors in the selected groups. The algorithm reads
    each of them and select the k nearest neighbors.
    It is possible to memory-map this index on disk, it reduces the RAM footprint to the size of
    its cluster vectors, but the index is slower in that case.
    -> More info: https://hal.inria.fr/inria-00514462/PDF/jegou_pq_postprint.pdf""",
    IndexBlock.IVF_HNSW: """IVF + HNSW (Inverted File Index + Hierarchical Navigable Small World index):
    The vector space is divided into several groups that are represented by a cluster vector. 
    The algorithm first looks for the closest cluster vectors using an HNSW index (instead of 
    iterating over them one by one in the version without HNSW). Then, The IVF mapping is used
    to read the vectors in the selected groups. The algorithm reads each of them and selects
    the k nearest neighbors.
    It is possible to memory-map this index on disk, it reduces the RAM footprint to the size of the
    hnsw index, but the index is slower in that case.
    -> More info: https://hal.inria.fr/inria-00514462/PDF/jegou_pq_postprint.pdf""",
    IndexBlock.PQ: """PQ (Product Quantization):
    The vectors are compressed using product quantization.
    Some clever optimizations are implemented to compute distances in the compressed space.
    -> More info: https://hal.inria.fr/inria-00514462/PDF/jegou_pq_postprint.pdf""",
    IndexBlock.OPQ: """OPQ (Optimized Product Quantization):
    The vectors are projected using a rotation matrix. The matrix is trained to minimize
    the average compression error.
    -> More info: http://kaiminghe.com/publications/pami13opq.pdf""",
    IndexBlock.HNSW: """HNSW (Hierarchical Navigable Small World):
    All the vectors are stored in a 2D array without compression, and
    a bidirectional graph is built on the top of the vectors to enable a fast search.
    This index can't be memory-mapped on disk.
    -> More info: https://arxiv.org/ftp/arxiv/papers/1603/1603.09320.pdf""",
    IndexBlock.PAD: """PAD (Padding):
    Preprocessing operations where a padding with 0s is added to the end of the vectors.""",
}


TUNABLE_PARAMETERS_DESCRIPTION_BLOCKS = {
    TunableParam.NPROBE: """    - nprobe: The number of vector groups to explore, the search time is proportional
      to this value. If nprobe is high, the recall will also be high.
      -> More info: https://hal.inria.fr/inria-00514462/PDF/jegou_pq_postprint.pdf""",
    TunableParam.EFSEARCH: """    - efsearch: The number of times a greedy search is done in the HNSW graph. The results
      of the different greedy searches are combined to get a more precise result.
      The search time is proportional to this value. If efsearch is high, the recall will be high.
      -> More info: https://arxiv.org/ftp/arxiv/papers/1603/1603.09320.pdf       """,
    TunableParam.HT: """    - ht (Hamming threshold): A threshold value to approximate the vector distances using the hamming distance.
      Computing the Hamming distances between two vectors is much faster than computing the real distance,
      but it is also less precise. We found out that it was better to deactivate this parameter as it
      decreases the recall score over speed improvements (default value is 2048)
      -> More info: https://arxiv.org/pdf/1609.01882.pdf""",
}
