"""
Index metadata for Faiss indices.
"""

import re
from enum import Enum
from math import ceil, log2

from autofaiss.utils.cast import cast_bytes_to_memory_string


class IndexType(Enum):
    FLAT = 0
    HNSW = 1
    OPQ_IVF = 2
    OPQ_IVF_HNSW = 3
    PAD_IVF_HNSW = 4
    NOT_SUPPORTED = 5


class Metadata:
    """
    Class to compute index metadata given the index_key, the number of vectors and their dimension.
    """

    def __init__(self, index_key: str, nb_vectors: int, dim_vector: int):

        self.index_key = index_key
        self.nb_vectors = nb_vectors
        self.dim_vector = dim_vector
        self.params = {}

        params = [int(x) for x in re.findall(r"\d+", index_key)]

        if any(re.findall(r"OPQ\d+_\d+,IVF\d+,PQ\d+", index_key)):
            self.index_type = IndexType.OPQ_IVF
            self.params["pq"] = params[3]
            self.params["nbits"] = params[4] if len(params) == 5 else 8  # default value
            self.params["ncentroids"] = params[2]
            self.params["out_d"] = params[1]
            self.params["M_OPQ"] = params[0]

        elif any(re.findall(r"OPQ\d+_\d+,IVF\d+_HNSW\d+,PQ\d+", index_key)):
            self.index_type = IndexType.OPQ_IVF_HNSW
            self.params["M_HNSW"] = params[3]
            self.params["pq"] = params[4]
            self.params["nbits"] = params[5] if len(params) == 6 else 8  # default value
            self.params["ncentroids"] = params[2]
            self.params["out_d"] = params[1]
            self.params["M_OPQ"] = params[0]

        elif any(re.findall(r"Pad\d+,IVF\d+_HNSW\d+,PQ\d+", index_key)):
            self.index_type = IndexType.PAD_IVF_HNSW
            self.params["out_d"] = params[0]
            self.params["M_HNSW"] = params[2]
            self.params["pq"] = params[3]
            self.params["nbits"] = params[4] if len(params) == 5 else 8  # default value
            self.params["ncentroids"] = params[1]

        elif any(re.findall(r"HNSW\d+", index_key)):
            self.index_type = IndexType.HNSW
            self.params["M_HNSW"] = params[0]
        elif index_key == "Flat":
            self.index_type = IndexType.FLAT
        else:
            self.index_type = IndexType.NOT_SUPPORTED

    def get_index_type(self) -> IndexType:
        return self.index_type

    def estimated_index_size_in_bytes(self) -> int:
        """
        Compute the estimated size of the index in bytes.
        """

        if self.index_type == IndexType.FLAT:
            return self.nb_vectors * self.dim_vector * 4

        if self.index_type == IndexType.HNSW:
            # M bidirectional links per vector in the HNSW graph
            hnsw_graph_in_bytes = self.nb_vectors * self.params["M_HNSW"] * 2 * 4
            vectors_size_in_bytes = self.nb_vectors * self.dim_vector * 4
            return vectors_size_in_bytes + hnsw_graph_in_bytes

        if self.index_type in [IndexType.OPQ_IVF, IndexType.OPQ_IVF_HNSW, IndexType.PAD_IVF_HNSW]:
            # We neglict the size of the OPQ table for the moment.
            code_size = ceil(self.params["pq"] * self.params["nbits"] / 8)
            cluster_size_byte = 1 + int((log2(self.params["ncentroids"]) - 1) // 8)
            vector_size_byte = code_size + cluster_size_byte

            vectors_size_in_bytes = self.nb_vectors * vector_size_byte
            centroid_size_in_bytes = self.params["ncentroids"] * self.dim_vector * 4

            total_size_in_byte = vectors_size_in_bytes + centroid_size_in_bytes

            if self.index_type in [IndexType.OPQ_IVF_HNSW, IndexType.PAD_IVF_HNSW]:
                total_size_in_byte += self.params["ncentroids"] * self.params["M_HNSW"] * 2 * 4

            return total_size_in_byte

        return -1

    def get_index_description(self) -> str:
        """
        Give a description of the index.
        """

        description = ""

        if self.index_type == IndexType.FLAT:
            description += "An index that stores all the vectors in a simple array without any compression\n"
        else:
            description += "#TODO"

        index_size_string = cast_bytes_to_memory_string(self.estimated_index_size_in_bytes())
        description += f"The size of the index should be around {index_size_string}."

        return description
