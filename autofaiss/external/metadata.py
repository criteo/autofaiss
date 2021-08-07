"""
Index metadata for Faiss indices.
"""

import re
from enum import Enum
from math import ceil, log2

from autofaiss.utils.cast import cast_bytes_to_memory_string
from autofaiss.external.descriptions import (
    INDEX_DESCRIPTION_BLOCKS,
    IndexBlock,
    TUNABLE_PARAMETERS_DESCRIPTION_BLOCKS,
    TunableParam,
)


class IndexType(Enum):
    FLAT = 0
    HNSW = 1
    OPQ_IVF_PQ = 2
    OPQ_IVF_HNSW_PQ = 3
    PAD_IVF_HNSW_PQ = 4
    NOT_SUPPORTED = 5


class IndexMetadata:
    """
    Class to compute index metadata given the index_key, the number of vectors and their dimension.

    Note: We don't create classes for each index type in order to keep the code simple.
    """

    def __init__(self, index_key: str, nb_vectors: int, dim_vector: int):

        self.index_key = index_key
        self.nb_vectors = nb_vectors
        self.dim_vector = dim_vector
        self.fast_description = ""
        self.description_blocs = []
        self.tunable_params = []
        self.params = {}

        params = [int(x) for x in re.findall(r"\d+", index_key)]

        if any(re.findall(r"OPQ\d+_\d+,IVF\d+,PQ\d+", index_key)):
            self.index_type = IndexType.OPQ_IVF_PQ
            self.fast_description = "An inverted file index (IVF) with quantization and OPQ preprocessing."
            self.description_blocs = [IndexBlock.IVF, IndexBlock.PQ, IndexBlock.OPQ]
            self.tunable_params = [TunableParam.NPROBE, TunableParam.HT]

            self.params["pq"] = params[3]
            self.params["nbits"] = params[4] if len(params) == 5 else 8  # default value
            self.params["ncentroids"] = params[2]
            self.params["out_d"] = params[1]
            self.params["M_OPQ"] = params[0]

        elif any(re.findall(r"OPQ\d+_\d+,IVF\d+_HNSW\d+,PQ\d+", index_key)):
            self.index_type = IndexType.OPQ_IVF_HNSW_PQ
            self.fast_description = "An inverted file index (IVF) with quantization, OPQ preprocessing, and HNSW index."
            self.description_blocs = [IndexBlock.IVF_HNSW, IndexBlock.HNSW, IndexBlock.PQ, IndexBlock.OPQ]
            self.tunable_params = [TunableParam.NPROBE, TunableParam.EFSEARCH, TunableParam.HT]

            self.params["M_HNSW"] = params[3]
            self.params["pq"] = params[4]
            self.params["nbits"] = params[5] if len(params) == 6 else 8  # default value
            self.params["ncentroids"] = params[2]
            self.params["out_d"] = params[1]
            self.params["M_OPQ"] = params[0]

        elif any(re.findall(r"Pad\d+,IVF\d+_HNSW\d+,PQ\d+", index_key)):
            self.index_type = IndexType.PAD_IVF_HNSW_PQ
            self.fast_description = (
                "An inverted file index (IVF) with quantization, a padding on input vectors, and HNSW index."
            )
            self.description_blocs = [IndexBlock.IVF_HNSW, IndexBlock.HNSW, IndexBlock.PQ, IndexBlock.PAD]
            self.tunable_params = [TunableParam.NPROBE, TunableParam.EFSEARCH, TunableParam.HT]

            self.params["out_d"] = params[0]
            self.params["M_HNSW"] = params[2]
            self.params["pq"] = params[3]
            self.params["nbits"] = params[4] if len(params) == 5 else 8  # default value
            self.params["ncentroids"] = params[1]

        elif any(re.findall(r"HNSW\d+", index_key)):
            self.index_type = IndexType.HNSW
            self.fast_description = "An HNSW index."
            self.description_blocs = [IndexBlock.HNSW]
            self.tunable_params = [TunableParam.EFSEARCH]

            self.params["M_HNSW"] = params[0]

        elif index_key == "Flat":
            self.index_type = IndexType.FLAT
            self.fast_description = "A simple flat index."
            self.description_blocs = [IndexBlock.FLAT]
            self.tunable_params = []

        else:
            self.index_type = IndexType.NOT_SUPPORTED
            self.fast_description = "No description for this index, feel free to contribute :)"
            self.description_blocs = []
            self.tunable_params = []

    def get_index_type(self) -> IndexType:
        """
        return the index type.
        """
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

        if self.index_type in [IndexType.OPQ_IVF_PQ, IndexType.OPQ_IVF_HNSW_PQ, IndexType.PAD_IVF_HNSW_PQ]:
            # We neglict the size of the OPQ table for the moment.
            code_size = ceil(self.params["pq"] * self.params["nbits"] / 8)
            cluster_size_byte = 1 + int((log2(self.params["ncentroids"]) - 1) // 8)
            vector_size_byte = code_size + cluster_size_byte

            vectors_size_in_bytes = self.nb_vectors * vector_size_byte
            centroid_size_in_bytes = self.params["ncentroids"] * self.dim_vector * 4

            total_size_in_byte = vectors_size_in_bytes + centroid_size_in_bytes

            if self.index_type in [IndexType.OPQ_IVF_HNSW_PQ, IndexType.PAD_IVF_HNSW_PQ]:
                total_size_in_byte += self.params["ncentroids"] * self.params["M_HNSW"] * 2 * 4

            if self.index_type in [IndexType.OPQ_IVF_PQ, IndexType.OPQ_IVF_HNSW_PQ]:
                total_size_in_byte += self.params["M_OPQ"] * self.params["out_d"] * 4

            return total_size_in_byte

        return -1

    def get_index_description(self, tunable_parameters_infos=False) -> str:
        """
        Gives a generic description of the index.
        """

        description = self.fast_description

        if self.index_type == IndexType.NOT_SUPPORTED:
            return description

        description += "\n"
        index_size_string = cast_bytes_to_memory_string(self.estimated_index_size_in_bytes())
        description += f"The size of the index should be around {index_size_string}.\n\n"
        description += "\n".join(INDEX_DESCRIPTION_BLOCKS[desc] for desc in self.description_blocs) + "\n\n"

        if tunable_parameters_infos:
            if not self.tunable_params:
                description += "No parameters can be tuned to find a query speed VS recall tradeoff\n\n"
            else:
                description += "List of parameters that can be tuned to find a query speed VS recall tradeoff:\n"
                description += (
                    "\n".join(TUNABLE_PARAMETERS_DESCRIPTION_BLOCKS[desc] for desc in self.tunable_params) + "\n\n"
                )

        description += """
For all indices except the flat index, the query speed can be adjusted.
The lower the speed limit the lower the recall. With a looser constraint
on the query time, the recall can be higher, but it is limited by the index
structure (if there is quantization for instance).
"""
        return description
