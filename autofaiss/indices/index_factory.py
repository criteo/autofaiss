""" functions that fixe faiss index_factory function """
# pylint: disable=invalid-name

import re
from typing import Optional

import faiss


def index_factory(d: int, index_key: str, metric_type: int, ef_construction: Optional[int] = None):
    """
    custom index_factory that fix some issues of
    faiss.index_factory with inner product metrics.
    """

    if metric_type == faiss.METRIC_INNER_PRODUCT:

        # make the index described by the key
        if any(re.findall(r"OPQ\d+_\d+,IVF\d+,PQ\d+", index_key)):
            params = [int(x) for x in re.findall(r"\d+", index_key)]

            cs = params[3]  # code size (in Bytes if nbits=8)
            nbits = params[4] if len(params) == 5 else 8  # default value
            ncentroids = params[2]
            out_d = params[1]
            M_OPQ = params[0]

            quantizer = faiss.index_factory(out_d, "Flat", metric_type)
            assert quantizer.metric_type == metric_type
            index_ivfpq = faiss.IndexIVFPQ(quantizer, out_d, ncentroids, cs, nbits, metric_type)
            assert index_ivfpq.metric_type == metric_type
            index_ivfpq.own_fields = True
            quantizer.this.disown()  # pylint: disable = no-member
            opq_matrix = faiss.OPQMatrix(d, M=M_OPQ, d2=out_d)
            # opq_matrix.niter = 50 # Same as default value
            index = faiss.IndexPreTransform(opq_matrix, index_ivfpq)
        elif any(re.findall(r"OPQ\d+_\d+,IVF\d+_HNSW\d+,PQ\d+", index_key)):
            params = [int(x) for x in re.findall(r"\d+", index_key)]

            M_HNSW = params[3]
            cs = params[4]  # code size (in Bytes if nbits=8)
            nbits = params[5] if len(params) == 6 else 8  # default value
            ncentroids = params[2]
            out_d = params[1]
            M_OPQ = params[0]

            quantizer = faiss.IndexHNSWFlat(out_d, M_HNSW, metric_type)
            if ef_construction is not None and ef_construction >= 1:
                quantizer.hnsw.efConstruction = ef_construction
            assert quantizer.metric_type == metric_type
            index_ivfpq = faiss.IndexIVFPQ(quantizer, out_d, ncentroids, cs, nbits, metric_type)
            assert index_ivfpq.metric_type == metric_type
            index_ivfpq.own_fields = True
            quantizer.this.disown()  # pylint: disable = no-member
            opq_matrix = faiss.OPQMatrix(d, M=M_OPQ, d2=out_d)
            # opq_matrix.niter = 50 # Same as default value
            index = faiss.IndexPreTransform(opq_matrix, index_ivfpq)

        elif any(re.findall(r"Pad\d+,IVF\d+_HNSW\d+,PQ\d+", index_key)):
            params = [int(x) for x in re.findall(r"\d+", index_key)]

            out_d = params[0]
            M_HNSW = params[2]
            cs = params[3]  # code size (in Bytes if nbits=8)
            nbits = params[4] if len(params) == 5 else 8  # default value
            ncentroids = params[1]

            remapper = faiss.RemapDimensionsTransform(d, out_d, True)

            quantizer = faiss.IndexHNSWFlat(out_d, M_HNSW, metric_type)
            if ef_construction is not None and ef_construction >= 1:
                quantizer.hnsw.efConstruction = ef_construction
            index_ivfpq = faiss.IndexIVFPQ(quantizer, out_d, ncentroids, cs, nbits, metric_type)
            index_ivfpq.own_fields = True
            quantizer.this.disown()  # pylint: disable = no-member

            index = faiss.IndexPreTransform(remapper, index_ivfpq)
        elif any(re.findall(r"HNSW\d+", index_key)):
            params = [int(x) for x in re.findall(r"\d+", index_key)]
            M_HNSW = params[0]
            index = faiss.IndexHNSWFlat(d, M_HNSW, metric_type)
            assert index.metric_type == metric_type
        elif index_key == "Flat" or any(re.findall(r"IVF\d+,Flat", index_key)):
            index = faiss.index_factory(d, index_key, metric_type)
        else:
            index = faiss.index_factory(d, index_key, metric_type)
            raise ValueError(
                (
                    "Be careful, faiss might not create what you expect when using the "
                    "inner product similarity metric, remove this line to try it anyway. "
                    "Happened with index_key: " + str(index_key)
                )
            )

    else:
        index = faiss.index_factory(d, index_key, metric_type)

    return index
