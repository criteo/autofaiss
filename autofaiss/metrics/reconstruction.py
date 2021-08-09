""" function to compute the reconstruction error """

from typing import Optional

import numpy as np
import faiss


def reconstruction_error(before, after, avg_norm_before: Optional[float] = None) -> float:
    """Computes the average reconstruction error"""
    diff = np.mean(np.linalg.norm(after - before, axis=1))
    if avg_norm_before is None:
        avg_norm_before = np.mean(np.linalg.norm(before, axis=1))
    return diff / avg_norm_before


def quantize_vec_without_modifying_index(index: faiss.Index, vecs: np.ndarray) -> np.ndarray:
    """Quantizes a batch of vectors if the index given uses quantization"""

    try:
        return index.sa_decode(index.sa_encode(vecs))
    except (TypeError, RuntimeError):  # error if the index doesn't use quantization
        return vecs
