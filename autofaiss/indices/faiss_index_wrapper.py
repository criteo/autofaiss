""" This file contains a wrapper class to create Faiss-like indices """

from abc import ABC, abstractmethod

import faiss
import numpy as np


class FaissIndexWrapper(ABC):
    """
    This abstract class is describing a Faiss-like index
    It is useful to use this wrapper to use benchmarking functions written for
    faiss in this library
    """

    # pylint: disable=invalid-name
    def __init__(self, d: int, metric_type: int):
        """
        __init__ function for FaissIndexWrapper

        Parameters:
        -----------
        d : int
            dimension of the vectors, named d to keep Faiss notation
        metric_type : int
            similarity metric used in the vector space, using faiss
            enumerate values (faiss.METRIC_INNER_PRODUCT and faiss.METRIC_L2)
        """

        self.d = d

        if metric_type in [faiss.METRIC_INNER_PRODUCT, "IP", "ip"]:
            self.metric_type = faiss.METRIC_INNER_PRODUCT
        elif metric_type in [faiss.METRIC_L2, "L2", "l2"]:
            self.metric_type = faiss.METRIC_L2
        else:
            raise NotImplementedError

    # pylint: disable=invalid-name
    @abstractmethod
    def search(self, x: np.ndarray, k: int):
        """
        Function that search the k nearest neighbours of a batch of vectors

        Parameters
        ----------
        x : 2D numpy.array of floats
            Batch of vectors of shape (batch_size, vector_dim)
        k : int
            Number of neighbours to retrieve for every vector

        Returns
        -------
        D : 2D numpy.array of floats
            Distances numpy array of shape (batch_size, k).
            Contains the distances computed by the index of the k nearest neighbours.
        I : 2D numpy.array of ints
            Labels numpy array of shape (batch_size, k).
            Contains the vectors' labels of the k nearest neighbours.
        """

        raise NotImplementedError

    # pylint: disable=invalid-name
    @abstractmethod
    def add(self, x: np.ndarray):
        """
        Function that adds vectors to the index

        Parameters
        ----------
        x : 2D numpy.array of floats
            Batch of vectors of shape (batch_size, vector_dim)
        """

        raise NotImplementedError
