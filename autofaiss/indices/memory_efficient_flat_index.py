""" This file contain a class describing a memory efficient flat index """

import heapq
from typing import List, Optional, Tuple

from embedding_reader import EmbeddingReader

import faiss
import numpy as np
from tqdm import trange

from autofaiss.indices.faiss_index_wrapper import FaissIndexWrapper


class MemEfficientFlatIndex(FaissIndexWrapper):
    """
    Faiss-like Flat index that can support any size of vectors
    without memory issues.
    Two search functions are available to use either batch of smaller faiss
    flat index or rely fully on numpy.
    """

    def __init__(self, d: int, metric_type: int):
        """
        __init__ function for MemEfficientFlatIndex

        Parameters:
        -----------
        d : int
            dimension of the vectors, named d to keep Faiss notation
        metric_type : int
            similarity metric used in the vector space, using faiss
            enumerate values (faiss.METRIC_INNER_PRODUCT and faiss.METRIC_L2)
        """

        super().__init__(d, metric_type)

        self.dim = d
        self.prod_emb = np.zeros((0, self.dim))
        self.embedding_reader: Optional[EmbeddingReader] = None

    def delete_vectors(self):
        """delete the vectors of the index"""
        self.prod_emb = np.zeros((0, self.dim))

    # pylint: disable=missing-function-docstring, invalid-name
    def add(self, x: np.ndarray):
        if self.prod_emb.shape[0] == 0:
            self.prod_emb = x.astype(np.float32)
        else:
            raise NotImplementedError("You can add vectors only once, delete them first with delete_vectors")

    def add_all(self, filename: str, nb_items: int):
        """
        Function that adds vectors to the index from a memmory-mapped array

        Parameters
        ----------
        filename : string
            path of the 2D numpy array of shape (nb_items, vector_dim)
            on the disk
        nb_items : int
            number of vectors in the 2D array (the dim is already known)
        """

        if self.prod_emb.shape[0] == 0:
            self.prod_emb = np.memmap(filename, dtype="float32", mode="r", shape=(nb_items, self.dim))
        else:
            raise NotImplementedError("You can add vectors only once, delete them first")

    def add_files(self, embedding_reader: EmbeddingReader):
        if self.embedding_reader is None:
            self.embedding_reader = embedding_reader
        else:
            raise NotImplementedError("You can add vectors only once, delete them first with delete_vectors")

    # pylint: disable too_many_locals
    def search_numpy(self, xq: np.ndarray, k: int, batch_size: int = 4_000_000):
        """
        Function that search the k nearest neighbours of a batch of vectors.
        This implementation is based on vectorized numpy function, it is slower than
        the search function based on batches of faiss flat indices.
        We keep this implementation because we can build new functions using this code.
        Moreover, the distance computation is more precise in numpy than the faiss implementation
        that optimizes speed over precision.

        Parameters
        ----------
        xq : 2D numpy.array of floats
            Batch of vectors of shape (batch_size, vector_dim)
        k : int
            Number of neighbours to retrieve for every vector
        batch_size : int
            Size of the batch of vectors that are explored.
            A bigger value is prefered to avoid multiple loadings
            of the vectors from the disk.

        Returns
        -------
        D : 2D numpy.array of floats
            Distances numpy array of shape (batch_size, k).
            Contains the distances computed by the index of the k nearest neighbours.
        I : 2D numpy.array of ints
            Labels numpy array of shape (batch_size, k).
            Contains the vectors' labels of the k nearest neighbours.
        """
        assert self.metric_type == faiss.METRIC_INNER_PRODUCT

        # Instanciate several heaps, (is there a way to have vectorized heaps?)
        h: List[List[Tuple[float, int]]] = [[] for _ in range(xq.shape[0])]

        # reshape input for vectorized distance computation
        xq_reshaped = np.expand_dims(xq, 1)

        # initialize index offset
        offset = 0

        # For each batch
        for i in trange(0, self.prod_emb.shape[0], batch_size):

            # compute distances in one tensor product
            dist_arr = np.sum((xq_reshaped * np.expand_dims(self.prod_emb[i : i + batch_size], 0)), axis=-1)

            # get index of the k biggest
            # pylint: disable=unsubscriptable-object # pylint/issues/3139
            max_k = min(k, dist_arr.shape[1])

            ind_k_max = np.argpartition(dist_arr, -max_k)[:, -max_k:]

            assert ind_k_max.shape == (xq.shape[0], max_k)

            # to be vectorized if it is indeed the bottleneck, (it's not for batch_size >> 10000)
            for j, inds in enumerate(ind_k_max):
                for ind, distance in zip(inds, dist_arr[j, inds]):
                    true_ind = offset + ind if ind != -1 else -1
                    if len(h[j]) < k:
                        heapq.heappush(h[j], (distance, true_ind))
                    else:
                        heapq.heappushpop(h[j], (distance, true_ind))

            offset += batch_size

        # Fill distance and indice matrix
        D = np.zeros((xq.shape[0], k), dtype=np.float32)
        I = np.full((xq.shape[0], k), fill_value=-1, dtype=np.int32)

        for i in range(xq.shape[0]):
            # case where we couldn't find enough vectors
            max_k = min(k, len(h[i]))
            for j in range(max_k):
                x = heapq.heappop(h[i])
                D[i][max_k - 1 - j] = x[0]
                I[i][max_k - 1 - j] = x[1]

        return D, I

    # pylint: disable=too-many-locals, arguments-differ
    def search(self, x: np.ndarray, k: int, batch_size: int = 4_000_000):
        """
        Function that search the k nearest neighbours of a batch of vectors

        Parameters
        ----------
        x : 2D numpy.array of floats
            Batch of vectors of shape (batch_size, vector_dim)
        k : int
            Number of neighbours to retrieve for every vector
        batch_size : int
            Size of the batch of vectors that are explored.
            A bigger value is prefered to avoid multiple loadings
            of the vectors from the disk.

        Returns
        -------
        D : 2D numpy.array of floats
            Distances numpy array of shape (batch_size, k).
            Contains the distances computed by the index of the k nearest neighbours.
        I : 2D numpy.array of ints
            Labels numpy array of shape (batch_size, k).
            Contains the vectors' labels of the k nearest neighbours.
        """

        if self.prod_emb is None:
            raise ValueError("The index is empty")

        # Cast in the right format for Faiss
        if x.dtype != np.float32:
            x = x.astype(np.float32)

        # xq for x query, a better name than x which is Faiss convention
        xq = x

        # Instanciate several heaps, (is there a way to have vectorized heaps?)
        h: List[List[Tuple[float, int]]] = [[] for _ in range(xq.shape[0])]

        # initialize index offset
        offset = 0

        # For each batch
        for i in trange(0, self.prod_emb.shape[0], batch_size):

            # instanciate a Flat index
            brute = faiss.IndexFlatIP(self.dim)
            # pylint: disable=no-value-for-parameter
            brute.add(self.prod_emb[i : i + batch_size])
            D_tmp, I_tmp = brute.search(xq, k)

            # to be vectorized if it is indeed the bottleneck, (it's not for batch_size >> 10000)
            for j, (distances, inds) in enumerate(zip(D_tmp, I_tmp)):
                for distance, ind in zip(distances, inds):
                    true_ind: int = offset + ind if ind != -1 else -1
                    if len(h[j]) < k:
                        heapq.heappush(h[j], (distance, true_ind))
                    else:
                        heapq.heappushpop(h[j], (distance, true_ind))

            offset += batch_size

        # Fill distance and indice matrix
        D = np.zeros((xq.shape[0], k), dtype=np.float32)
        I = np.full((xq.shape[0], k), fill_value=-1, dtype=np.int32)

        for i in range(xq.shape[0]):
            # case where we couldn't find enough vectors
            max_k = min(k, len(h[i]))
            for j in range(max_k):
                x = heapq.heappop(h[i])
                D[i][max_k - 1 - j] = x[0]
                I[i][max_k - 1 - j] = x[1]

        return D, I

    def search_files(self, x: np.ndarray, k: int, batch_size: int):

        if self.embedding_reader is None:
            raise ValueError("The index is empty")

        # Cast in the right format for Faiss
        if x.dtype != np.float32:
            x = x.astype(np.float32)

        # xq for x query, a better name than x which is Faiss convention
        xq = x

        # Instanciate several heaps, (is there a way to have vectorized heaps?)
        h: List[List[Tuple[float, int]]] = [[] for _ in range(xq.shape[0])]

        # initialize index offset
        offset = 0

        # For each batch
        for emb_array, _ in self.embedding_reader(batch_size):
            # for i in trange(0, self.prod_emb.shape[0], batch_size):

            # instanciate a Flat index
            brute = faiss.IndexFlatIP(self.dim)
            # pylint: disable=no-value-for-parameter
            brute.add(emb_array)
            D_tmp, I_tmp = brute.search(xq, k)

            # to be vectorized if it is indeed the bottleneck, (it's not for batch_size >> 10000)
            for j, (distances, inds) in enumerate(zip(D_tmp, I_tmp)):
                for distance, ind in zip(distances, inds):
                    true_ind: int = offset + ind if ind != -1 else -1
                    if len(h[j]) < k:
                        heapq.heappush(h[j], (distance, true_ind))
                    else:
                        heapq.heappushpop(h[j], (distance, true_ind))

            offset += emb_array.shape[0]

        # Fill distance and indice matrix
        D = np.zeros((xq.shape[0], k), dtype=np.float32)
        I = np.full((xq.shape[0], k), fill_value=-1, dtype=np.int32)

        for i in range(xq.shape[0]):
            # case where we couldn't find enough vectors
            max_k = min(k, len(h[i]))
            for j in range(max_k):
                x = heapq.heappop(h[i])  # type: ignore
                D[i][max_k - 1 - j] = x[0]
                I[i][max_k - 1 - j] = x[1]

        return D, I
