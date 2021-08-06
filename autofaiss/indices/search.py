""" function related to search on indices """

from typing import Iterable, Tuple

import numpy as np


def knn_query(index, query, ksearch: int) -> Iterable[Tuple[Tuple[int, int], float]]:
    """Do a knn search and return a list of the closest items and the associated distance"""

    dist, ind = index.search(np.expand_dims(query, 0), ksearch)

    distances = dist[0]

    item_dist = list(zip(ind[0], distances))

    return item_dist
