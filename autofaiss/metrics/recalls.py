""" function to compute different kind of recalls """

from typing import List, Optional

import faiss
import numpy as np


def r_recall_at_r_single(
    query: np.ndarray,
    ground_truth: np.ndarray,
    other_index: faiss.Index,
    r_max: int = 40,
    eval_item_ids: Optional[np.ndarray] = None,
) -> List[int]:
    """Compute an R-recall@R array for each R in range [1, R_max]"""
    # O(r_max)

    _, inds = other_index.search(np.expand_dims(query, 0), r_max)

    res = inds[0]

    recall_count = []
    s_true = set()
    s_pred = set()
    tot = 0
    for p_true, p_pred in zip(ground_truth[:r_max], res):
        if eval_item_ids is not None and p_pred != -1:
            p_pred = eval_item_ids[p_pred]
        if p_true == p_pred and p_true != -1:
            tot += 1
        else:
            if p_true in s_pred and p_true != -1:
                tot += 1
            if p_pred in s_true and p_pred != -1:
                tot += 1

        s_true.add(p_true)
        s_pred.add(p_pred)
        recall_count.append(tot)

    return recall_count


def r_recall_at_r(
    query: np.ndarray,
    ground_truth: np.ndarray,
    other_index: faiss.Index,
    r_max: int = 40,
    eval_item_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute an R-recall@R array for each R in range [1, R_max] for
    a single query.
    """
    # O(r_max)

    r_lim = min(r_max, other_index.ntotal)

    if r_lim <= 0:
        return np.ones((max(r_max, 0),))

    total = np.zeros((r_max,))
    for i in range(query.shape[0]):

        # If the ground truth contains -1 (missing elements), the recall definition must change.
        # We should divide by the number of elements possible to retrieve, not r_lim
        r_lim_fix = min(r_lim, np.min(np.where(ground_truth[i] == -1)[0])) if -1 in ground_truth[i] else r_lim

        res_for_one = r_recall_at_r_single(
            query[i], ground_truth[i], other_index, r_max, eval_item_ids
        ) / np.concatenate((np.arange(1, r_lim_fix + 1, 1), np.full(r_max - r_lim_fix, r_lim_fix)))
        total += np.array(res_for_one)

    return total / query.shape[0]


def one_recall_at_r_single(
    query: np.ndarray,
    ground_truth: np.ndarray,
    other_index: faiss.Index,
    r_max: int = 40,
    eval_item_ids: Optional[np.ndarray] = None,
) -> List[int]:
    """
    Compute an 1-recall@R array for each R in range [1, r_max] for
    a single query.
    """
    # O(r_max)

    _, inds = other_index.search(np.expand_dims(query, 0), 1)

    first = inds[0][0]
    if eval_item_ids is not None and first != -1:
        first = eval_item_ids[first]

    # return empty array if no product is found by other_index
    if first == -1:
        return [0 for _ in ground_truth[:r_max]]

    recall_count = []

    seen = False
    for p_true in ground_truth[:r_max]:
        if p_true == first:
            seen = True
        recall_count.append(1 if seen else 0)

    return recall_count


def one_recall_at_r(
    query: np.ndarray,
    ground_truth: np.ndarray,
    other_index: faiss.Index,
    r_max: int = 40,
    eval_item_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute an 1-recall@R array for each R in range [1, r_max]"""
    # O(r_max)

    if r_max <= 0:
        return np.zeros((0,))

    _, first = other_index.search(query, 1)

    if eval_item_ids is not None:
        first = np.vectorize(lambda e: eval_item_ids[e] if e != -1 else -1)(first)  # type: ignore

    recall_array = np.cumsum((ground_truth[:, :r_max] == first) & (first != -1), axis=-1)

    avg_recall = np.mean(recall_array, axis=0)

    return avg_recall
