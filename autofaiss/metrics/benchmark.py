""" functions to compare different indices """

import time

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm as tq

from autofaiss.indices.index_utils import format_speed_ms_per_query, get_index_size, speed_test_ms_per_query
from autofaiss.metrics.recalls import r_recall_at_r_single, one_recall_at_r_single
from autofaiss.utils.cast import cast_bytes_to_memory_string


def avg_speed_dict_ms_per_query(indices_dict, vectors, k_closest: int = 40, timeout_s: float = 5):
    """compute the average query speed of a dictionary of indices"""
    speed_dict = {}

    for index_key in indices_dict:
        speed = speed_test_ms_per_query(indices_dict[index_key], vectors, k_closest, timeout_s)
        speed_dict[index_key] = speed

    return speed_dict


def index_sizes_in_bytes_dict(indices_dict):
    """compute sizes of indices in a dictionary of indices"""
    size_dict = {}

    for index_key in indices_dict:
        size_dict[index_key] = get_index_size(indices_dict[index_key])

    return size_dict


def benchmark_index(
    indices_dict, gt_test, test_points, vectors_size_in_bytes, save_path=None, speed_dict=None, size_dict=None
):
    """
    Compute recall curves for the indices.
    """

    perfect_index_label = "perfect index"

    if perfect_index_label not in indices_dict:
        indices_dict[perfect_index_label] = None
        if speed_dict:
            speed_dict[perfect_index_label] = vectors_size_in_bytes

    k_max = gt_test.shape[1]

    plt.figure(figsize=(16, 8))

    k_values = np.arange(0, k_max + 1)

    avg_one_recall_at_r = {}
    avg_r_recall_at_r = {}

    timout_s = 5.0

    comp_size = vectors_size_in_bytes

    for index_key in tq(list(sorted(indices_dict.keys()))):
        if index_key not in indices_dict:
            continue

        index = indices_dict[index_key]

        if index_key == "Flat" or (index is None):
            y_r_recall_at_r = np.arange(1, k_max + 1)
            y_one_recall_at_r = np.ones(k_max)
            tot = 1
        else:
            y_r_recall_at_r = np.zeros(k_max)
            y_one_recall_at_r = np.zeros(k_max)
            tot = 0
            start_time = time.time()
            for i, item in enumerate(test_points):
                y_r_recall_at_r += np.array(r_recall_at_r_single(item, gt_test[i], index, k_max))
                y_one_recall_at_r += np.array(one_recall_at_r_single(item, gt_test[i], index, k_max))
                tot += 1
                if time.time() - start_time > timout_s and tot > 150:
                    break

        avg_r_recall_at_r[index_key] = y_r_recall_at_r / tot
        avg_one_recall_at_r[index_key] = y_one_recall_at_r / tot

    info_string = {index_key: "" for index_key in indices_dict}

    initial_size_string = cast_bytes_to_memory_string(comp_size)

    for index_key in indices_dict:
        if index_key in speed_dict:
            info_string[index_key] += f"avg speed: {format_speed_ms_per_query(speed_dict[index_key])}, "
        if index_key in size_dict:
            info_string[index_key] += (
                f"(Size: {cast_bytes_to_memory_string(size_dict[index_key])} "
                f"({(100*size_dict[index_key]/comp_size):.1f}% of {initial_size_string})"
            )

    plt.subplot(121)

    for index_key in sorted(indices_dict.keys()):
        if index_key not in indices_dict:
            continue

        label = f"{index_key:<30} Index, {info_string[index_key]}"

        plt.plot(k_values, np.concatenate(([0], avg_r_recall_at_r[index_key])), label=label)

    plt.xlabel("k, number of nearests items")
    plt.ylabel("k-recall@k")
    plt.vlines(40, 0, k_max)
    plt.legend()
    plt.tight_layout()

    plt.subplot(122)

    for index_key in sorted(indices_dict.keys()):
        if index_key not in indices_dict:
            continue

        label = f"{index_key:<30} Index, {info_string[index_key]}"

        plt.plot(k_values, np.concatenate(([0], 100 * avg_one_recall_at_r[index_key])), label=label)

    plt.xlabel("k, number of nearests items")
    plt.ylabel("1-Recall@k")
    plt.vlines(100, 0, k_max)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
