""" test utils functions """
# pylint: disable= invalid-name

import numpy as np
import pytest

from autofaiss.utils.array_functions import multi_array_split


def test_multi_array_split():
    """test multi_array_split fct number 1"""
    assert len(list(multi_array_split([np.zeros((123, 2)), np.zeros((123, 5))], 41))) == 41


@pytest.mark.parametrize("seed", list(range(1, 10)))
def test_multi_array_split_2(seed):
    """test multi_array_split fct number 2"""

    np.random.seed(seed)
    length = np.random.randint(1, 100)
    nb_chunk = np.random.randint(1, length + 1)
    dim1 = np.random.randint(10)
    dim2 = np.random.randint(10)

    a = np.random.randint(0, 10000, (length, dim1))
    b = np.random.randint(0, 10000, (length, dim2))

    c = list(multi_array_split([a, b], nb_chunk))

    a2 = np.concatenate([x[0] for x in c])
    b2 = np.concatenate([x[1] for x in c])

    assert np.all(a == a2)
    assert np.all(b == b2)
