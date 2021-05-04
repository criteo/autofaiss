""" Various optimization algorithms """

from typing import Callable

# pylint: disable=invalid-name
def discrete_binary_search(is_ok: Callable[[int], bool], n: int) -> int:
    """
    Binary search in a function domain

    Parameters
    ----------
    is_ok : bool
        Boolean monotone function defined on range(n)
    n : int
        length of the search scope

    Returns
    -------
    i : int
        first index i such that is_ok(i) or returns n if is_ok is all False

    :complexity: O(log(n))
    """

    lo = 0
    hi = n
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if mid >= n or is_ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
