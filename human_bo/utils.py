"""Odd functions one may need but is otherwise not really core."""

import torch


def recursively_filter_dict(d: dict, predicate):
    """Will do DFS through `d` and return all elements for which `predicate` returns True

    Useful if you want to do something on all elements with some property
    in a dictionary of some unknown or variable depth.

    For example, if you want to sum all "leaves":

        `sum(return_all_elements(d, lambda _, v: not isinstance(v, dict)))``

    :d: the dictionary to get leaves of
    """
    for k, v in d.items():
        if predicate(k, v):
            yield v

        if isinstance(v, dict):
            for v in recursively_filter_dict(v, predicate):
                yield v


def normalize_data(x: torch.Tensor) -> torch.Tensor:
    """Normalizes `x` such that its mean is 0 and standard deviation is 1.

    NOTE: does not support more than 1 dimension for `x`

    :x: 1-dimensional data to normalize
    :returns: normalized  `x`
    """
    assert (
        len(x.shape) == 1 or len(x.shape) == 2 and x.shape[1] == 1
    ), f"`normalize_data` currently does not support multiple {x.shape} dimensions"

    mean = x.mean()
    std = x.std()

    return (x - mean) / std
