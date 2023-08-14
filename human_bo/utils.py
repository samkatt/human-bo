"""Odd functions one may need but is otherwise not really core."""


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
