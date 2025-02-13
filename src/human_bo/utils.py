"""Odd functions one may need but is otherwise not really core."""

import os
import sys
import traceback
import warnings


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    """Custom function to print (warning) traces properly

    Taken from https://stackoverflow.com/questions/22373927/get-traceback-of-warnings.

    Really, just for debugging and figuring out where the warnings are generated from.

    To use:

        import warnings
        warnings.showwarning = utils.warn_with_traceback
    """
    log = file if hasattr(file, "write") else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


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
            yield from recursively_filter_dict(v, predicate)


def exit_if_exists(path: str, negate=False):
    """Exits with an error if `path` exists.

    Set `negate` to true if you want to fail *if `path` does not exist*
    """
    if os.path.exists(path) is not negate:
        msg = "does not exist" if negate else "already exists"
        raise ValueError(f"File {path} {msg}, aborting run!")


def create_directory_if_does_not_exist(path: str):
    if not os.path.exists(path):
        print(f"Creating {path}")
        os.makedirs(path)
