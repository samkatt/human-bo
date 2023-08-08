"""Contains data and functions for handling experiment configurations"""

from argparse import Namespace
from typing import Any


CONFIG = {
    "seed": {
        "type": int,
        "shorthand": "s",
        "help": "Random seed to run the experiment",
        "tags": {},
    },
    "budget": {
        "type": int,
        "shorthand": "b",
        "help": "Number of queries",
        "tags": {"experiment-hyper-parameter"},
    },
    "n_init": {
        "type": int,
        "shorthand": "ni",
        "help": "Number of initial data points",
        "tags": {"experiment-hyper-parameter"},
    },
    "kernel": {
        "type": str,
        "shorthand": "k",
        "help": "Kernel of the GP",
        "tags": {"experiment-parameter"},
    },
    "acqf": {
        "type": str,
        "shorthand": "a",
        "help": "Acquisition function used",
        "tags": {"experiment-parameter"},
    },
    "function": {
        "type": str,
        "shorthand": "f",
        "help": "Test function to find max of",
        "tags": {"experiment-parameter"},
    },
    "oracle": {
        "type": str,
        "shorthand": "o",
        "help": "The mechanism through which queries are given",
        "tags": {"experiment-parameter"},
    },
}


def from_ns(ns: Namespace) -> dict[str, Any]:
    """Generates a configuration dictionary from a (arg parse) name space

    See `CONFIG` for the building block: we simply return a `dict[str, value]` of them.
    """
    ns_dict = vars(ns)

    conf = {}
    for k, v in CONFIG.items():
        conf[k] = v["type"](ns_dict[k])

    return conf
