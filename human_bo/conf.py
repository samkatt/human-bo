"""Contains data and functions for handling experiment configurations"""

import math
from argparse import Namespace
from typing import Any

CONFIG = {
    "seed": {
        "type": int,
        "shorthand": "s",
        "help": "Random seed to run the experiment",
        "tags": {},
        "parser-arguments": {"required": True},
    },
    "budget": {
        "type": int,
        "shorthand": "b",
        "help": "Number of queries",
        "tags": {"experiment-hyper-parameter"},
        "parser-arguments": {"required": True},
    },
    "n_init": {
        "type": int,
        "shorthand": "ni",
        "help": "Number of initial data points",
        "tags": {"experiment-hyper-parameter"},
        "parser-arguments": {"required": True},
    },
    "kernel": {
        "type": str,
        "shorthand": "k",
        "help": "Kernel of the GP",
        "tags": {"experiment-parameter"},
        "parser-arguments": {"required": True},
    },
    "acqf": {
        "type": str,
        "shorthand": "a",
        "help": "Acquisition function used",
        "tags": {"experiment-parameter"},
        "parser-arguments": {"required": True},
    },
    "problem": {
        "type": str,
        "shorthand": "p",
        "help": "Test function to find max of",
        "tags": {"experiment-parameter"},
        "parser-arguments": {
            "required": True,
            "choices": {
                "Zhou": {"dims": 1, "optimal_x": [[0.34], [0.68]]},
                "Forrester": {"dims": 1, "optimal_x": [[1.0]]},
                "Hartmann": {
                    "dims": 6,
                    "optimal_x": [
                        [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]
                    ],
                },
                "Branin": {
                    "dims": 2,
                    "optimal_x": [
                        [-math.pi, 12.275],
                        [math.pi, 2.275],
                        [9.42478, 2.475],
                    ],
                },
                "Rosenbrock": {"dims": "n", "optimal_x": [[1.0, 1.0]]},
            },
        },
    },
    "problem_noise": {
        "type": float,
        "shorthand": "e",
        "help": "The Gaussian noise (variation) with which function `f` is observed",
        "tags": {"experiment-hyper-parameter"},
        "parser-arguments": {"default": 0.01},
    },
    "user_model": {
        "type": str,
        "shorthand": "u",
        "help": "The mechanism through which queries are given",
        "tags": {"experiment-parameter"},
        "parser-arguments": {"choices": {"oracle", "gauss"}, "default": "oracle"},
    },
}


def from_ns(ns: Namespace) -> dict[str, Any]:
    """Generates a configuration dictionary from a (arg parse) name space

    See `CONFIG` for the building block: we simply return a `dict[str, value]` of them.
    """
    ns_dict = vars(ns)

    conf = {}
    for k, v in CONFIG.items():
        # v["type"] returns class (e.g. `int`).
        # We use that to cast it to the correct value
        conf[k] = v["type"](ns_dict[k])

    return conf
