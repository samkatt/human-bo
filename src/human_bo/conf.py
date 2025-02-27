"""Contains data and functions for handling experiment configurations"""

import math
from typing import Any

CONFIG: dict[str, dict[str, Any]] = {
    "seed": {
        "type": int,
        "shorthand": "s",
        "help": "Random seed to run the experiment.",
        "tags": {},
        "parser-arguments": {"default": 0},
    },
    "budget": {
        "type": int,
        "shorthand": "b",
        "help": "Number of queries.",
        "tags": {"experiment-hyper-parameter"},
        "parser-arguments": {"default": 10},
    },
    "n_init": {
        "type": int,
        "shorthand": "ni",
        "help": "Number of initial data points.",
        "tags": {"experiment-hyper-parameter"},
        "parser-arguments": {"default": 0},
    },
    "kernel": {
        "type": str,
        "shorthand": "k",
        "help": "Kernel of the GP.",
        "tags": {"experiment-parameter"},
        "parser-arguments": {
            "default": "Default",
            "choices": {"RBF", "Matern", "Default"},
        },
    },
    "acqf": {
        "type": str,
        "shorthand": "a",
        "help": "Acquisition function used.",
        "tags": {"experiment-parameter"},
        "parser-arguments": {"default": "EI", "choices": {"UCB", "MES", "EI"}},
    },
    "problem": {
        "type": str,
        "shorthand": "p",
        "help": "Test function to find max of.",
        "tags": {"experiment-parameter"},
        "parser-arguments": {
            "required": True,
            "choices": {
                "Zhou": {"dims": 1, "optimal_x": [[0.34], [0.68]]},
                "Ackley1D": {"dims": 1, "optimal_x": [[0.0]]},
                "DixonPrice1D": {"dims": 1, "optimal_x": [[0.0]]},
                "Griewank1D": {"dims": 1, "optimal_x": [[0.0]]},
                "Levy1D": {"dims": 1, "optimal_x": [[1.0]]},
                "Rastrigin1D": {"dims": 1, "optimal_x": [[0.0]]},
                "StyblinskiTang1D": {"dims": 1, "optimal_x": [[-39.166166]]},
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
                "Rosenbrock2D": {"dims": 2, "optimal_x": [[1.0, 1.0]]},
                "BraninCurrin": {"dims": 2, "num_objectives": 2},
            },
        },
    },
    "problem_noise": {
        "type": float,
        "shorthand": "e",
        "help": "The Gaussian noise (variation) with which function `f` is observed.",
        "tags": {"experiment-hyper-parameter"},
        "parser-arguments": {"default": [0.1], "nargs": "+"},
    },
}


def get_values_with_tag(
    exp_params: dict[str, Any],
    tag: str,
    exp_conf: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Returns values in `exp_params` of entries with keys that have `tag` in `exp_conf"""
    if exp_conf is None:
        exp_conf = CONFIG
    return [
        str(v)
        for k, v in exp_params.items()
        if k in exp_conf and tag in exp_conf[k]["tags"]
    ]
