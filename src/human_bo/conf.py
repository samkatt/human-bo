"""Contains data and functions for handling experiment configurations"""

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
    "acqf": {
        "type": str,
        "shorthand": "a",
        "help": "Acquisition function used.",
        "tags": {"experiment-parameter"},
        "parser-arguments": {
            "choices": {"UCB", "MES", "EI", "random"},
            "default": "EI",
        },
    },
    "ucb_beta": {
        "type": float,
        "shorthand": "c",
        "help": "Exploration constant used in UCB",
        "tags": {"acqf-option", "experiment-hyper-parameter"},
        "parser-arguments": {},
    },
    "problem": {
        "type": str,
        "shorthand": "p",
        "help": "Test function to find max of.",
        "tags": {"experiment-parameter"},
        "parser-arguments": {
            "required": True,
            "choices": {
                "Zhou": {"dims": 1},
                "Levy1D": {"dims": 1},
                "Forrester": {"dims": 1},
                "Branin": {"dims": 2},
                "Currin": {"dims": 2},
                "BraninCurrin": {"dims": 2, "num_objectives": 2},
                "DTLZ2": {},
                "VLMOP2": {"num_objectives": 2},
            },
        },
    },
    "problem_noise": {
        "type": float,
        "shorthand": "e",
        "help": "The Gaussian noise (variation) with which function `f` is observed.",
        "tags": {"experiment-hyper-parameter", "problem-parameters"},
        "parser-arguments": {"nargs": "+"},
    },
}


def get_entries_with_tag(
    exp_params: dict[str, Any],
    tag: str,
    exp_conf: dict[str, dict[str, Any]] | None = None,
):
    """Returns {k: v} in `exp_params` of entries with keys that have `tag` in `exp_conf`"""
    if exp_conf is None:
        exp_conf = CONFIG
    return {
        k: v
        for k, v in exp_params.items()
        if k in exp_conf and tag in exp_conf[k]["tags"]
    }


def get_values_with_tag(
    exp_params: dict[str, Any],
    tag: str,
    exp_conf: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Returns values in `exp_params` of entries with keys that have `tag` in `exp_conf`"""
    return [str(v) for v in get_entries_with_tag(exp_params, tag, exp_conf).values()]
