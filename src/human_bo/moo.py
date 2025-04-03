"""Core functionality for multi-objective optimization."""

from typing import Any
import random

CONFIG: dict[str, dict[str, Any]] = {
    "preference_weights": {
        "type": float,
        "shorthand": "w",
        "help": "The (linear) utility weight preferences.",
        "tags": {"experiment-hyper-parameter", "problem-parameters"},
        "parser-arguments": {
            "nargs": "+",
        },
    },
    "x_dim": {
        "type": int,
        "shorthand": "d",
        "help": "Number of dimensions of `x` (query).",
        "tags": {"experiment-hyper-parameter", "problem-parameters"},
        "parser-arguments": {"default": 2},
    },
    "o_dim": {
        "type": int,
        "shorthand": "o",
        "help": "Number of objectives.",
        "tags": {"experiment-hyper-parameter", "problem-parameters"},
        "parser-arguments": {"default": 2},
    },
}


def sample_preference_weights(o_dim: int):
    assert o_dim > 1

    weights = [random.uniform(0, 1) for _ in range(o_dim)]
    total = sum(weights)

    return [w / total for w in weights]
