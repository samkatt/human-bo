"""Core functionality for multi-objective optimization."""

import random
from collections.abc import Callable
from typing import Any, TypeVar

CONFIG: dict[str, dict[str, Any]] = {
    "scalarization_weights": {
        "type": float,
        "shorthand": "w",
        "help": "The (linear) scalarization weights.",
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


def sample_scalarization_weights(o_dim: int):
    assert o_dim > 1

    weights = [random.uniform(0, 1) for _ in range(o_dim)]
    total = sum(weights)

    return [w / total for w in weights]


T = TypeVar("T")


def generate_front(n: int, gen: Callable[[], T], comp: Callable[[T, T], int]) -> set[T]:
    """Generate `n` non-dominating points of type `T` according to `comp`."""
    front: set[T] = set()

    while len(front) < n:
        candidate = gen()
        dominating = set()

        for x in front:
            rel = comp(candidate, x)

            if rel == 1:
                dominating.add(x)
            elif rel == -1:
                assert len(dominating) == 0
                continue

        for x in dominating:
            front.remove(x)
        front.add(candidate)

    return front
