"""Core functionality for multi-objective optimization."""

import random
from collections.abc import Callable
from typing import Any, TypeVar

import tensorflow_probability as tfp
import tensorflow as tf

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
        "tags": {"experiment-parameter", "problem-parameters"},
        "parser-arguments": {"default": 2},
    },
    "o_dim": {
        "type": int,
        "shorthand": "o",
        "help": "Number of objectives.",
        "tags": {"experiment-parameter", "problem-parameters"},
        "parser-arguments": {"default": 2},
    },
    "type_agent": {
        "type": str,
        "shorthand": "t",
        "help": "The type of agent to use.",
        "tags": {"experiment-parameter"},
        "parser-arguments": {
            "choices": {"random", "bo", "composite", "utility-learner"},
            "required": True,
        },
    },
}


def sample_scalarization_weights(o_dim: int) -> list[float]:
    """Returns (random) `o_dim` weights that sum to 1."""
    assert o_dim > 1

    weights = [random.uniform(0, 1) for _ in range(o_dim)]
    total = sum(weights)

    return [w / total for w in weights]


def log_likelihood_linear_utility(
    w: tf.Tensor, o: tf.Tensor, u: tf.Tensor, var: tf.Tensor | None = None
) -> tf.Tensor:
    """Gives the likelihood of the data given `w` weights.

    We assume `u ~ o * w + e`, which leads to
        `p(u | o, w) = N(u; m=o * w, s=e)`
    """
    # We should be getting errors if this does not hold, but I fear for broadcasting.
    assert len(o.shape) == 2 and o.shape[-1] == w.shape[0] and o.shape[0] == u.shape[0]

    if var is None:
        var = tf.constant(0.1, dtype=tf.float64)

    mean = tf.tensordot(o, tf.squeeze(w), axes=1)
    dist = tfp.distributions.Normal(mean, var)

    return tf.reduce_sum(dist.log_prob(tf.squeeze(u)))


def scalarize_objectives(
    objectives: tf.Tensor, scalarization_weights: tf.Tensor
) -> tf.Tensor:
    """Calculates (linear) combination of `objectives` given `scalarization_weights`.

    In practice, returns matrix multiplication `objectives * scalarization_weights`.

    Will cast `objectives` into [..., o_dim] to do the multiplication.
    """
    assert scalarization_weights.ndim is not None and scalarization_weights.ndim <= 2
    assert (
        objectives.ndim == 2 and objectives.shape[-1] == scalarization_weights.shape[0]
    )

    return tf.matmul(objectives, tf.reshape(scalarization_weights, (-1, 1)))


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
