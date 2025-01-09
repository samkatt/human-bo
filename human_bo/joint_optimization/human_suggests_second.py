"""Human *may* suggest a second query given a suggestion of the AI

In particular, here we consider the setup in which both human and AI are interested in optimizing a function `f`.
The human may know something about it, though the AI is (presumably) better at optimization.

Per usual, we consider the instance where the budget is the main cost, so we care for sample efficiency.
As a result, whether the human suggests a ...
"""

import torch

from human_bo import factories


def create_test_both_queries_problem(problem: str, problem_noise: float):
    """Creates a "problem" (for BO) for human suggests second problem.

    We expect this problem to take two actions (queries),
    and return two observations (y + noise) and some diagnostics.
    """
    f = factories.pick_test_function(problem)

    def problem_step(x_ai, x_human):
        y_ai = f(x_ai)
        y_human = f(x_human)

        observation_ai = y_ai + problem_noise * torch.randn(size=y_ai.shape)
        observation_human = y_human + problem_noise * torch.randn(size=y_human.shape)

        return (observation_ai, observation_human), {
            "true_1": y_ai,
            "observed_1": observation_ai,
            "true_human": y_human,
            "observed_human": observation_human,
        }

    return problem_step
