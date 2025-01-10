"""Human *may* suggest a second query given a suggestion of the AI

In particular, here we consider the setup in which both human and AI are interested in optimizing a function `f`.
The human may know something about it, though the AI is (presumably) better at optimization.

Per usual, we consider the instance where the budget is the main cost, so we care for sample efficiency.
As a result, whether the human suggests a ...
"""

import torch

from typing import Callable


def create_test_both_queries_problem(
    f: Callable[[torch.Tensor], torch.Tensor], problem_noise: float
):
    """Creates a "problem" (for BO) for human suggests second problem.

    We expect this problem to take two actions (queries),
    and return two observations (y + noise) and some diagnostics.

    The argument `f` is the `x -> y` function to be optimized.
    """

    def problem_step(x_ai, x_human):
        y_ai = f(x_ai)
        y_human = f(x_human)

        observation_ai = y_ai + torch.normal(0, problem_noise, size=y_ai.shape)
        observation_human = y_human + torch.normal(0, problem_noise, size=y_human.shape)

        return {"y_ai": observation_ai, "y_human": observation_human}, {
            "true_ai": y_ai,
            "observed_ai": observation_ai,
            "true_human": y_human,
            "observed_human": observation_human,
        }

    return problem_step


class PlainJointAI:
    def __init__(
        self,
        plain_ai: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        init_x,
        init_y,
    ):
        self.init_x = init_x
        self.init_y = init_y
        self.bo = plain_ai

    def pick_queries(self, history: list) -> tuple[torch.Tensor, dict]:
        """Returns next query `x` given history

        In practice will combine initial data points `self.init_x` and `self.init_y` 
        with those observed in `history` to do some Bayesian optimization and optimize some acquisition function.

        See `create_test_both_queries_problem` and `core.ai_then_human_optimization_experiment`
        for expected API.
        """

        # Base case has no history. Perform BO on the initial points.
        if not history:
            return self.bo(self.init_x, self.init_y), {
                "stats": "Joint AI agent has no stats implemented yet."
            }

        # At this point, we know there is some history:
        # Concatenate the observed x's and y's to our initial samples.
        x = torch.cat(
            (
                self.init_x,
                torch.cat(
                    [x for t in history for x in [t["ai_action"], t["human_action"]]]
                ),
            )
        )
        y = torch.cat(
            (
                self.init_y,
                torch.cat(
                    [
                        y
                        for t in history
                        for y in [t["outcome"]["y_ai"], t["outcome"]["y_human"]]
                    ]
                ),
            )
        )

        return self.bo(x, y), {"stats": "Joint AI agent has no stats implemented yet."}
