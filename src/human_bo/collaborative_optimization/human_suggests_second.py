"""Human *may* suggest a second query given a suggestion of the AI

In particular, here we consider the setup in which both human and AI are interested in optimizing a function `f`.
The human may know something about it, though the AI is (presumably) better at optimization.

Per usual, we consider the instance where the budget is the main cost, so we care for sample efficiency.
As a result, whether the human suggests a ...
"""

from typing import Any

import torch

from human_bo import core, interaction_loops

CONFIG = {
    "user": {
        "type": str,
        "shorthand": "u",
        "help": "The (real) user behavior.",
        "tags": {"experiment-parameter"},
        "parser-arguments": {"choices": {"random", "bo", "noop"}, "required": True},
    }
}


class BayesOptUser(interaction_loops.User):
    def __init__(
        self, bounds, kernel, acqf, x_init, y_init, acqf_options: dict[str, Any]
    ):
        self.bo = core.PlainBO(kernel, acqf, bounds, acqf_options)
        self.x, self.y = x_init, y_init

    def pick_action(self, query) -> tuple[Any, dict[str, Any]]:
        user_query, user_query_stats = self.bo.pick_queries(self.x, self.y)
        return torch.cat((query, user_query)), user_query_stats

    def observe(self, action, feedback, evaluation) -> None:
        del evaluation

        self.x = torch.cat((self.x, action))
        self.y = torch.cat((self.y, feedback["feedback"]))


class NoopUser(interaction_loops.User):
    def pick_action(self, query) -> tuple[Any, dict[str, Any]]:
        return query, {}

    def observe(self, action, feedback, evaluation) -> None:
        del action, feedback, evaluation


class RandomUser(interaction_loops.User):
    def __init__(self, bounds):
        self.bounds = bounds

    def pick_action(self, query) -> tuple[Any, dict[str, Any]]:
        action = torch.cat((query, core.random_queries(self.bounds)))
        return action, {}

    def observe(self, action, feedback, evaluation) -> None:
        del action, feedback, evaluation


def create_user(
    user: str, f, kernel: str, acqf: str, n_init: int, acqf_options: dict[str, Any]
) -> tuple[interaction_loops.User, dict[str, Any]]:
    """Creates a user for the `interaction_loops.basic_loop` (`interaction_loops.User`) expects."""
    if user == "random":
        return RandomUser(f._bounds), {}
    if user == "bo":
        x_init, y_init = core.sample_initial_points(f, f._bounds, n_init)
        return BayesOptUser(
            f._bounds,
            kernel,
            acqf,
            x_init,
            y_init,
            acqf_options,
        ), {"initial_points": {"x": x_init, "y": y_init}}
    if user == "noop":
        return NoopUser(), {}

    raise ValueError(f"No user model for {user}")
