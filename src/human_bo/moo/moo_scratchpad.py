"""Testing and example code for multi-objective optimization."""

from typing import Any, Callable

import torch
from botorch.acquisition import objective
from botorch.test_functions import multi_objective as moo_test_functions

from human_bo import core, interaction_loops, reporting

type UtilityFunction = Callable[[torch.Tensor], torch.Tensor]

CONFIG = {
    "preference_weights": {
        "type": float,
        "shorthand": "w",
        "help": "List of weights representing user objective preferences, must sum to one!",
        "tags": {""},
        "parser-arguments": {"nargs": "+"},
    }
}


def moo_learn_subjective_function():
    """Experiment on multi-objective optimization where the only unknown is one (unmeasurable) objective function."""
    # Parameters.
    noise_std = [0.2, 0.4]
    test_function = moo_test_functions.BraninCurrin
    budget = 20
    # unknown_output_idx = 0

    # n_init = 6
    # num_restarts_acqf = 8
    # raw_samples_acqf = 128

    # Setup problem.
    moo_function = test_function(noise_std)

    preference_weights = core.random_queries(
        [(0, 1) for _ in range(moo_function.dim)]
    ).squeeze()
    preference_weights = preference_weights / preference_weights.sum()

    utility_function = objective.LinearMCObjective(preference_weights)

    # Setup interaction components.
    agent = core.RandomAgent(moo_function.bounds)
    problem = MOOProblem(moo_function, utility_function)
    evaluation = MOOEvaluation(moo_function, utility_function, reporting.print_dot)

    print("Starting subjective MOO loop...")
    res = interaction_loops.basic_loop(agent, problem, evaluation, budget)
    print("Ended subjective MOO loop!")

    return res


class MOOProblem(interaction_loops.Problem):
    """Typical MOO problem."""

    def __init__(self, moo_function, utility_function: UtilityFunction):
        self.moo_function = moo_function
        self.utility_function = utility_function

    def give_feedback(self, query) -> tuple[Any, dict[str, Any]]:
        objectives = self.moo_function(query)
        utility = self.utility_function(objectives)

        return utility, {"objectives": objectives}

    def observe(self, query, feedback, evaluation) -> None:
        del query, feedback, evaluation


class MOOEvaluation(interaction_loops.Evaluation):
    """Evaluates the true value of query and tracks regret and max y."""

    def __init__(self, moo_function, utility_function: UtilityFunction, report_step):
        self.moo_function = moo_function
        self.utility_function = utility_function
        self.u_max = -torch.inf
        self.report_step = report_step
        self.step = 0

    def __call__(
        self,
        query,
        feedback,
        query_stats: dict[str, Any],
        feedback_stats: dict[str, Any],
        **kwargs,
    ) -> tuple[Any, dict[str, Any]]:
        del feedback, kwargs

        objectives_true = self.moo_function(query, noise=False)
        utility_true = self.utility_function(objectives_true)

        self.u_max = max(self.u_max, utility_true.max().item())
        # TODO: implement regret tracking.

        evaluation = {
            "utility_true": utility_true,
            "y_max": self.u_max,
            "objectives_true": objectives_true,
            "query_stats": query_stats,
            "feedback_stats": feedback_stats,
        }

        self.step += 1
        self.report_step(evaluation, self.step)

        return None, evaluation
