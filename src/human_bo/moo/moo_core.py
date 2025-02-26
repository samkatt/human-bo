"""Some core utilities for multi-objective optimization."""

from typing import Any, Callable

import torch
from botorch.acquisition import objective
from botorch.models import model as botorch_model
from botorch.posteriors import posterior
from botorch.posteriors import torch as torch_posterior
from torch import distributions

from human_bo import core, interaction_loops, reporting

CONFIG = {
    "preference_weights": {
        "type": float,
        "shorthand": "w",
        "help": "List of weights representing user objective preferences, must sum to one!",
        "tags": {""},
        "parser-arguments": {"nargs": "+"},
    },
    "algorithm": {
        "type": str,
        "shorthand": "x",
        "help": "Implementation of the AI agent.",
        "tags": {"experiment-parameter"},
        "parser-arguments": {
            "required": True,
            "choices": {"random", "bo"},
        },
    },
}

type UtilityFunction = Callable[[torch.Tensor], torch.Tensor]


def create_utility_function(w: list[float]) -> UtilityFunction:
    """Currently just a simple function that creates a `LinearMCObjective` from botorch."""
    return objective.LinearMCObjective(torch.Tensor(w))


class MOOEvaluation(interaction_loops.Evaluation):
    """Evaluates the true value of query and tracks regret and max y."""

    def __init__(
        self,
        moo_function,
        utility_function: UtilityFunction,
        report_step: reporting.StepReport,
    ):
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


class ObjectiveFunctionModel(botorch_model.Model):
    """Transforms a deterministic function into"""

    _num_outputs: int

    def __init__(self, f, noise_std):
        super().__init__()
        self.f = f
        self.noise_std = torch.Tensor(noise_std)
        self._num_outputs = len(noise_std)

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    def forward(self, x: torch.Tensor) -> distributions.Distribution:
        return distributions.MultivariateNormal(self.f(x), torch.diag(self.noise_std))

    def posterior(
        self,
        X: torch.Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | torch.Tensor = False,
        posterior_transform: objective.PosteriorTransform | None = None,
    ) -> posterior.Posterior:
        del observation_noise

        if output_indices:
            X = X[..., output_indices]

        ret = torch_posterior.TorchPosterior(distribution=self(X))
        if posterior_transform is not None:
            ret = posterior_transform(ret)

        return ret


def create_AI(
    moo_function,
    algorithm: str,
    kernel: str,
    acqf: str,
) -> interaction_loops.Agent:
    ai_mapping: dict[str, interaction_loops.Agent] = {
        "random": core.RandomAgent(moo_function.bounds),
        "bo": core.BO_Agent(
            moo_function._bounds, kernel, acqf, torch.Tensor(), torch.Tensor()
        ),
    }

    try:
        return ai_mapping[algorithm]
    except KeyError as error:
        raise ValueError(
            f"{algorithm} not accepted algorithm (not in {ai_mapping.keys()})"
        ) from error
