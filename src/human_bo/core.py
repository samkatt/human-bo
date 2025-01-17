"""Main functions for running experiments"""

from typing import Any

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from human_bo import factories, reporting, test_functions
from human_bo.conf import CONFIG


def human_feedback_experiment(
    problem: str,
    user_model: str,
    kernel: str,
    acqf: str,
    n_init: int,
    seed: int,
    budget: int,
    problem_noise: float,
    report_step: reporting.StepReport,
) -> dict[str, Any]:
    """Main loop that handles all the work."""

    torch.manual_seed(seed)

    # Construct actual pieces from settings.
    problem_function = factories.pick_test_function(problem, problem_noise)
    optimal_x = CONFIG["problem"]["parser-arguments"]["choices"][problem]["optimal_x"]
    optimal_y = problem_function.optimal_value

    user = factories.pick_user_model(
        user_model,
        optimal_x[torch.randint(0, len(optimal_x), size=(1,))],
        problem_function,
    )

    ai = PlainBO(kernel, acqf, problem_function._bounds)

    print("WARNING: initial samples not generated with user model.")
    x, y = test_functions.sample_initial_points(
        problem_function, problem_function._bounds, n_init
    )

    y_to_user = y.detach().clone()  # the y-values observed by the user.
    y_from_user = y.detach().clone()  # the y-values given by the user to the AI.

    # Statistics to compute regret.
    y_true = torch.Tensor()
    y_max = -torch.inf

    # Main loop
    for i in range(budget):

        # Generate new data.
        candidates = ai.pick_queries(x, y_from_user)

        y_to_user_new = problem_function(candidates)
        y_from_user_new = user(candidates, y_to_user_new)

        # Add data to existing set.
        x = torch.cat((x, candidates))
        y_to_user = torch.cat((y_to_user, y_to_user_new))
        y_from_user = torch.cat((y_from_user, y_from_user_new))

        # Statistics for online reporting (regret).
        y_true_new = problem_function.evaluate_true(candidates)
        y_true = torch.cat((y_true, y_true_new))

        y_max = max(y_max, y_true_new.max().item())
        regret = optimal_y - y_max

        report_step({"y_true": y_true, "y_max": y_max, "regret": regret}, i)

    return {
        "x": x,
        "y": y_from_user,
        "y_true": y_to_user,
    }


def random_queries(bounds: list[tuple[float, float]], n: int = 1) -> torch.Tensor:
    """Create `n` random tensor with values within `bounds`"""
    lower, upper = torch.Tensor(bounds).T
    return torch.rand(size=[n, len(bounds)]) * (upper - lower) + lower


class PlainBO:
    def __init__(
        self,
        kernel: str,
        acqf: str,
        bounds: list[tuple[float, float]],
        num_restarts: int = 10,
        raw_samples: int = 512,
    ):
        """Creates a typical BO agent."""
        self.kernel = kernel
        self.acqf = acqf

        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

        self.bounds = torch.tensor(bounds).T
        self.dim = self.bounds.shape[1]

    def pick_queries(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Performs BO given input `x` and `y` data points."""

        gpr = SingleTaskGP(
            x,
            y,
            covar_module=factories.pick_kernel(self.kernel, self.dim),
            input_transform=Normalize(d=self.dim),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gpr.likelihood, gpr)
        fit_gpytorch_mll(mll)

        candidates, _ = optimize_acqf(
            acq_function=factories.pick_acqf(
                self.acqf, Standardize(m=1)(y)[0], gpr, self.bounds
            ),
            bounds=self.bounds,
            q=1,  # batch size, i.e. we only query one point
            num_restarts=10,
            raw_samples=512,
        )

        return candidates
