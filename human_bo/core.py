"""Main functions for running experiments"""

from typing import Any

import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from human_bo import factories
from human_bo.conf import CONFIG


def human_feedback_experiment(
    function: str,
    user_model: str,
    kernel: str,
    acqf: str,
    n_init: int,
    seed: int,
    budget: int,
    function_noise: float,
) -> dict[str, Any]:
    """Main loop that handles all the work."""

    torch.manual_seed(seed)

    # Construct actual pieces from settings.
    problem = factories.pick_test_function(function)
    bounds = torch.tensor(problem._bounds).T
    dim = bounds.shape[1]
    optimal_x = CONFIG["function"]["parser-arguments"]["choices"][function]["optimal_x"]

    observation_function = factories.pick_user_model(
        user_model, optimal_x[torch.randint(0, len(optimal_x), size=(1,))], problem
    )

    # Initial training.
    train_x = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_init, dim)
    true_y = problem(train_x).view(-1, 1)
    train_y = observation_function(train_x, true_y)
    train_y = train_y + function_noise * torch.randn(size=train_y.shape)

    # Main loop
    for _ in range(budget):
        print(".", end="", flush=True)

        gpr = SingleTaskGP(
            train_x,
            train_y,
            covar_module=factories.pick_kernel(kernel, dim),
            input_transform=Normalize(d=dim),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gpr.likelihood, gpr)
        fit_gpytorch_model(mll)

        candidates, _ = optimize_acqf(
            acq_function=factories.pick_acqf(
                acqf, Standardize(m=1)(train_y)[0], gpr, bounds
            ),
            bounds=bounds,
            q=1,  # batch size, i.e. we only query one point
            num_restarts=10,
            raw_samples=512,
        )

        new_true_y = problem(candidates)
        # TODO: add noise?
        new_train_y = observation_function(candidates, new_true_y)

        train_x = torch.cat((train_x, candidates))
        train_y = torch.cat((train_y, new_train_y.view(-1, 1)))
        true_y = torch.cat((true_y, new_true_y.view(-1, 1)))

    print("")

    return {
        "train_x": train_x,
        "train_y": train_y,
        "true_y": true_y,
    }
