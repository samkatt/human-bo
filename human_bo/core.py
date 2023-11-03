"""Main functions for running experiments (`eval_model`)"""

import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Any

from human_bo import factories
from human_bo.conf import CONFIG


def eval_model(
    function: str,
    user_model: str,
    kernel: str,
    acqf: str,
    n_init: int,
    seed: int,
    budget: int,
) -> dict[str, Any]:
    """Main loop that handles all the work."""

    torch.manual_seed(seed)

    # Construct actual pieces from settings.
    problem = factories.pick_test_function(function)
    bounds = torch.tensor(problem._bounds).T
    dim = bounds.shape[1]
    optimal_x = CONFIG["function"]["choices"][function]["optimal_x"]

    observation_function = factories.pick_user_model(
        user_model, optimal_x[torch.randint(0, len(optimal_x), size=(1,))], problem
    )

    # Initial training.
    # TODO: hardcoded noise level, but should be function-dependent.
    sigma = 0.01
    train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_init, dim)
    true_Y = problem(train_X).view(-1, 1)
    train_Y = observation_function(train_X, true_Y)
    train_Y = train_Y + sigma * torch.randn(size=train_Y.shape)

    gpr = SingleTaskGP(
        train_X,
        train_Y,
        covar_module=factories.pick_kernel(kernel, dim),
        input_transform=Normalize(d=dim),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gpr.likelihood, gpr)
    fit_gpytorch_model(mll)

    # Main loop
    for _ in range(budget):
        print(".", end="", flush=True)

        candidates, _ = optimize_acqf(
            acq_function=factories.pick_acqf(
                acqf, Standardize(m=1)(train_Y)[0], gpr, bounds
            ),
            bounds=bounds,
            q=1,  # batch size, i.e. we only query one point
            num_restarts=10,
            raw_samples=512,
        )

        true_y = problem(candidates)
        train_y = observation_function(candidates, true_y)

        train_X = torch.cat((train_X, candidates))
        train_Y = torch.cat((train_Y, train_y.view(-1, 1)))
        true_Y = torch.cat((true_Y, true_y.view(-1, 1)))

        gpr = SingleTaskGP(
            train_X,
            train_Y,
            covar_module=factories.pick_kernel(kernel, dim),
            input_transform=Normalize(d=dim),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gpr.likelihood, gpr)
        fit_gpytorch_model(mll)

    print("")

    return {
        "train_X": train_X,
        "train_Y": train_Y,
        "true_Y": true_Y,
    }
