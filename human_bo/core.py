from typing import Any
import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from human_bo import factories
from human_bo.conf import CONFIG


def eval_model(
    function: str,
    oracle: str,
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

    observation_function = factories.pick_oracle(
        oracle, optimal_x[torch.randint(0, len(optimal_x), size=(1,))], problem
    )
    K = factories.pick_kernel(kernel, dim)

    # Initial training.
    # TODO: hardcoded noise level, but should be function-dependent.
    sigma = 0.01
    train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_init, dim)
    true_Y = problem(train_X).view(-1, 1)
    train_Y = observation_function(train_X, true_Y)
    train_Y = train_Y + sigma * torch.randn(size=train_Y.shape)

    gpr = SingleTaskGP(train_X, train_Y, covar_module=K)
    mll = ExactMarginalLogLikelihood(gpr.likelihood, gpr)
    fit_gpytorch_model(mll, max_retries=10)

    ##### BO LOOP
    regrets = torch.zeros(budget + 1)
    regrets[0] = problem.optimal_value - train_Y.max()
    for b in range(budget):
        print(f"Query {b}")

        candidates, _ = optimize_acqf(
            acq_function=factories.pick_acqf(acqf, train_Y, gpr, bounds),
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

        regrets[b + 1] = problem.optimal_value - train_Y.max()

        gpr = SingleTaskGP(train_X, train_Y, covar_module=K)
        mll = ExactMarginalLogLikelihood(gpr.likelihood, gpr)
        fit_gpytorch_model(mll, max_retries=10)

    return {
        "train_X": train_X,
        "train_Y": train_Y,
        "regrets": regrets,
        "true_Y": true_Y,
    }
