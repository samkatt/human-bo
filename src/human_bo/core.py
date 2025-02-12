"""Main functions for running experiments"""

from typing import Callable

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from human_bo import factories


def random_queries(
    bounds: list[tuple[float, float]] | torch.Tensor, n: int = 1
) -> torch.Tensor:
    """Create `n` random tensor with values within `bounds`"""
    assert isinstance(n, int)

    if not torch.is_tensor(bounds):
        bounds = torch.Tensor(bounds).T

    lower, upper = bounds
    return torch.rand(size=[n, bounds.shape[1]]) * (upper - lower) + lower


def sample_initial_points(
    f: Callable[[torch.Tensor], torch.Tensor],
    input_bounds: list[tuple[float, float]] | torch.Tensor,
    n_init: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates `n_init` random `x -> y` values.

    Will return them as a `(x, y)` tuple.
    """
    assert isinstance(n_init, int)

    x = random_queries(input_bounds, n_init)
    y = f(x)

    return x, y


class PlainBO:
    """Bayesian optimization agent."""

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
        if len(x) == 0:
            print(
                "WARN: PlainBO::pick_queries is returning randomly because of empty x."
            )
            return random_queries(self.bounds)

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
