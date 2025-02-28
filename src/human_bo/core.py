"""Main functions for running experiments"""

from typing import Any, Callable

import torch
from botorch.acquisition import analytic, qLogNoisyExpectedImprovement, qMaxValueEntropy
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, model
from botorch.models.transforms import input as input_transform
from botorch.models.transforms import outcome as outcome_transform
from botorch.optim import optimize_acqf
from gpytorch import kernels
from gpytorch.mlls import ExactMarginalLogLikelihood

from human_bo import interaction_loops


def create_acqf(
    acqf: str,
    x: torch.Tensor,
    botorch_model: model.Model,
    bounds: torch.Tensor,
    **acqf_options,
) -> analytic.AcquisitionFunction:
    """Instantiate the given acqf.

    :acqf: string representation of the acquisition function to pick
    :x: observed x values.
    :botorch_model: the model (GP) used by the acquisition function
    :bounds: [x,y] bounds on the function
    :acqf_options: depend on chosen acqf. E.g. for "UCB" we expect "ucb_beta"
    """

    # create MES acquisition function
    mes_n_candidates = 100  # size of candidate set to approximate MES
    mes_candidate_set = torch.rand(mes_n_candidates, bounds.size(1))
    mes_candidate_set = bounds[0] + (bounds[1] - bounds[0]) * mes_candidate_set
    mes = qMaxValueEntropy(botorch_model, mes_candidate_set)

    acqf_mapping: dict[str, analytic.AcquisitionFunction] = {
        "UCB": analytic.UpperConfidenceBound(
            botorch_model, beta=acqf_options["ucb_beta"]
        ),
        "MES": mes,
        "EI": qLogNoisyExpectedImprovement(botorch_model, x),
    }

    try:
        return acqf_mapping[acqf]
    except KeyError as error:
        raise KeyError(
            f"{acqf} is not an accepted acquisition function (not in {acqf_mapping.keys()})"
        ) from error


def pick_kernel(ker: str, dim: int) -> kernels.ScaleKernel:
    """Instantiate the given kernel.

    :ker: string representation of the kernel
    :dim: number of dimensions of the kernel
    """

    # ScaleKernel adds the amplitude hyper-parameter.
    kernel_mapping: dict = {
        "RBF": kernels.ScaleKernel(kernels.RBFKernel(ard_num_dims=dim)),
        "Matern": kernels.ScaleKernel(kernels.MaternKernel(ard_num_dims=dim)),
        "Default": None,
    }

    try:
        return kernel_mapping[ker]
    except KeyError as error:
        raise KeyError(
            f"{ker} is not an accepted kernel (not in {kernel_mapping.keys()})"
        ) from error


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


class RandomAgent(interaction_loops.Agent):
    def __init__(self, bounds):
        self.bounds = bounds

    def pick_query(self) -> tuple[Any, dict[str, Any]]:
        random_query = random_queries(self.bounds)
        return random_query, {}

    def observe(self, query, feedback, evaluation) -> None:
        del query, feedback, evaluation


def fit_gp(x, y, kernel, input_bounds: torch.Tensor | None = None) -> SingleTaskGP:
    """My go-to function for fitting GPs.

    Will normalize input (`x`) and standardize output (`y`).
    """
    assert y.dim() == 1 and x.dim() == 2

    dim = x.shape[-1]

    gp = SingleTaskGP(
        x,
        y.unsqueeze(-1),
        covar_module=kernel,
        input_transform=input_transform.Normalize(d=dim, bounds=input_bounds),
        outcome_transform=outcome_transform.Standardize(m=1),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))

    return gp


class PlainBO:
    """Bayesian optimization agent."""

    def __init__(
        self,
        kernel: str,
        acqf: str,
        bounds: list[tuple[float, float]],
        acqf_options: dict[str, Any],
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

        self.acqf_options = acqf_options

    def pick_queries(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs BO given input `x` and `y` data points."""
        if x.nelement() == 0:
            print(
                "WARN: PlainBO::pick_queries is returning randomly because of empty x."
            )
            return random_queries(self.bounds), torch.Tensor(0)

        gp = fit_gp(x, y, pick_kernel(self.kernel, self.dim), self.bounds)

        candidates, acqf_val = optimize_acqf(
            acq_function=create_acqf(
                self.acqf, x, gp, self.bounds, **self.acqf_options
            ),
            bounds=self.bounds,
            q=1,  # batch size, i.e. we only query one point
            num_restarts=10,
            raw_samples=512,
        )

        return candidates, acqf_val


class BO_Agent(interaction_loops.Agent):
    """Simple Bayes optimization Agent."""

    def __init__(
        self,
        bounds,
        kernel: str,
        acqf: str,
        x_init: torch.Tensor,
        y_init: torch.Tensor,
        acqf_options: dict[str, Any],
    ):
        self.bo = PlainBO(kernel, acqf, bounds, acqf_options)
        self.x, self.y = x_init, y_init

    def pick_query(self) -> tuple[Any, dict[str, Any]]:
        query, val = self.bo.pick_queries(self.x, self.y)
        return query, {"acqf_value": val}

    def observe(self, query, feedback, evaluation) -> None:
        del evaluation

        self.x = torch.cat((self.x, query))
        self.y = torch.cat((self.y, feedback))
