"""Various 'constructor' functions to initiate components according to program inputs"""

import torch
from botorch.acquisition import qMaxValueEntropy
from botorch.acquisition.analytic import (
    AcquisitionFunction,
    LogExpectedImprovement,
    UpperConfidenceBound,
)
from botorch.models import SingleTaskGP
from botorch.test_functions import Branin, Hartmann, Rosenbrock, SyntheticTestFunction
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel

from human_bo.test_functions import Forrester, Zhou


def pick_test_function(func: str, noise: float) -> SyntheticTestFunction:
    """Instantiate the given function to optimize.

    :func: string description of the test function to return
    :noise: standard deviation of the noise
    """

    test_function_mapping: dict[str, SyntheticTestFunction] = {
        "Forrester": Forrester(noise_std=noise),
        "Zhou": Zhou(noise_std=noise),
        "Hartmann": Hartmann(negate=True, noise_std=noise),
        "Branin": Branin(negate=True, noise_std=noise),
        "Rosenbrock": Rosenbrock(
            dim=2, negate=True, bounds=[(-5.0, 5.0), (-5.0, 5.0)], noise_std=noise
        ),
    }

    try:
        return test_function_mapping[func]
    except KeyError as error:
        raise KeyError(
            f"{func} is not an accepted test function (not in {test_function_mapping.keys()})"
        ) from error


def pick_kernel(ker: str, dim: int) -> ScaleKernel:
    """Instantiate the given kernel.

    :ker: string representation of the kernel
    :dim: number of dimensions of the kernel
    """

    # ScaleKernel adds the amplitude hyper-parameter.
    kernel_mapping: dict = {
        "RBF": ScaleKernel(RBFKernel(ard_num_dims=dim)),
        "Matern": ScaleKernel(MaternKernel(ard_num_dims=dim)),
        "Default": None,
    }

    try:
        return kernel_mapping[ker]
    except KeyError as error:
        raise KeyError(
            f"{ker} is not an accepted kernel (not in {kernel_mapping.keys()})"
        ) from error


def pick_acqf(
    acqf: str, y: torch.Tensor, gpr: SingleTaskGP, bounds: torch.Tensor
) -> AcquisitionFunction:
    """Instantiate the given acqf.

    :acqf: string representation of the acquisition function to pick
    :y: initial y values (probably to compute the max)
    :gpr: the GP used by the acquisition function
    :bounds: [x,y] bounds on the function
    """

    # create MES acquisition function
    mes_n_candidates = 100  # size of candidate set to approximate MES
    mes_candidate_set = torch.rand(mes_n_candidates, bounds.size(1))
    mes_candidate_set = bounds[0] + (bounds[1] - bounds[0]) * mes_candidate_set
    mes = qMaxValueEntropy(gpr, mes_candidate_set)

    acqf_mapping: dict[str, AcquisitionFunction] = {
        "UCB": UpperConfidenceBound(gpr, beta=0.2),
        "MES": mes,
        "EI": LogExpectedImprovement(gpr, y.max()),
    }

    try:
        return acqf_mapping[acqf]
    except KeyError as error:
        raise KeyError(
            f"{acqf} is not an accepted acquisition function (not in {acqf_mapping.keys()})"
        ) from error
