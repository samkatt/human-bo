from botorch.models import SingleTaskGP
import torch
from botorch.acquisition import qMaxValueEntropy
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    UpperConfidenceBound,
    AcquisitionFunction,
)
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from botorch.test_functions import (
    Branin,
    Hartmann,
    Rosenbrock,
    SyntheticTestFunction,
    synthetic,
)
from human_bo.conf import CONFIG
import human_bo.oracles as oracles
from human_bo.test_functions import Forrester, Zhou


def pick_test_function(func: str) -> SyntheticTestFunction:
    """Instantiate the given function to optimize.

    :func: string description of the test function to return
    """

    test_function_mapping: dict[str, SyntheticTestFunction] = {
        "Forrester": Forrester(),
        "Zhou": Zhou(),
        "Hartmann": Hartmann(negate=True),
        "Branin": Branin(negate=True),
        "Rosenbrock": Rosenbrock(dim=2, negate=True, bounds=[(-5.0, 5.0), (-5.0, 5.0)]),
    }

    try:
        return test_function_mapping[func]
    except KeyError:
        raise KeyError(
            f"{func} is not an accepted test function (not in {test_function_mapping.keys()})"
        )


def pick_kernel(ker: str, dim: int) -> ScaleKernel:
    """Instantiate the given kernel.

    :ker: string representation of the kernel
    :dim: number of dimensions of the kernel
    """

    # ScaleKernel adds the amplitude hyper-parameter
    kernel_mapping: dict[str, ScaleKernel] = {
        "RBF": ScaleKernel(RBFKernel(ard_num_dims=dim)),
        "Matern": ScaleKernel(MaternKernel(ard_num_dims=dim)),
    }

    try:
        return kernel_mapping[ker]
    except:
        raise KeyError(
            f"{ker} is not an accepted kernel (not in {kernel_mapping.keys()})"
        )


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
    mes_candidate_set = torch.rand(
        mes_n_candidates, bounds.size(1), device=bounds.device, dtype=bounds.dtype
    )
    mes_candidate_set = bounds[0] + (bounds[1] - bounds[0]) * mes_candidate_set
    mes = qMaxValueEntropy(gpr, mes_candidate_set)

    acqf_mapping: dict[str, AcquisitionFunction] = {
        "UCB": UpperConfidenceBound(
            gpr, beta=0.2  # 0.2 is basic value for normalized data,
        ),
        "MES": mes,
        "EI": ExpectedImprovement(gpr, y.max()),
    }

    try:
        return acqf_mapping[acqf]
    except KeyError:
        raise KeyError(
            f"{acqf} is not an accepted acquisition function (not in {acqf_mapping.keys()})"
        )


def pick_oracle(
    o, optimal_x: list[float], problem: SyntheticTestFunction
) -> oracles.Oracle:
    """Instantiates the `Oracle` described by `o`

    :optimal_x: optimal x values
    :problem: The underlying function to optimize for
    """
    oracle_mapping = {
        "truth": oracles.truth_oracle,
        "gauss": oracles.GaussianOracle(
            optimal_x,
            problem._bounds,
        ),
    }

    try:
        return oracle_mapping[o]
    except KeyError:
        raise KeyError(
            f"{o} is not an accepted oracle (not in {oracle_mapping.keys()})"
        )
