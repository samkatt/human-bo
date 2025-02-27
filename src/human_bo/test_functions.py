"""Test functions that are not implemented in BoTorch."""

import torch
from botorch import test_functions
from botorch.test_functions import base


class Zhou(test_functions.SyntheticTestFunction):
    """The Zhou (1-dimensional) function (https://www.sfu.ca/~ssurjano/zhou98.html)"""

    dim = 1
    _bounds = [(-0.0, 1.0)]
    _optimizers = [tuple([1 / 3]), tuple([2 / 3])]
    _optimal_value = 2.002595246981888

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        def phi_zou(X: torch.Tensor) -> torch.Tensor:
            return (2 * torch.pi) ** (-0.5) * torch.exp(-0.5 * X**2)

        part1 = 10 * (X[:, 0] - 1 / 3)
        part2 = 10 * (X[:, 0] - 2 / 3)
        return 5 * (phi_zou(part1) + phi_zou(part2))


class Forrester(test_functions.SyntheticTestFunction):
    """The Forrester (1-dimensional) function (https://www.sfu.ca/~ssurjano/forretal08.html)"""

    dim = 1
    _bounds = [(-0.0, 1.0)]
    _optimizers = [tuple([0.7572])]
    _optimal_value = 6.020738786441099

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        return -((6 * X[:, 0] - 2) ** 2) * torch.sin(12 * X[:, 0] - 4)


def pick_test_function(func: str, noise: float) -> test_functions.SyntheticTestFunction:
    """Instantiate the given function to optimize.

    :func: string description of the test function to return
    :noise: standard deviation of the noise
    """

    test_function_mapping: dict[str, test_functions.SyntheticTestFunction] = {
        "Forrester": Forrester(noise_std=noise),
        "Zhou": Zhou(noise_std=noise),
        "Hartmann": test_functions.Hartmann(negate=True, noise_std=noise),
        "Branin": test_functions.Branin(negate=True, noise_std=noise),
        "Rosenbrock2D": test_functions.Rosenbrock(
            dim=2, negate=True, bounds=[(-5.0, 5.0), (-5.0, 5.0)], noise_std=noise
        ),
        "Ackley1D": test_functions.Ackley(dim=1, noise_std=noise, negate=True),
        "DixonPrice1D": test_functions.DixonPrice(dim=1, noise_std=noise, negate=True),
        "Griewank1D": test_functions.Griewank(dim=1, noise_std=noise, negate=True),
        "Levy1D": test_functions.Levy(dim=1, noise_std=noise, negate=True),
        "Rastrigin1D": test_functions.Rastrigin(dim=1, noise_std=noise, negate=True),
        "StyblinskiTang1D": test_functions.StyblinskiTang(
            dim=1, noise_std=noise, negate=True
        ),
    }

    try:
        return test_function_mapping[func]
    except KeyError as error:
        raise KeyError(
            f"{func} is not an accepted (single objective) test function (not in {test_function_mapping.keys()})"
        ) from error


def pick_moo_test_function(
    func: str, noise: list[float] | None
) -> base.MultiObjectiveTestProblem:
    """Instantiate the given multi-objective function to optimize.

    :func: string description of the test function to return
    :noise: standard deviations of the noise, None means no noise.
    """

    test_function_mapping: dict[str, base.MultiObjectiveTestProblem] = {
        "BraninCurrin": test_functions.BraninCurrin(noise_std=noise),
    }

    try:
        return test_function_mapping[func]
    except KeyError as error:
        raise KeyError(
            f"{func} is not an accepted MOO test function (not in {test_function_mapping.keys()})"
        ) from error
