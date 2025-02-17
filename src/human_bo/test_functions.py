"""Test functions that are not implemented in BoTorch."""

from typing import List, Optional, Tuple

import torch
from botorch import test_functions
from torch import Tensor, sin


class Zhou(test_functions.SyntheticTestFunction):
    """The Zhou (1-dimensional) function (https://www.sfu.ca/~ssurjano/zhou98.html)"""

    _optimal_value = 2.002595246981888

    def __init__(
        self,
        noise_std: Optional[float] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of the observation noise.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = 1
        self._bounds = [(-0.0, 1.0) for _ in range(self.dim)]
        self._optimizers = [tuple(1 / 3 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, bounds=bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        def phi_zou(X: Tensor) -> Tensor:
            return (2 * torch.pi) ** (-0.5) * torch.exp(-0.5 * X**2)

        part1 = 10 * (X - 1 / 3)
        part2 = 10 * (X - 2 / 3)
        return 5 * (phi_zou(part1) + phi_zou(part2))


class Forrester(test_functions.SyntheticTestFunction):
    """The Forrester (1-dimensional) function (https://www.sfu.ca/~ssurjano/forretal08.html)"""

    def __init__(
        self,
        noise_std: Optional[float] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """Initiates the Forrester function

        Forrester, A., Sobester, A., & Keane, A. (2008).
        Engineering design via surrogate modelling: a practical guide. Wiley.

        The actual function is:
            `f(x) = (6x - 2)^2 * sin(12x - 4)`

        I am negating this by nature, so returning `- f(x)`

        :noise_std: Standard deviation of the observation noise.
        :bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = 1
        self._bounds = [(-0.0, 1.0) for _ in range(self.dim)]
        self._optimizers = [tuple(1 / 3 for _ in range(self.dim))]

        self._optimal_value = 6.020738786441099
        super().__init__(noise_std=noise_std, bounds=bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return -((6 * X - 2) ** 2) * sin(12 * X - 4)


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
        "Rosenbrock": test_functions.Rosenbrock(
            dim=2, negate=True, bounds=[(-5.0, 5.0), (-5.0, 5.0)], noise_std=noise
        ),
    }

    try:
        return test_function_mapping[func]
    except KeyError as error:
        raise KeyError(
            f"{func} is not an accepted test function (not in {test_function_mapping.keys()})"
        ) from error
