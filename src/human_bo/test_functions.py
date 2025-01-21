"""Test functions that are not implemented in BoTorch."""

from typing import Callable, List, Optional, Tuple

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from torch import Tensor, sin


class Zhou(SyntheticTestFunction):
    """The Zhou (1-dimensional) function (https://www.sfu.ca/~ssurjano/zhou98.html)"""

    _optimal_value = 2.002595246981888

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = 1
        self._bounds = [(-0.0, 1.0) for _ in range(self.dim)]
        self._optimizers = [tuple(1 / 3 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        def phi_zou(X: Tensor) -> Tensor:
            return (2 * torch.pi) ** (-0.5) * torch.exp(-0.5 * X**2)

        part1 = 10 * (X - 1 / 3)
        part2 = 10 * (X - 2 / 3)
        return 5 * (phi_zou(part1) + phi_zou(part2))


class Forrester(SyntheticTestFunction):
    """The Forrester (1-dimensional) function (https://www.sfu.ca/~ssurjano/forretal08.html)"""

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = True,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """Initiates the Forrester function

        Forrester, A., Sobester, A., & Keane, A. (2008).
        Engineering design via surrogate modelling: a practical guide. Wiley.

        The actual function is:
            `f(x) = (6x - 2)^2 * sin(12x - 4)`

        :noise_std: Standard deviation of the observation noise.
        :negate: If True, negate the function.
        :bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = 1
        self._bounds = [(-0.0, 1.0) for _ in range(self.dim)]
        self._optimizers = [tuple(1 / 3 for _ in range(self.dim))]

        self._optimal_value = -6.020738786441099 if negate else 15.829731945974109
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return (6 * X - 2) ** 2 * sin(12 * X - 4)


def random_queries(bounds: list[tuple[float, float]], n: int = 1) -> torch.Tensor:
    """Create `n` random tensor with values within `bounds`"""
    lower, upper = torch.Tensor(bounds).T
    return torch.rand(size=[n, len(bounds)]) * (upper - lower) + lower


def sample_initial_points(
    f: Callable[[torch.Tensor], torch.Tensor],
    input_bounds: list[tuple[float, float]],
    n_init: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates `n_init` random `x -> y` values.

    Will return them as a `(x, y)` tuple.
    """
    assert isinstance(n_init, int) and n_init > 0

    x = random_queries(input_bounds, n_init)
    y = f(x)

    return x, y
