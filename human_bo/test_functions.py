"""Test functions that are not implemented in BoTorch"""

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
            return (2 * torch.pi) ** (-0.5) * torch.exp(-0.5 * X ** 2)

        part1 = 10 * (X - 1 / 3)
        part2 = 10 * (X - 2 / 3)
        return 5 * (phi_zou(part1) + phi_zou(part2))


class Forrester(SyntheticTestFunction):
    """The Forrester (1-dimensional) function (https://www.sfu.ca/~ssurjano/forretal08.html)"""

    _optimal_value = 15.829731945974109  # Hand-computed `evaluate_true(1.0)`

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
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return (6 * X - 2) ** 2 * sin(12 * X - 4)


def sample_initial_points(
    f: Callable[[torch.Tensor], torch.Tensor],
    input_bounds: list[tuple[float, float]],
    n_init: int,
    problem_noise: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates `n_init` random `x -> y` values.

    Will return them as a `(x, train_y, true_y)` tuple.
    In particular:
        - `x` will be `n_init` values uniformly sampled within the `bounds`.
        - `y_train[i]` will be the observed value of `x[i]` (`y_true[i]` plus Gaussian noise with deviation `problem_noise`).
        - `y_true` is the true value of `f(x[i])`.
    """
    assert isinstance(n_init, int) and n_init > 0
    assert isinstance(problem_noise, float) and problem_noise >= 0

    bounds = torch.tensor(input_bounds).T
    dim = bounds.shape[1]

    train_x = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_init, dim)
    true_y = f(train_x).view(-1, 1)
    train_y = true_y + torch.normal(0, problem_noise, size=true_y.shape)

    return train_x, train_y, true_y
