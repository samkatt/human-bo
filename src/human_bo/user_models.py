"""Implements user models for BO

A user model here is just something that determines the `y` value that is seen
by the agent. This could just be the true `y` value of the function of which we
are trying to find the mean ("truth"), but also a noisy observation, or human
interpretation.
"""

from collections.abc import Callable

import torch
from torch.distributions import Distribution, multivariate_normal, normal

type UserModel = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def oracle(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Returns `y` unmodified (implementation of `UserModel`)"""
    return y


class GaussianUserModel:
    """The `UserModel` that returns y from a Gaussian around the max x."""

    def __init__(self, optimal_x: list[float], bounds: list[tuple[float, float]]):
        """Creates the Gaussian around `optimal_x` of deviation `s`

        :optimal_x: the optimal value(s) for x
        :bounds: The x bounds (min/max of each dimension)
        """
        assert len(bounds) == len(optimal_x)

        sigma = [(max - min) / 5 for min, max in bounds]
        dim = len(optimal_x)

        self.y_multiplier = float(dim)

        if dim == 1:
            self.gauss: Distribution = normal.Normal(
                torch.tensor(optimal_x), scale=sigma[0]
            )
        else:
            self.gauss = multivariate_normal.MultivariateNormal(
                torch.tensor(optimal_x), torch.diag(torch.tensor(sigma))
            )

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Returns `Normal(x)`

        :x: the input value to our Gaussian
        :y: ignored
        """
        return torch.exp(self.gauss.log_prob(x)) * self.y_multiplier
