"""Implements user models for BO

A user model here is just something that determines the `y` value that is seen
by the agent. This could just be the true `y` value of the function of which we
are trying to find the mean ("truth"), but also a noisy observation, or human
interpretation.
"""

from typing import Protocol

import torch
from torch.distributions import Distribution, multivariate_normal, normal


class UserModel(Protocol):
    """Describes the API of a user model.

    In this case, all we need it to do is take `y` values and give their observation.
    """

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Our API: return the 'observed' value at `x` where `y` was the value of real `f(x)`"""
        ...


def oracle(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Returns the true values `y` as an oracle

    :x: ignored
    :y: The true underlying y values of the problem to observe
    :returns: observations of `y`
    """
    return y


class GaussianUserModel(UserModel):
    """The user model that returns y from a Gaussian around the max x."""

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
