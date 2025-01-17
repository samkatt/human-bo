"""Specific functions for human giving feedback experiments"""

from collections.abc import Callable

import torch
from torch import distributions

from human_bo import conf


def update_config():
    """This function updates the configurations to set up for human feedback experiments

    This needs to be run at the start of any script on these type of experiments.
    """
    # Add `user_model` as a configuration.
    conf.CONFIG["user_model"] = {
        "type": str,
        "shorthand": "u",
        "help": "The mechanism through which queries are given.",
        "tags": {"experiment-parameter"},
        "parser-arguments": {"choices": {"oracle", "gauss"}},
    }

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
            self.gauss: distributions.Distribution = distributions.normal.Normal(
                torch.tensor(optimal_x), scale=sigma[0]
            )
        else:
            self.gauss = distributions.multivariate_normal.MultivariateNormal(
                torch.tensor(optimal_x), torch.diag(torch.tensor(sigma))
            )

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Returns `Normal(x)`

        :x: the input value to our Gaussian
        :y: ignored
        """
        return torch.exp(self.gauss.log_prob(x)) * self.y_multiplier
