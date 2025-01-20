"""Specific functions for human giving feedback experiments"""

from collections.abc import Callable

import torch
from botorch.test_functions import SyntheticTestFunction
from torch import distributions

CONFIG = {
    "user_model": {
        "type": str,
        "shorthand": "u",
        "help": "The mechanism through which queries are given.",
        "tags": {"experiment-parameter"},
        "parser-arguments": {"choices": {"oracle", "gauss"}},
    }
}


type UserModel = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def oracle(_x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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

    def __call__(self, x: torch.Tensor, _y: torch.Tensor) -> torch.Tensor:
        """Returns `Normal(x)`

        :x: the input value to our Gaussian
        :_y: ignored
        """
        return torch.exp(self.gauss.log_prob(x)) * self.y_multiplier


def pick_user_model(
    u, optimal_x: list[float], problem: SyntheticTestFunction
) -> UserModel:
    """Instantiates the `UserModel` described by `u`

    :optimal_x: optimal x values
    :problem: The underlying function to optimize for
    """
    user_model_mapping: dict[str, UserModel] = {
        "oracle": oracle,
        "gauss": GaussianUserModel(
            optimal_x,
            problem._bounds,
        ),
    }

    try:
        return user_model_mapping[u]
    except KeyError as error:
        raise KeyError(
            f"{u} is not an accepted user model (not in {user_model_mapping.keys()})"
        ) from error
