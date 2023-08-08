"""Implements oracles in BO

An "oracle" here is just something that determines the `y` value that is seen
by the agent. This could just be the true `y` value of the function of which we
are trying to find the mean ("truth"), but also a noisy observation, or human 
interpretation.
"""

from typing import Protocol

from botorch.acquisition.analytic import torch


class Oracle(Protocol):
    """Describes the API of an oracle.

    In this case, all we need it to do is take `y` values and give their observation.
    """

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        ...

def truth_oracle(y: torch.Tensor) -> torch.Tensor:
    """Returns the true values `y` as an oracle

    :y: The true underlying y values of the problem to observe
    :returns: observations of `y`
    """
    return y
