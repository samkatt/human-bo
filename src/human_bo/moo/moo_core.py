"""Some core utilities for multi-objective optimization."""

import torch
from botorch.acquisition import objective
from botorch.models import model as botorch_model
from botorch.posteriors import posterior
from botorch.posteriors import torch as torch_posterior
from torch import distributions


class ObjectiveFunctionModel(botorch_model.Model):
    """Transforms a deterministic function into"""

    _num_outputs: int

    def __init__(self, f, noise_std):
        super().__init__()
        self.f = f
        self.noise_std = torch.Tensor(noise_std)
        self._num_outputs = len(noise_std)

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    def forward(self, x: torch.Tensor) -> distributions.Distribution:
        return distributions.MultivariateNormal(self.f(x), torch.diag(self.noise_std))

    def posterior(
        self,
        X: torch.Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | torch.Tensor = False,
        posterior_transform: objective.PosteriorTransform | None = None,
    ) -> posterior.Posterior:
        del observation_noise

        if output_indices:
            X = X[..., output_indices]

        ret = torch_posterior.TorchPosterior(distribution=self(X))
        if posterior_transform is not None:
            ret = posterior_transform(ret)

        return ret
