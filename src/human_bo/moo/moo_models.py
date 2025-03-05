"""Models (botorch) for inference in MOO."""

import torch
from botorch.acquisition import objective
from botorch.models import model
from botorch.posteriors import posterior as botorch_posterior
from botorch.posteriors import torch as torch_posterior
from torch import distributions


# TODO: allow for diagnostics.
class UnknownCompositeModel(model.Model):
    """Assumes X -> Y factorizes into simple composite function `g` of unknown parameters.

    This model assumes that data `X <- Y` is a simple composite function of more complex functions `f`.
    In particular, `Y = g(f(x); w)` with `f` having multiple outputs.

    This model assumes `f` is known, but the parameters `w` are not (the functional form of `g`, given `w` is known.).
    """

    # Unsure if this is necessary, since we have the property, but just following conventions of botorch.
    _num_outputs = 1

    def __init__(self, f, x, y):
        super().__init__()
        assert y.dim() == 1

        self.f = f

        # Bayesian Linear Regression (see `https://botorch.org/docs/tutorials/custom_model`)
        n, p = x.shape

        assert n > p  # This inference will not work if `x` has more columns than rows.
        self.df = n - p

        self.L = torch.linalg.cholesky(x.T @ x)
        self.weights = torch.cholesky_solve(x.T, self.L) @ y

        r = y - self(x)
        self.s_squared = (1 / self.df) * r.T @ r  # originally `r.T @ r`, but `r` is guaranteed 1-D in this codebase.

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights

    def posterior(
        self,
        X: torch.Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | torch.Tensor = False,
        posterior_transform: objective.PosteriorTransform | None = None,
    ) -> botorch_posterior.Posterior:
        del observation_noise

        breakpoint()  # TODO: check `UnknownComposite.posterior`

        # Bayesian Linear Regression (see `https://botorch.org/docs/tutorials/custom_model`)
        n, q, _ = X.shape
        if output_indices:
            X = X[..., output_indices]

        loc = self(X)
        # Full covariance matrix of all test points.
        cov = self.s_squared * (
            torch.eye(n, n) + X.squeeze() @ torch.cholesky_solve(X.squeeze().T, self.L)
        )
        scale = torch.diag(cov).reshape(n, q, self.num_outputs)
        breakpoint()  # TODO: scale is float?
        posterior_predictive_dist = distributions.StudentT(
            df=self.df, loc=loc, scale=scale
        )
        posterior = torch_posterior.TorchPosterior(
            distribution=posterior_predictive_dist
        )
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)

        return posterior
