from typing import Optional, Union

import torch
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.optim.optimize import optimize_acqf
from botorch.posteriors.posterior import Posterior
from botorch.posteriors.torch import TorchPosterior
from botorch.sampling.base import MCSampler
from botorch.sampling.get_sampler import GetSampler
from botorch.sampling.stochastic_samplers import ForkedRNGSampler
from torch import Tensor, distributions


@GetSampler.register(distributions.StudentT)
def _get_sampler_torch(
    posterior: TorchPosterior,
    sample_shape: torch.Size,
    *,
    seed: int | None = None,
) -> MCSampler:
    # Use `ForkedRNGSampler` to ensure determinism in acquisition function evaluations.
    return ForkedRNGSampler(sample_shape=sample_shape, seed=seed)


class BayesianRegressionModel(Model):
    _num_outputs: int
    df: int
    s_squared: Tensor
    beta: Tensor
    L: Tensor

    def __init__(self) -> None:
        super(BayesianRegressionModel, self).__init__()

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.beta

    def fit(self, x: Tensor, y: Tensor) -> None:
        self._num_outputs = y.shape[-1]
        n, p = x.shape
        self.df = n - p
        # Rather than V = torch.linalg.inv(x.T @ x) as in BDA
        # instead use L = torch.linalg.cholesky(x.T @ x) for stability.
        # To use L, we can simply replace operations like:
        # x = V @ b
        # with a call to `torch.cholesky_solve`:
        # x = torch.cholesky_solve(b, L)
        self.L = torch.linalg.cholesky(x.T @ x)
        # Least squares estimate
        # self.beta = torch.cholesky_solve(x.T, self.L) @ y
        self.beta = torch.cholesky_solve(x.T, self.L) @ y
        # Model's residuals from the labels.
        r: Tensor = y - self(x)
        # Sample variance
        self.s_squared = (1 / self.df) * r.T @ r

        breakpoint()

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
    ) -> Posterior:
        # Squeeze out the q dimension if needed.
        breakpoint()
        n, q, _ = X.shape
        if output_indices:
            X = X[..., output_indices]
        loc = self(X)
        # Full covariance matrix of all test points.
        cov = self.s_squared * (
            torch.eye(n, n) + X.squeeze() @ torch.cholesky_solve(X.squeeze().T, self.L)
        )
        # The batch semantics of BoTorch evaluate each data point in their own batch.
        # So, we extract the diagonal representing Var[\tilde y_i | y_i] of each test point.
        scale = torch.diag(cov).reshape(n, q, self.num_outputs)
        # Form the posterior predictive dist according to Sec 14.2, Pg 357 of BDA.
        posterior_predictive_dist = distributions.StudentT(
            df=self.df, loc=loc, scale=scale
        )
        posterior = TorchPosterior(distribution=posterior_predictive_dist)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior


# Set the seed for reproducibility
torch.manual_seed(1)
# Double precision is highly recommended for BoTorch.
# See https://github.com/pytorch/botorch/discussions/1444
torch.set_default_dtype(torch.float64)

train_X = torch.rand(20, 2) * 2
Y = 1 - (train_X - 0.5).norm(dim=-1, keepdim=True)
Y += 0.1 * torch.rand_like(Y)
bounds = torch.stack([torch.zeros(2), 2 * torch.ones(2)])

train_X.shape  # torch.Size([20, 2])

bayesian_regression_model = BayesianRegressionModel()
bayesian_regression_model.fit(train_X, Y)

Y.shape
train_X.shape

# from human_bo.moo import moo_models

# my_model = moo_models.UnknownCompositeModel(lambda X: 1 - (X - 0.5).norm(dim=-1, keepdim=True), train_X, Y)

blah = optimize_acqf(
    qUpperConfidenceBound(bayesian_regression_model, beta=0.4),
    bounds=bounds,
    q=1,
    num_restarts=5,
    raw_samples=10,
)

1 + 1
