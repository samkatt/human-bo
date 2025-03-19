"""Models (botorch) for inference in MOO."""

from collections.abc import Callable

import torch
from botorch.acquisition import objective
from botorch.models import ensemble, model
from botorch.posteriors import posterior as botorch_posterior
from botorch.posteriors import torch as torch_posterior
from botorch.sampling.base import MCSampler
from botorch.sampling.get_sampler import GetSampler
from botorch.sampling.stochastic_samplers import ForkedRNGSampler
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

    def __init__(self, f, x, y, n_samples: int = 100):
        super().__init__()
        assert y.dim() == self._num_outputs

        # In this setting, x -> o -> y.
        # We observe `x` and know `f: x -> o`.
        self.f = f

        # We construct what the observed objectives `o` would have been without noise:
        o = self.f(x, noise=False)

        # Our main component is a distribution over weights.
        # We first initialize some prior by uniformly sampling from the weight space.
        # (There are many ways of doing this, this way is really lazy and relatively poor.)
        # TODO: improve prior over weights in `UnknownCompositeModel`.
        self.n_samples = 100
        utility_weights = torch.rand([n_samples, o.shape[-1]])
        utility_weights = utility_weights / utility_weights.sum(dim=1, keepdim=True)

        # We then calculate the *log* likelihood for each weight, because likelihood goes to 0 fast.
        # The log likelihood of `Normal(y | mu, sigma)` is `- ln(sigma) - 1/2 ln(2pi) - 1/2 ((y - mu) / sigma)^2`.

        # We assume some variance `sigma`, but there should be a better way:
        #   - Learn this from the data.
        #   - Infer this from known variances in `self.f`.
        # FIX: computation of variance.
        self.sigma = 5
        sigma_2 = self.sigma**2

        # In our case, each `(o, w)` combination creates its own normal distribution with mean `mu = o @ w`.
        # For multiple `(o,y)` combinations, the log likelihood is then a sum:
        # `log_likelihood(w, os, ys) = -n ln(sigma) - n/2 ln(2pi) - 1 / 2sigma^2 sum[ (y_i - o_i @ w)^2 ]`

        part_1 = n_samples * torch.log(torch.tensor(self.sigma))
        part_2 = n_samples * torch.log(torch.tensor(2 * torch.pi)) / 2
        # TODO: Make this a tensor operation.
        llhoods = torch.stack(
            [
                -part_1 - part_2 - sum((y - o @ w) ** 2) / sigma_2
                for w in utility_weights
            ]
        )

        # We now put them back in regular probability space by taking their exponential.
        # We also want to make sure they sum to one, to have a true distribution.

        # We use a mathematical trick - subtract the max - as this maintains their relative values.
        # But make the exponential numerically more stable:
        lhoods = (llhoods - llhoods.max()).exp()

        self.utility_probabilities = distributions.Categorical(lhoods / lhoods.sum())
        self.utility_weights = utility_weights

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        breakpoint()  # TODO: figure out where it is used implement `UnknownCompositeModel.forward`
        return torch.Tensor()
        # return self.f(x) @ self.weights

    def posterior(
        self,
        X: torch.Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | torch.Tensor = False,
        posterior_transform: objective.PosteriorTransform | None = None,
    ) -> botorch_posterior.Posterior:
        """This function is necessary to implement `model.Model`.

        We must, given some input `X` of shape return a distribution over `y`.
        This posterior must implement `rsample(self, sample_shape) -> Tensor`.

        X: A `b x q x d`-dim Tensor, where `d` is the dimension of the
            feature space, `q` is the number of points considered jointly,
            and `b` is the batch dimension.

        Other arguments I have little understanding of, so I copied them from
        existing implementations.
        """
        del observation_noise  # no need for this.

        # I copied this from existing posterior implementations.
        # No idea exactly what it does, but I hope it is sufficient.
        if output_indices:
            print(
                "WARN: applying `output_indices` in `UnknownCompositeModel`, and I am not sure what it does"
            )
            X = X[..., output_indices]

        # TODO: investigate possibility of representing as Normal distribution.

        # Here we implement the `rsample` function and use `BotorchPosteriorFromFunction`
        # to implement the interface required by Botorch.

        # The idea is that we simply sample as follows:
        # To get a sample, we:
        #   1. Sample objectives from X `o ~ f(x)`.
        #   2. Sample weights from weights distribution `w ~ p(w)`.
        #   3. Return utility `u = o @ w`.

        n, q, x_dim = X.shape

        # XXX: This below is an attempt at using my own posterior (see `BotorchEnsembleFromFunction`).
        def sample(sample_shape: torch.Size) -> torch.Tensor:

            o = self.f(
                X.expand(*sample_shape, *X.shape)
            )  # (n_samples, n_x, n_q, o_dim)

            w = self.utility_weights[
                self.utility_probabilities.sample([n])
            ]  # (n_x, w_dim)
            w = w.expand([*sample_shape, *w.shape]).unsqueeze(
                -1
            )

            # TODO: Make this a tensor operation.

            y = o @ w

            breakpoint()
            return y

        posterior = BotorchPosteriorFromFunction(sample)

        # I copied this from existing posterior implementations.
        # No idea exactly what it does, but I hope it is sufficient.
        if posterior_transform is not None:
            print(
                "WARN: applying `output_indices` in `UnknownCompositeModel`, and I am not sure what it does"
            )
            posterior = posterior_transform(posterior)

        return posterior


# This below is an attempt at creating posterior using ensemble methods.
class BotorchEnsembleFromFunction(ensemble.EnsembleModel):
    """An ensemble implementation for creating posteriors in Botorch."""

    _num_outputs: int = 1
    num_samples: int

    def __init__(self, f, x, y, n_samples: int = 100):
        super().__init__()
        assert y.dim() == self._num_outputs

        # In this setting, x -> o -> y.
        # We observe `x` and know `f: x -> o`.
        self.f = f

        # We construct what the observed objectives `o` would have been without noise:
        o = self.f(x, noise=False)

        # Our main component is a distribution over weights.
        # We first initialize some prior by uniformly sampling from the weight space.
        # (There are many ways of doing this, this way is really lazy and relatively poor.)
        # TODO: improve prior over weights in `BotorchEnsembleFromFunction`.
        self.num_samples = 100
        utility_weights = torch.rand([self.num_samples, o.shape[-1]])
        utility_weights = utility_weights / utility_weights.sum(dim=1, keepdim=True)

        # We then calculate the *log* likelihood for each weight, because likelihood goes to 0 fast.
        # The log likelihood of `Normal(y | mu, sigma)` is `- ln(sigma) - 1/2 ln(2pi) - 1/2 ((y - mu) / sigma)^2`.

        # We assume some variance `sigma`, but there should be a better way:
        #   - Learn this from the data.
        #   - Infer this from known variances in `self.f`.
        # FIX: computation of variance.
        self.sigma = 5
        sigma_2 = self.sigma**2

        # In our case, each `(o, w)` combination creates its own normal distribution with mean `mu = o @ w`.
        # For multiple `(o,y)` combinations, the log likelihood is then a sum:
        # `log_likelihood(w, os, ys) = -n ln(sigma) - n/2 ln(2pi) - 1 / 2sigma^2 sum[ (y_i - o_i @ w)^2 ]`

        part_1 = self.num_samples * torch.log(torch.tensor(self.sigma))
        part_2 = self.num_samples * torch.log(torch.tensor(2 * torch.pi)) / 2
        # TODO: Make this a tensor operation.
        llhoods = torch.stack(
            [
                -part_1 - part_2 - sum((y - o @ w) ** 2) / sigma_2
                for w in utility_weights
            ]
        )

        # We now put them back in regular probability space by taking their exponential.
        # We also want to make sure they sum to one, to have a true distribution.

        # We use a mathematical trick - subtract the max - as this maintains their relative values.
        # But make the exponential numerically more stable:
        lhoods = (llhoods - llhoods.max()).exp()

        # We now save these - we are done fitting - and get ready to have `self.forward` called.
        self.utility_probabilities = distributions.Categorical(lhoods / lhoods.sum())
        self.utility_weights = utility_weights

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """`((b), num_samples, q, 1)`, with `s`"""

        # We are going to generate `self.num_samples` outputs for each `X`.
        # Each sample goes as follows:
        #   1. Sample o ~ f(x)
        #   2. Sample w ~ p(w) (from fitting)
        #   3. u ~ o * w

        b, q, _ = X.shape

        breakpoint()

        # O.shape = `(s, b, q, m)`
        O = self.f(X.expand(self.num_samples, *X.shape))
        o_dim = O.shape[-1]

        assert O.shape == torch.Size([self.num_samples, b, q, o_dim])

        # w.shape = (`s, m`)
        W = self.utility_weights[self.utility_probabilities.sample([self.num_samples])]
        # XXX
        W = W.expand([self.num_samples, *W.shape])

        assert W.shape == torch.Size([self.num_samples, o_dim])
        # (n_s, n_x, w_dim, 1)
        samples = O @ W  # shape is `(s, b, q, 1)`

        assert sample.shape == torch.Size([self.num_samples, b, q, 1])

        return samples.transpose(0, 1)


# This below is an attempt to create my own posterior.
class BotorchPosteriorFromFunction(botorch_posterior.Posterior):
    """A simple implementation of the `botorch_posterior.Posterior`

    Apparently only needs to implement `rsample`.
    I do not quite like this whole class-oriented implementation,
    so I simply assume that someone can give me the implementation of `rsample`
    and then I'll just bind it to its name here.
    """

    def __init__(
        self, f: Callable[[torch.Size], torch.Tensor], device=None, dtype=None
    ):
        """Assumes `f` follows the guidelines of `rsample` below."""
        super().__init__()
        self.f = f

        self._device = device if device else torch.get_default_device()
        self._dtype = dtype if dtype else torch.get_default_dtype()

    @property
    def device(self) -> torch.device:
        """Required interface."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Required interface."""
        return self._dtype

    def rsample(
        self,
        sample_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """Here we just defer to our function `f`"""
        if sample_shape is None:
            sample_shape = torch.Size()

        return self.f(sample_shape)


@GetSampler.register(BotorchPosteriorFromFunction)
def _get_sampler_torch(
    posterior: torch_posterior.TorchPosterior,
    sample_shape: torch.Size,
    *,
    seed: int | None = None,
) -> MCSampler:
    del posterior  # Not used further here, used to shut up linters.

    # Use `ForkedRNGSampler` to ensure determinism in acquisition function evaluations.
    return ForkedRNGSampler(sample_shape=sample_shape, seed=seed)
