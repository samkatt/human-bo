"""Testing and example code for multi-objective optimization."""

from typing import Callable

import torch
from botorch import fit_gpytorch_mll, posteriors
from botorch.acquisition import logei, objective
from botorch.models import gp_regression, model, model_list_gp_regression
from botorch.models.transforms import input, outcome
from botorch.optim.optimize import optimize_acqf
from botorch.posteriors import torch as torch_posteriors
from botorch.test_functions import multi_objective as moo_test_functions
from botorch.utils.multi_objective.box_decompositions import dominated
from gpytorch.mlls import ExactMarginalLogLikelihood, sum_marginal_log_likelihood
from torch.distributions import multivariate_normal

from human_bo import test_functions


def run_moo_example():
    """Basic, very simple multi-objective optimization.

    Inspired by https://botorch.org/docs/tutorials/multi_objective_bo/,
    this is mostly a tutorial for typical MOO: finding Pareto front.
    """
    noise_std = [0.2, 0.4]
    test_func = moo_test_functions.BraninCurrin
    n_init = 6
    budget = 20

    num_restarts_acqf = 8
    raw_samples_acqf = 128

    problem = test_func(noise_std, negate=True)
    train_x = test_functions.random_queries(problem.bounds, n_init)
    train_y = problem(train_x)

    true_y = problem(train_x, noise=False)

    x_dim = train_x.shape[1]
    y_dim = train_y.shape[1]

    preference_weights = test_functions.random_queries(
        [(0, 1) for _ in range(y_dim)]
    ).squeeze()
    preference_weights = preference_weights / preference_weights.sum()
    max_utility = torch.matmul(true_y, preference_weights).max().item()

    stats = [
        {
            "volume": dominated.DominatedPartitioning(problem.ref_point, Y=true_y)
            .compute_hypervolume()
            .item(),
            "max_utility": max_utility,
        }
    ]

    for _ in range(budget):

        # Train on current data.
        gps = model_list_gp_regression.ModelListGP(
            *[
                gp_regression.SingleTaskGP(
                    train_x,
                    train_y[:, i : i + 1],
                    outcome_transform=outcome.Standardize(m=1),
                    input_transform=input.Normalize(x_dim),
                )
                for i in range(y_dim)
            ]
        )

        mll = sum_marginal_log_likelihood.SumMarginalLogLikelihood(gps.likelihood, gps)
        fit_gpytorch_mll(mll)

        # Acquire next data.
        acq = logei.qLogNoisyExpectedImprovement(
            model=gps,
            X_baseline=train_x,
            objective=objective.LinearMCObjective(preference_weights),
        )

        queries, _ = optimize_acqf(
            acq,
            problem.bounds,
            q=1,
            num_restarts=num_restarts_acqf,
            raw_samples=raw_samples_acqf,
        )
        feedback = problem(queries)

        train_x = torch.cat((train_x, queries))
        train_y = torch.cat((train_y, feedback))

        # Reporting statistics.
        queries_true_value = problem(queries, noise=False)
        true_y = torch.cat((true_y, queries_true_value))
        max_utility = max(
            max_utility,
            torch.matmul(queries_true_value, preference_weights).max().item(),
        )

        stats.append(
            {
                "volume": dominated.DominatedPartitioning(problem.ref_point, Y=true_y)
                .compute_hypervolume()
                .item(),
                "max_utility": max_utility,
            }
        )
        print(".", end="", flush=True)

    print(stats)


def foo():
    noise_std = [0.2, 0.4]
    test_func = moo_test_functions.BraninCurrin
    n_init = 6
    budget = 20

    num_restarts_acqf = 8
    raw_samples_acqf = 128

    problem = test_func(noise_std, negate=True)
    train_x = test_functions.random_queries(problem.bounds, n_init)
    train_y = problem(train_x)

    true_y = -problem.evaluate_true(train_x)

    x_dim = train_x.shape[1]
    y_dim = train_y.shape[1]

    preference_weights = test_functions.random_queries(
        [(0, 1) for _ in range(y_dim)]
    ).squeeze()
    preference_weights = preference_weights / preference_weights.sum()
    max_utility = torch.matmul(true_y, preference_weights).max().item()

    stats = [
        {
            "volume": dominated.DominatedPartitioning(problem.ref_point, Y=true_y)
            .compute_hypervolume()
            .item(),
            "max_utility": max_utility,
        }
    ]

    for _ in range(budget):

        # Train on current data.
        list_of_models: list[model.Model] = [
            BotorchModelFromCallable(
                lambda x: problem(x)[..., :, i], noise_std=noise_std[i]
            )
            for i in range(y_dim - 1)
        ]

        user_objective_gp = gp_regression.SingleTaskGP(
            train_x,
            # TODO: Perhaps use the `...` syntax here?
            train_y[:, -2:-1],
            outcome_transform=outcome.Standardize(m=1),
            input_transform=input.Normalize(x_dim),
        )

        mll = ExactMarginalLogLikelihood(
            user_objective_gp.likelihood, user_objective_gp
        )
        fit_gpytorch_mll(mll)
        models = model.ModelList(*list_of_models, user_objective_gp)

        # Acquire next data.
        acq = logei.qLogNoisyExpectedImprovement(
            model=models,
            X_baseline=train_x,
            objective=objective.LinearMCObjective(preference_weights),
        )

        queries, _ = optimize_acqf(
            acq,
            problem.bounds,
            q=1,
            num_restarts=num_restarts_acqf,
            raw_samples=raw_samples_acqf,
        )
        feedback = problem(queries)

        train_x = torch.cat((train_x, queries))
        train_y = torch.cat((train_y, feedback))

        # Reporting statistics.
        queries_true_value = -problem.evaluate_true(queries)
        true_y = torch.cat((true_y, queries_true_value))
        max_utility = max(
            max_utility,
            torch.matmul(queries_true_value, preference_weights).max().item(),
        )

        stats.append(
            {
                "volume": dominated.DominatedPartitioning(problem.ref_point, Y=true_y)
                .compute_hypervolume()
                .item(),
                "max_utility": max_utility,
            }
        )
        print(".", end="", flush=True)

    print(stats)


class BotorchModelFromCallable(model.Model):

    def __init__(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        noise_std: float,
    ):
        self.f = f
        self.noise_std = noise_std

    def __call__(self, X):
        return self.f(X)

    def posterior(
        self,
        X: torch.Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | torch.Tensor = False,
        posterior_transform: objective.PosteriorTransform | None = None,
    ) -> posteriors.Posterior:
        del output_indices, observation_noise
        Y = self(X)
        posterior = torch_posteriors.TorchPosterior(
            multivariate_normal.MultivariateNormal(Y, self.noise_std)
        )

        if posterior_transform is not None:
            return posterior_transform(posterior)

        return posterior


if __name__ == "__main__":
    foo()
