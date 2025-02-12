"""Testing and example code for multi-objective optimization."""

from typing import Any, Callable

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

from human_bo import core, interaction_loops, reporting

type UtilityFunction = Callable[[torch.Tensor], torch.Tensor]


def moo_learn_subjective_function():
    """Experiment on multi-objective optimization where the only unknown is one (unmeasurable) objective function."""
    # Parameters.
    noise_std = [0.2, 0.4]
    test_function = moo_test_functions.BraninCurrin
    budget = 20
    unknown_output_idx = 0

    # n_init = 6
    # num_restarts_acqf = 8
    # raw_samples_acqf = 128

    # Setup problem.
    moo_function = test_function(noise_std)

    preference_weights = core.random_queries(
        [(0, 1) for _ in range(moo_function.dim)]
    ).squeeze()
    preference_weights = preference_weights / preference_weights.sum()

    utility_function = objective.LinearMCObjective(preference_weights)

    # Setup interaction components.
    agent = BayesOptGivenOneUnknownObjective(moo_function, unknown_output_idx)
    problem = MOOProblem(moo_function, utility_function)
    evaluation = MOOEvaluation(moo_function, utility_function, reporting.print_dot)

    res = interaction_loops.basic_interleaving(agent, problem, evaluation, budget)

    print(res)


class BayesOptGivenOneUnknownObjective(interaction_loops.Agent):
    """Bayes opt of some utility given 1 unknown objective."""

    def __init__(self, moo_function, unknown_output_idx: int):
        # TODO: set up surrogate model over output `unknown_output_idx` and Bayes optimization.
        self.moo_function = moo_function
        self.unknown_output_idx = unknown_output_idx

    def pick_query(self) -> tuple[Any, dict[str, Any]]:
        # TODO: implement.
        raise NotImplementedError()

    def observe(self, query, feedback, evaluation) -> None:
        del query, feedback, evaluation
        # TODO: implement.
        raise NotImplementedError()


class MOOProblem(interaction_loops.Problem):
    """Typical MOO problem."""

    def __init__(self, moo_function, utility_function: UtilityFunction):
        self.moo_function = moo_function
        self.utility_functiion = utility_function

    def give_feedback(self, query) -> tuple[Any, dict[str, Any]]:
        objectives = self.moo_function(query)
        feedback = self.utility_functiion(objectives)

        return feedback, {"objectives": objectives}

    def observe(self, query, feedback, evaluation) -> None:
        del query, feedback, evaluation


class MOOEvaluation(interaction_loops.Evaluation):
    """Evaluates the true value of query and tracks regret and max y."""

    def __init__(self, moo_function, utility_function: UtilityFunction, report_step):
        self.moo_function = moo_function
        self.utility_function = utility_function
        self.u_max = -torch.inf
        self.report_step = report_step
        self.step = 0

    def __call__(self, query, feedback) -> tuple[Any, dict[str, Any]]:
        del feedback

        objectives_true = self.moo_function(query, noise=False)
        utility_true = self.utility_function(objectives_true)

        self.u_max = max(self.u_max, utility_true.max().item())
        # TODO: implement regret tracking.

        evaluation = {
            "utility_true": utility_true,
            "y_max": self.u_max,
            "objectives_true": objectives_true,
        }

        self.report_step(evaluation, self.step)
        self.step += 1

        return None, evaluation


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
    train_x = core.random_queries(problem.bounds, n_init)
    train_y = problem(train_x)

    true_y = problem(train_x, noise=False)

    x_dim = train_x.shape[1]
    y_dim = train_y.shape[1]

    preference_weights = core.random_queries([(0, 1) for _ in range(y_dim)]).squeeze()
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
    train_x = core.random_queries(problem.bounds, n_init)
    train_y = problem(train_x)

    true_y = -problem.evaluate_true(train_x)

    x_dim = train_x.shape[1]
    y_dim = train_y.shape[1]

    preference_weights = core.random_queries([(0, 1) for _ in range(y_dim)]).squeeze()
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
    moo_learn_subjective_function()
