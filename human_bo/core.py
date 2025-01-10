"""Main functions for running experiments"""

from typing import Any

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from human_bo import factories, test_functions
from human_bo.conf import CONFIG


def human_feedback_experiment(
    problem: str,
    user_model: str,
    kernel: str,
    acqf: str,
    n_init: int,
    seed: int,
    budget: int,
    problem_noise: float,
) -> dict[str, Any]:
    """Main loop that handles all the work."""

    torch.manual_seed(seed)

    # Construct actual pieces from settings.
    problem_function = factories.pick_test_function(problem)
    bounds = torch.tensor(problem_function._bounds).T
    dim = bounds.shape[1]
    optimal_x = CONFIG["problem"]["parser-arguments"]["choices"][problem]["optimal_x"]

    user = factories.pick_user_model(
        user_model,
        optimal_x[torch.randint(0, len(optimal_x), size=(1,))],
        problem_function,
    )

    ai = PlainBO(kernel, acqf, problem_function._bounds)

    # TODO: ignoring `user`.
    print("WARNING: initial samples not generated with user model.")
    train_x, train_y, true_y = test_functions.sample_initial_points(
        problem_function, problem_function._bounds, n_init, problem_noise
    )

    # Main loop
    for _ in range(budget):
        print(".", end="", flush=True)

        candidates = ai.pick_queries(train_x, train_y)

        new_true_y = problem_function(candidates) + torch.normal(
            0, problem_noise, size=[1]
        )
        new_train_y = user(candidates, new_true_y)

        train_x = torch.cat((train_x, candidates))
        train_y = torch.cat((train_y, new_train_y.view(-1, 1)))
        true_y = torch.cat((true_y, new_true_y.view(-1, 1)))

    print("")

    return {
        "train_x": train_x,
        "train_y": train_y,
        "true_y": true_y,
    }


# TODO: add types when converged.
def ai_then_human_optimization_experiment(
    ai,
    human,
    problem,
    seed: int,
    budget: int,
):
    """Main loop for AI suggestion then Human pick joint optimization

    Pretty straightforward interactive experiment setup:
        1. Ask action from `ai`
        2. Ask action from `human` _given AI action_
        3. Apply both actions to `problem`

    The actual implementation depends heavily on how `ai`, `human`, and `problem` are implemented!
    """

    torch.manual_seed(seed)
    history = []
    stats = []

    for _ in range(budget):
        print(".", end="", flush=True)

        ai_action, ai_stats = ai(history)
        human_action, human_stats = human(history, ai_action)
        outcome, outcome_stats = problem(ai_action, human_action)

        stats.append({"ai": ai_stats, "human": human_stats, "outcome": outcome_stats})
        history.append(
            {"ai_action": ai_action, "human_action": human_action, "outcome": outcome}
        )

    print("")
    return {"history": history, "stats": stats}


def random_query(bounds: list[tuple[float, float]]) -> torch.Tensor:
    """Create a random tensor with values within `bounds`"""
    random_0_1 = torch.rand(size=[len(bounds)])
    return torch.tensor([(x * (u - l)) + l for x, [l, u] in zip(random_0_1, bounds)])


class PlainBO:
    def __init__(
        self,
        kernel: str,
        acqf: str,
        bounds: list[tuple[float, float]],
        num_restarts: int = 10,
        raw_samples: int = 512,
    ):
        """Creates a typical BO agent."""
        self.kernel = kernel
        self.acqf = acqf

        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

        self.bounds = torch.tensor(bounds).T
        self.dim = self.bounds.shape[1]

    def pick_queries(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Performs BO given input `x` and `y` data points."""

        gpr = SingleTaskGP(
            x,
            y,
            covar_module=factories.pick_kernel(self.kernel, self.dim),
            input_transform=Normalize(d=self.dim),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gpr.likelihood, gpr)
        fit_gpytorch_mll(mll)

        candidates, _ = optimize_acqf(
            acq_function=factories.pick_acqf(
                self.acqf, Standardize(m=1)(y)[0], gpr, self.bounds
            ),
            bounds=self.bounds,
            q=1,  # batch size, i.e. we only query one point
            num_restarts=10,
            raw_samples=512,
        )

        return candidates
