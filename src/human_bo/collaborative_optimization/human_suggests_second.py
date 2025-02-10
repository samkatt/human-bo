"""Human *may* suggest a second query given a suggestion of the AI

In particular, here we consider the setup in which both human and AI are interested in optimizing a function `f`.
The human may know something about it, though the AI is (presumably) better at optimization.

Per usual, we consider the instance where the budget is the main cost, so we care for sample efficiency.
As a result, whether the human suggests a ...
"""

from typing import Any, Callable

import torch
from botorch.test_functions import synthetic

from human_bo import core, reporting, test_functions

CONFIG = {
    "user": {
        "type": str,
        "shorthand": "u",
        "help": "The (real) user behavior.",
        "tags": {"experiment-parameter"},
        "parser-arguments": {"choices": {"random", "bo", "noop"}},
    }
}


# TODO: add types when converged.
def ai_then_human_optimization_experiment(
    ai,
    human,
    f: synthetic.SyntheticTestFunction,
    report_step: reporting.StepReport,
    seed: int,
    budget: int,
) -> dict[str, list]:
    """Main loop for AI suggestion then Human pick joint optimization

    Pretty straightforward interactive experiment setup:
        1. Ask action from `ai`
        2. Ask action from `human` _given AI action_
        3. Apply both actions to `problem`

    The actual implementation depends heavily on how `ai`, `human`, and `problem` are implemented!
    """

    torch.manual_seed(seed)

    history: list[dict[str, Any]] = []
    x = torch.Tensor()
    y = torch.Tensor()

    # For keeping statistics.
    stats = []

    y_optimal = f.optimal_value
    y_max = -torch.inf

    # Actual interaction and experiment loop.
    step = 0
    while step < budget:
        ai_action, ai_stats = ai(x, y, history)
        x_new, human_stats = human(x, y, ai_action, history)

        y_new = f(x_new)

        x = torch.cat((x, x_new))
        y = torch.cat((y, y_new))

        history.append(
            {
                "ai_action": ai_action,
                "x": x_new,
                "y": y_new,
            }
        )

        # Statistics for online reporting.
        y_true_new = f.evaluate_true(x)

        y_max = max(y_max, y_true_new.max().item())
        regret = y_optimal - y_max

        step_data = {
            "ai": ai_stats,
            "human": human_stats,
            "regret": regret,
            "y_max": y_max,
            "y_true": y_true_new,
        }

        stats.append(step_data)
        report_step(step_data, step)

        step += len(y_new)

    return {"history": history, "stats": stats}


class PlainJointAI:
    """Bayesian optimization agent in the joint optimization setting."""

    def __init__(
        self,
        plain_ai: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        init_x,
        init_y,
    ):
        self.init_x = init_x
        self.init_y = init_y
        self.bo = plain_ai

    def pick_queries(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Returns next query given data `x` and `y`.

        In practice will combine initial data points `self.init_x` and `self.init_y`
        with `x` and `y`.
        """

        all_x = torch.cat((self.init_x, x))
        all_y = torch.cat((self.init_y, y))

        return self.bo(all_x, all_y), {
            "stats": "Joint AI agent has no stats implemented yet."
        }


def create_user(exp_conf: dict[str, Any], f):
    """Creates a user model for human-then-AI experiment given experiment configurations."""
    match exp_conf["user"]:
        case "random":
            return lambda _x, _y, ai_action, _hist: (
                torch.cat((ai_action, test_functions.random_queries(f._bounds))),
                {"stats": "Random user has no stats yet."},
            )

        case "bo":
            bo_human = core.PlainBO(exp_conf["kernel"], exp_conf["acqf"], f._bounds)
            x_init, y_init = test_functions.sample_initial_points(
                f, f._bounds, exp_conf["n_init"]
            )
            human = PlainJointAI(bo_human.pick_queries, x_init, y_init)

            def human_response(x, y, ai_action, _history):
                x, stats = human.pick_queries(x, y)
                return torch.cat((ai_action, x)), stats

            return human_response

        case "noop":
            return lambda _x, _y, ai_action, _hist: (
                ai_action,
                {"stats": "Noop user has not stats."},
            )

    raise KeyError(f"{exp_conf['user']} is not a valid user option.")
