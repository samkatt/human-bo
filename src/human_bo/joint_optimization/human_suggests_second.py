"""Human *may* suggest a second query given a suggestion of the AI

In particular, here we consider the setup in which both human and AI are interested in optimizing a function `f`.
The human may know something about it, though the AI is (presumably) better at optimization.

Per usual, we consider the instance where the budget is the main cost, so we care for sample efficiency.
As a result, whether the human suggests a ...
"""

from typing import Any, Callable

import torch
from botorch.test_functions import synthetic

from human_bo import conf, core, reporting, test_functions


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
    stats = []

    optimal_y = f.optimal_value
    y_max = -torch.inf

    for step in range(budget):

        x_ai, ai_stats = ai(history)
        # TODO: we may want to give human all control over x (to overwrite, for example).
        x_human, human_stats = human(history, x_ai)

        # TODO: merge these calls. (or just let human give all)
        y_ai = f(x_ai)
        y_human = f(x_human)

        history.append(
            {
                "x_ai": x_ai,
                "x_human": x_human,
                "y_ai": y_ai,
                "y_human": y_human,
            }
        )

        # Statistics for online reporting.
        # TODO: keep track of true `y` to get the true `regret`.
        y_max = max(y_max, torch.cat((y_ai, y_human)).max().item())
        regret = optimal_y - y_max

        step_data = {
            "ai": ai_stats,
            "human": human_stats,
            "regret": regret,
            "y_max": y_max,
        }

        stats.append(step_data)
        report_step(step_data, step)

    return {"history": history, "stats": stats}


def update_config():
    """This function updates the configurations to set up for human suggests second experiments.

    This needs to be run at the start of any script on these type of experiments.
    """
    # Add `user_model` as a configuration.
    conf.CONFIG["user"] = {
        "type": str,
        "shorthand": "u",
        "help": "The (real) user behavior.",
        "tags": {"experiment-parameter"},
        "parser-arguments": {"choices": {"random", "bo"}},
    }


class PlainJointAI:
    def __init__(
        self,
        plain_ai: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        init_x,
        init_y,
    ):
        self.init_x = init_x
        self.init_y = init_y
        self.bo = plain_ai

    def pick_queries(self, history: list) -> tuple[torch.Tensor, dict]:
        """Returns next query `x` given history

        In practice will combine initial data points `self.init_x` and `self.init_y`
        with those observed in `history` to do some Bayesian optimization and optimize some acquisition function.

        See `joint_optimization.human_suggests_second.ai_then_human_optimization_experiment`
        for expected API.
        """

        # Base case has no history. Perform BO on the initial points.
        if not history:
            return self.bo(self.init_x, self.init_y), {
                "stats": "Joint AI agent has no stats implemented yet."
            }

        # At this point, we know there is some history:
        # Concatenate the observed x's and y's to our initial samples.
        x = torch.cat(
            (
                self.init_x,
                torch.cat([x for t in history for x in [t["x_ai"], t["x_human"]]]),
            )
        )
        y = torch.cat(
            (
                self.init_y,
                torch.cat([y for t in history for y in [t["y_ai"], t["y_human"]]]),
            )
        )

        return self.bo(x, y), {"stats": "Joint AI agent has no stats implemented yet."}


def create_user(exp_conf: dict[str, Any], f):
    """Creates a user model for human-then-AI experiment given experiment configurations."""
    match exp_conf["user"]:
        case "random":
            return lambda hist, stats: (
                core.random_queries(f._bounds),
                "Random user has no stats yet.",
            )
        case "bo":
            bo_human = core.PlainBO(exp_conf["kernel"], exp_conf["acqf"], f._bounds)
            x_init, y_init = test_functions.sample_initial_points(
                f, f._bounds, exp_conf["n_init"]
            )
            human = PlainJointAI(bo_human.pick_queries, x_init, y_init)
            return lambda hist, stats: human.pick_queries(hist)

    raise KeyError(f"{exp_conf['user']} is not a valid user option.")
