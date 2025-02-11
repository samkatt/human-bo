#!/usr/bin/env python

"""Main entry point: runs collaborative BO."""

import argparse
from typing import Any

import torch

from human_bo import conf, core, interaction_loops, reporting, utils
from human_bo.collaborative_optimization import human_suggests_second
from human_bo.factories import pick_test_function
from human_bo.test_functions import sample_initial_points


def main():
    """Main entry human-then-AI experiments."""
    torch.set_default_dtype(torch.double)

    exp_conf = conf.CONFIG
    exp_conf.update(human_suggests_second.CONFIG)

    parser = argparse.ArgumentParser(description="Command description.")
    for arg, values in exp_conf.items():
        parser.add_argument(
            "-" + values["shorthand"],
            "--" + arg,
            help=values["help"],
            type=values["type"],
            **values["parser-arguments"],
        )

    parser.add_argument(
        "-f", "--save_path", help="Name of saving directory.", type=str, required=True
    )
    parser.add_argument("--wandb", help="Wandb configuration file.", type=str)

    args = parser.parse_args()
    exp_params = conf.from_ns(args)

    experiment_name = (
        "_".join(
            [
                str(v)
                for k, v in exp_params.items()
                if "experiment-parameter" in exp_conf[k]["tags"]
            ]
        )
        + "_"
        + str(exp_params["seed"])
    )
    path = args.save_path + "/" + experiment_name + ".pt"

    utils.exit_if_exists(path)
    utils.exit_if_exists(args.save_path, negate=True)

    res: dict[str, Any] = {"conf": exp_params}

    torch.manual_seed(exp_params["seed"])

    # Create problem.
    f = pick_test_function(exp_params["problem"], exp_params["problem_noise"])
    problem = Problem(f)

    report_step = (
        reporting.initiate_and_create_wandb_logger(args.wandb, exp_params)
        if args.wandb
        else reporting.print_dot
    )
    evaluation = Evaluation(f, report_step)

    # Create Agents
    x_init_ai, y_init_ai = sample_initial_points(f, f._bounds, exp_params["n_init"])
    agent = AI(
        f._bounds, exp_params["kernel"], exp_params["acqf"], x_init_ai, y_init_ai
    )
    res["ai_conf"] = {"x": x_init_ai, "y": y_init_ai}

    user, user_conf = human_suggests_second.create_user(
        exp_params["user_model"],
        f,
        exp_params["kernel"],
        exp_params["acqf"],
        exp_params["n_init"],
    )
    res["user_model_conf"] = user_conf

    # Run actual experiment.
    print(f"Running experiment for {path}")

    res.update(
        interaction_loops.ai_advices_human_loop(
            agent, user, problem, evaluation, exp_params["budget"]
        )
    )

    torch.save(res, path)


class AI(interaction_loops.Agent):
    """This is a simple Bayes optimization agent."""

    def __init__(self, bounds, kernel, acqf, x_init, y_init):
        self.bo = core.PlainBO(kernel, acqf, bounds)
        self.x, self.y = x_init, y_init

    def pick_query(self) -> tuple[Any, dict[str, Any]]:
        query = self.bo.pick_queries(self.x, self.y)
        return query, {}

    def observe(self, query, feedback, evaluation) -> None:
        del query, evaluation

        self.x = torch.cat((self.x, feedback["action"]))
        self.y = torch.cat((self.y, feedback["feedback"]))


class Problem(interaction_loops.Problem):
    """The problem in the AI-then-Human setting simply evaluates the query given."""

    def __init__(self, problem_function):
        self.f = problem_function

    def give_feedback(self, query) -> tuple[Any, dict[str, Any]]:
        feedback = self.f(query)
        return feedback, {}

    def observe(self, query, feedback, evaluation) -> None:
        del query, feedback, evaluation


class Evaluation(interaction_loops.Evaluation):
    """Evaluates the true value of query and tracks regret and max y."""

    def __init__(self, problem_function, report_step):
        self.problem_function = problem_function
        self.y_max = -torch.inf
        self.report_step = report_step

        # In this problem, the user is allowed to pick different number of queries.
        # To respect this, we keep track of number of queries.
        self.number_of_queries = 0

    def __call__(self, query, feedback) -> tuple[Any, dict[str, Any]]:
        del feedback

        # In this problem, the user is allowed to pick different number of queries.
        # To respect this, we keep track of number of queries.
        self.number_of_queries += query.shape[0]

        y_true = self.problem_function(query, noise=False)

        self.y_max = max(self.y_max, y_true.max().item())
        regret = self.problem_function.optimal_value - self.y_max

        evaluation = {"y_true": y_true, "y_max": self.y_max, "regret": regret}

        self.report_step(evaluation, self.number_of_queries)

        return None, evaluation


if __name__ == "__main__":
    main()
