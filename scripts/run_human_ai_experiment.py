#!/usr/bin/env python

"""Main entry point: runs typical BO with (potentially) human giving the feedback."""

import argparse
from typing import Any

import torch

from human_bo import (
    conf,
    core,
    factories,
    human_feedback_experiments,
    interaction_loops,
    reporting,
    utils,
)


def main():
    """Main entry human-feedback experiments."""
    torch.set_default_dtype(torch.double)

    exp_conf = conf.CONFIG
    exp_conf.update(human_feedback_experiments.CONFIG)

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

    experiment_name = "_".join(
        conf.get_values_with_tag(exp_params, "experiment-parameter", exp_conf)
        + [str(exp_params["seed"])]
    )

    path = args.save_path + "/" + experiment_name + ".pt"

    utils.exit_if_exists(path)
    utils.create_directory_if_does_not_exist(args.save_path)

    torch.manual_seed(exp_params["seed"])

    # Create problem and evaluation.
    problem_function = factories.pick_test_function(
        exp_params["problem"], exp_params["problem_noise"]
    )
    report_step = (
        reporting.initiate_and_create_wandb_logger(args.wandb, exp_params, exp_conf)
        if args.wandb
        else reporting.print_dot
    )
    evaluation = Evaluation(problem_function, report_step)

    # Create Agents
    x_init, y_init = core.sample_initial_points(
        problem_function, problem_function._bounds, exp_params["n_init"]
    )

    ai = AI(
        problem_function._bounds,
        exp_params["kernel"],
        exp_params["acqf"],
        x_init,
        y_init,
    )
    human = Human(
        exp_params["user_model"],
        conf.CONFIG["problem"]["parser-arguments"]["choices"][exp_params["problem"]][
            "optimal_x"
        ],
        problem_function,
    )

    print(f"Running experiment for {path}")
    res = interaction_loops.basic_interleaving(
        ai, human, evaluation, exp_params["budget"]
    )
    res["conf"] = exp_params
    res["initial_points"] = {"x": x_init, "y": y_init}

    torch.save(res, path)


class AI(interaction_loops.Agent):
    """Simple Bayes optimization Agent"""

    def __init__(self, bounds, kernel, acqf, x_init, y_init):
        self.bo = core.PlainBO(kernel, acqf, bounds)
        self.x, self.y = x_init, y_init

    def pick_query(self) -> tuple[Any, dict[str, Any]]:
        query, val = self.bo.pick_queries(self.x, self.y)
        return query, {"acqf_value": val}

    def observe(self, query, feedback, evaluation) -> None:
        del evaluation

        self.x = torch.cat((self.x, query))
        self.y = torch.cat((self.y, feedback))


class Human(interaction_loops.Problem):
    """The 'problem' in BO, represented by (optional) user model."""

    def __init__(self, user_model, x_optimal, problem_function):
        self.problem_function = problem_function
        self.user = human_feedback_experiments.pick_user_model(
            user_model,
            x_optimal[torch.randint(0, len(x_optimal), size=(1,))],
            problem_function,
        )

    def give_feedback(self, query) -> tuple[Any, dict[str, Any]]:
        y_observed = self.problem_function(query)
        feedback = self.user(query, y_observed)

        return feedback, {"y_observed": y_observed}

    def observe(self, query, feedback, evaluation) -> None:
        del query, feedback, evaluation


class Evaluation(interaction_loops.Evaluation):
    def __init__(self, problem_function, report_step):
        self.problem_function = problem_function
        self.y_max = -torch.inf
        self.step = 0
        self.report_step = report_step

    def __call__(self, query, feedback) -> tuple[Any, dict[str, Any]]:
        del feedback

        y_true = self.problem_function(query, noise=False)

        self.y_max = max(self.y_max, y_true.max().item())
        regret = self.problem_function.optimal_value - self.y_max

        evaluation = {"y_true": y_true, "y_max": self.y_max, "regret": regret}

        self.report_step(evaluation, self.step)
        self.step += 1

        return None, evaluation


if __name__ == "__main__":
    main()
