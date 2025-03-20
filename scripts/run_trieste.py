#!/usr/bin/env python

"""Main entry point: runs typical BO with (potentially) human giving the feedback."""

import argparse
from typing import Any

import numpy as np
import tensorflow as tf
import trieste

from human_bo import (
    conf,
    core,
    human_feedback_experiments,
    interaction_loops,
    reporting,
    test_functions,
    utils,
)


def main():
    """Main entry human-feedback experiments."""
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
        "-f", "--save_dir", help="Name of saving directory.", type=str, required=True
    )
    parser.add_argument("--wandb", help="Wandb configuration file.", type=str)
    exp_params = vars(parser.parse_args())

    experiment_name = "_".join(
        conf.get_values_with_tag(exp_params, "experiment-parameter", exp_conf)
        + [str(exp_params["seed"])]
    )

    path = exp_params["save_dir"] + "/" + experiment_name + ".pt"

    utils.exit_if_exists(path)
    utils.create_directory_if_does_not_exist(exp_params["save_dir"])

    tf.random.set_seed(exp_params["seed"])
    np.random.seed(exp_params["seed"])

    # Create problem and evaluation.
    # TODO: consider different problems.
    trieste_problem = trieste.objectives.single_objectives.Branin
    report_step = (
        reporting.initiate_and_create_wandb_logger(
            exp_params["wandb"], exp_params, exp_conf
        )
        if exp_params["wandb"]
        else reporting.print_dot
    )
    evaluation = Evaluation(trieste_problem, report_step)

    # Create Agents
    x_init = trieste_problem.search_space.sample(exp_params["n_init"])
    data_init = trieste_problem.objective(x_init)

    ai = TriesteBO(data_init, trieste_problem.search_space)
    # TODO: consider noise.
    problem = Problem(trieste_problem)

    print(f"Running experiment for {path}")
    res = interaction_loops.basic_loop(ai, problem, evaluation, exp_params["budget"])
    res["conf"] = exp_params
    res["conf"]["experiment_type"] = "human-feedback"
    res["initial_points"] = {"x": x_init, "y": y_init}

    torch.save(res, path)

    print(f"Done experiments, saved results in {path}")


class Problem(interaction_loops.Problem):
    """The 'problem' in BO, represented by (optional) user model."""

    def __init__(
        self, trieste_problem: trieste.objectives.single_objectives.ObjectiveTestProblem
    ):
        self.problem = trieste_problem
        self.observer = trieste.objectives.utils.mk_observer(self.problem.objective)

    def give_feedback(self, query) -> tuple[Any, dict[str, Any]]:
        breakpoint()  # TODO: infer what is query and implement `Problem.give_feedback`.
        feedback = self.observer(query)

        # TODO: record true y

        return feedback, {}
        # return feedback, {"y_observed": y_observed, "y_true": y_true}

    def observe(self, query, feedback, evaluation) -> None:
        del query, feedback, evaluation


class Evaluation(interaction_loops.Evaluation):
    def __init__(self, problem, report_step: reporting.StepReport):
        self.problem = problem
        self.y_max = tf.constant("inf")
        self.step = 0
        self.report_step = report_step

    def __call__(
        self,
        query,
        feedback,
        query_stats: dict[str, Any],
        feedback_stats: dict[str, Any],
        **kwargs,
    ) -> tuple[Any, dict[str, Any]]:
        del feedback, feedback_stats, kwargs

        breakpoint()  # TODO: store true and max value.

        y_true = 0
        # y_true = self.observer(query, noise=False)
        # self.y_max = tf.maximum(self.y_max, y_true)

        evaluation = {
            "y_true": y_true,
            "y_max": self.y_max,
        }

        if "map_arg_max" in query_stats:
            evaluation["regret"] = self.problem(query_stats["map_arg_max"], noise=False)

        self.step += 1
        self.report_step(evaluation, self.step)

        return None, evaluation


class TriesteBO(interaction_loops.Agent):

    def __init__(self, data, search_space):
        # TODO: account for different models.
        self.data = data
        self.search_space = search_space

    def pick_query(self) -> tuple[Any, dict[str, Any]]:
        model = trieste.models.gpflow.models.GaussianProcessRegression(
            trieste.models.gpflow.builders.build_gpr(self.data, self.search_space)
        )
        ask_only = trieste.ask_tell_optimization.AskTellOptimizerNoTraining(
            self.search_space, self.data, model
        )

        query = ask_only.ask()

        breakpoint()  # TODO: return data.

        return query, {}

    def observe(self, query, feedback, evaluation) -> None:
        # TODO: test updating, as opposed to re-creating, model.
        breakpoint()  # TODO: update model


if __name__ == "__main__":
    main()
