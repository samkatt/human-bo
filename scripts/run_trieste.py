#!/usr/bin/env python

"""Main entry point: runs Trieste BO."""

import argparse
import pickle
from typing import Any

import numpy as np
import tensorflow as tf
import trieste

from human_bo import (
    conf,
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

    trieste.logging.set_tensorboard_writer(
        tf.summary.create_file_writer(
            exp_params["save_dir"] + "/tensorboard/" + experiment_name
        )
    )

    # Create problem and evaluation.
    # TODO: consider different problems.
    trieste_problem = test_functions.TriesteLevy1
    observer = trieste.objectives.utils.mk_observer(trieste_problem.objective)

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
    data_init = observer(x_init)

    ai = TriesteBO(data_init, trieste_problem.search_space)
    # TODO: consider noise.
    problem = Problem(observer)

    print(f"Running experiment for {path}")
    res = interaction_loops.basic_loop(ai, problem, evaluation, exp_params["budget"])

    # Post-process data for easy visualization later.
    res["conf"] = exp_params
    res["conf"]["experiment_type"] = "trieste"

    res["results"] = {
        "data_init": {"x": np.array(x_init), "y": np.array(data_init.observations)},
        "queries": np.stack(res["query"]),
        "observations": np.stack([f.observations for f in res["feedback"]]),
        "y_max": np.stack([d["y_max"] for d in res["evaluation_stats"]]),
    }

    if "map_arg_max" in res["query_stats"][0]:
        breakpoint()  # TODO: verify below.
        map_arg_max = np.stack([i["map_arg_max"] for i in res["query_stats"]])
        map_max = np.array(observer(tf.convert_to_tensor(map_arg_max)).observations)
        res["results"]["map"] = {"arg_max": map_arg_max, "max": map_max}

    # torch.save(res, path)
    with open(path, "wb") as f:
        pickle.dump(res, f)

    print(f"Done experiments, saved results in {path}")


class Problem(interaction_loops.Problem):
    """The 'problem' in BO, represented by (optional) user model."""

    def __init__(self, observer: trieste.observer.Observer):
        self.observer = observer

    def give_feedback(self, query) -> tuple[Any, dict[str, Any]]:
        # TODO: record true value without noise.
        feedback = self.observer(query)
        return feedback, {}

    def observe(self, query, feedback, evaluation) -> None:
        del query, feedback, evaluation


class Evaluation(interaction_loops.Evaluation):
    def __init__(self, problem, report_step: reporting.StepReport):
        self.problem = problem
        self.y_max = -np.inf
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
        del query, feedback_stats, kwargs

        assert isinstance(feedback, trieste.data.Dataset)

        y_observed = feedback.observations[0, 0]
        self.y_max = tf.maximum(self.y_max, y_observed)

        evaluation = {
            # TODO: "y_observed": y_true,
            "y_max": float(self.y_max),
        }

        if "map_arg_max" in query_stats:
            evaluation["regret"] = self.problem(query_stats["map_arg_max"], noise=False)

        self.step += 1
        self.report_step(evaluation, self.step)

        return None, evaluation


class TriesteBO(interaction_loops.Agent):

    # TODO: add types.
    def __init__(self, data, search_space):
        # TODO: account for different acquisition functions.
        self.data = data
        self.search_space = search_space
        self.step = -1
        self.ask_tell = None

    def pick_query(self) -> tuple[Any, dict[str, Any]]:
        self.step += 1

        assert self.ask_tell is None

        if len(self.data) < 2:
            print(
                "WARN (TriesteBO.pick_query): not enough data, returning random sample."
            )
            return self.search_space.sample(1), {}

        trieste.logging.set_step_number(self.step)
        model = trieste.models.gpflow.models.GaussianProcessRegression(
            trieste.models.gpflow.builders.build_gpr(self.data, self.search_space)
        )
        self.ask_tell = trieste.ask_tell_optimization.AskTellOptimizerNoTraining(
            self.search_space, self.data, model
        )

        query = self.ask_tell.ask()

        # TODO: return argmax map.
        return query, {}

    def observe(self, query, feedback, evaluation) -> None:
        del query, evaluation
        self.data = self.data + feedback
        if not self.ask_tell is None:
            self.ask_tell.tell(feedback)
            self.ask_tell = None


if __name__ == "__main__":
    main()
