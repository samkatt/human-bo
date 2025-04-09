#!/usr/bin/env python

"""Main entry point: runs MOO."""

import argparse
import pickle
import random
from typing import Any

import numpy as np
import tensorflow as tf
import trieste

from human_bo import conf, interaction_loops, moo, reporting, trieste_api, utils


def main():
    """Main entry human-feedback experiments."""
    exp_conf = conf.CONFIG
    exp_conf.update(moo.CONFIG)

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
    path = exp_params["save_dir"] + "/" + experiment_name + ".pkl"

    utils.exit_if_exists(path)
    utils.create_directory_if_does_not_exist(exp_params["save_dir"])

    tf.random.set_seed(exp_params["seed"])
    np.random.seed(exp_params["seed"])
    random.seed(exp_params["seed"])

    # Create problem and evaluation.
    if exp_params["scalarization_weights"] is None:
        exp_params["scalarization_weights"] = moo.sample_scalarization_weights(
            exp_params["o_dim"]
        )

    scalarization_weights = tf.convert_to_tensor(
        exp_params["scalarization_weights"], tf.float64
    )
    assert 0.99 < sum(scalarization_weights) < 1.01, "Preference weights must sum to 1"
    assert len(scalarization_weights) == exp_params["o_dim"], "Enter `| -o| ` scalars"

    trieste_problem = trieste_api.create_trieste_test_function(
        exp_params["problem"], exp_params["x_dim"], exp_params["o_dim"]
    )
    assert isinstance(
        trieste_problem, trieste.objectives.multi_objectives.MultiObjectiveTestProblem
    )
    problem = Problem(
        trieste_problem, scalarization_weights, exp_params["problem_noise"]
    )

    report_step = (
        reporting.initiate_and_create_wandb_logger(
            exp_params["wandb"], exp_params, exp_conf
        )
        if exp_params["wandb"]
        else reporting.print_dot
    )
    evaluation = Evaluation(trieste_problem, scalarization_weights, report_step)

    # Create Agents
    x_init = trieste_problem.search_space.sample(exp_params["n_init"])
    o_init = problem.observer(x_init)
    assert isinstance(o_init, trieste.data.Dataset) and isinstance(
        o_init.observations, tf.Tensor
    )
    data_init = trieste.data.Dataset(
        x_init,
        trieste_api.scalarize_objectives(o_init.observations, scalarization_weights),
    )

    if exp_params["acqf"] != "random":
        ai: interaction_loops.Agent = trieste_api.TriesteBO(
            data_init,
            trieste_problem.search_space,
            exp_params["acqf"],
            acqf_options=conf.get_entries_with_tag(exp_params, "acqf-option"),
        )
    else:
        ai = trieste_api.RandomAgent(trieste_problem.search_space)

    print(f"Running experiment for {path}")
    res = interaction_loops.basic_loop(ai, problem, evaluation, exp_params["budget"])

    # Post-process data for easy visualization later.
    res["conf"] = exp_params
    res["conf"]["experiment_type"] = "trieste"

    map_y = np.stack(
        [i["map"]["y"][0] if "map" in i else [np.nan] for i in res["query_stats"]]
    )
    map_x = np.stack(
        [
            (i["map"]["x"][0] if "map" in i else np.full(trieste_problem.dim, np.nan))
            for i in res["query_stats"]
        ]
    )
    map_o = [
        (
            i["map"]["o"][0]
            if "map" in i and "o" in i["map"]
            else np.full(exp_params["o_dim"], np.nan)
        )
        for i in res["query_stats"]
    ]

    res["results"] = {
        "data_init": {
            "x": np.array(x_init),
            "o": np.array(o_init),
            "y": np.array(data_init.observations),
        },
        "queries": np.stack(res["query"]),
        "observations": np.stack([f.observations for f in res["feedback"]]),
        "objectives": np.stack([r["objectives"] for r in res["feedback_stats"]]),
        "y_min": np.stack([d["y_min"] for d in res["evaluation_stats"]]),
        "map": {"arg_max": map_x, "max": map_y, "obj": map_o},
    }

    with open(path, "wb") as f:
        pickle.dump(res, f)

    print(f"Done experiments, saved results in {path}")


class Problem(interaction_loops.Problem):
    """The 'problem' in MOO, represented by test and scalar functions."""

    def __init__(
        self,
        trieste_problem: trieste.objectives.multi_objectives.MultiObjectiveTestProblem,
        scalarization_weights: tf.Tensor,
        problem_noise: list[float] | None,
    ):
        self.observer = trieste_api.create_trieste_observer(
            trieste_problem.objective, noise_stdev=problem_noise
        )
        self.scalarization_weights = scalarization_weights

    def give_feedback(self, query) -> tuple[Any, dict[str, Any]]:
        objectives = self.observer(query)
        assert isinstance(objectives, trieste.data.Dataset)
        assert isinstance(objectives.observations, tf.Tensor)

        cost = trieste_api.scalarize_objectives(
            objectives.observations, self.scalarization_weights
        )

        # The agent gets to observe only the outcome cost, so the feedback is `(x, u)`.
        feedback = trieste.data.Dataset(query, cost)
        return feedback, {"objectives": np.array(objectives.observations)}

    def observe(self, query, feedback, evaluation) -> None:
        del query, feedback, evaluation


class Evaluation(interaction_loops.Evaluation):
    """Evaluation of MOO problem, mostly about recording true values without noise."""

    def __init__(
        self,
        problem: trieste.objectives.multi_objectives.MultiObjectiveTestProblem,
        scalarization_weights: tf.Tensor,
        report_step: reporting.StepReport,
    ):
        self.problem = problem
        self.scalarization_weights = scalarization_weights

        self.obs_min, self.y_min = np.inf, np.inf
        self.step = -1
        self.report_step = report_step

    def __call__(
        self,
        query,
        feedback,
        query_stats: dict[str, Any],
        feedback_stats: dict[str, Any],
        **kwargs,
    ) -> tuple[Any, dict[str, Any]]:
        del query_stats, feedback_stats, kwargs
        self.step += 1

        y_observed = np.array(feedback.observations)[0, 0]
        self.obs_min = min(self.obs_min, y_observed)

        o_true = self.problem.objective(query)
        assert isinstance(o_true, tf.Tensor)

        y_true = np.array(
            trieste_api.scalarize_objectives(o_true, self.scalarization_weights)
        )[0, 0]
        self.y_min = min(self.y_min, y_true)

        evaluation = {
            "obs_min": self.obs_min,
            "o_true": np.array(o_true)[0],
            "y_min": self.y_min,
        }
        self.report_step(evaluation, self.step)

        return None, evaluation


if __name__ == "__main__":
    main()
