#!/usr/bin/env python

"""Main entry point: runs Trieste BO."""

import argparse
import pickle
from typing import Any

import numpy as np
import tensorflow as tf
import trieste

from human_bo import conf, interaction_loops, reporting, trieste_api, utils


def main():
    """Main entry human-feedback experiments."""
    exp_conf = conf.CONFIG

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

    # Create problem and evaluation.
    trieste_problem = trieste_api.create_trieste_test_function(exp_params["problem"])
    assert isinstance(
        trieste_problem, trieste.objectives.single_objectives.SingleObjectiveTestProblem
    )
    observer = trieste_api.create_trieste_observer(
        trieste_problem.objective, noise_stdev=exp_params["problem_noise"]
    )
    problem = Problem(observer)

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

    assert isinstance(data_init, trieste.data.Dataset)

    if exp_params["acqf"] != "random":
        ai: interaction_loops.Agent = TriesteBO(
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

    res["results"] = {
        "data_init": {"x": np.array(x_init), "y": np.array(data_init.observations)},
        "queries": np.stack(res["query"]),
        "observations": np.stack([f.observations for f in res["feedback"]]),
        "y_min": np.stack([d["y_min"] for d in res["evaluation_stats"]]),
        "map": {"arg_max": map_x, "max": map_y},
    }

    with open(path, "wb") as f:
        pickle.dump(res, f)

    print(f"Done experiments, saved results in {path}")


class Problem(interaction_loops.Problem):
    """The 'problem' in BO, represented by (optional) user model."""

    def __init__(self, observer: trieste.observer.Observer):
        self.observer = observer

    def give_feedback(self, query) -> tuple[Any, dict[str, Any]]:
        feedback = self.observer(query)
        return feedback, {}

    def observe(self, query, feedback, evaluation) -> None:
        del query, feedback, evaluation


class Evaluation(interaction_loops.Evaluation):
    def __init__(
        self,
        problem: trieste.objectives.single_objectives.SingleObjectiveTestProblem,
        report_step: reporting.StepReport,
    ):
        self.problem = problem
        self.minimum = float(np.array(problem.minimum)[0])
        assert isinstance(self.minimum, float)

        self.obs_min, self.y_min = np.inf, np.inf
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
        del feedback_stats, kwargs

        assert isinstance(feedback, trieste.data.Dataset)

        y_observed = np.array(feedback.observations)[0, 0]
        self.obs_min = min(self.obs_min, y_observed)

        y_true = np.array(self.problem.objective(query))[0, 0]
        self.y_min = min(self.y_min, y_true)

        evaluation = {
            "obs_min": self.obs_min,
            "y_true": y_true,
            "y_min": self.y_min,
            "regret_obs": self.obs_min - self.minimum,
            "regret_true": self.y_min - self.minimum,
        }

        if "map" in query_stats:
            y_arg_map = float(
                np.array(self.problem.objective(query_stats["map"]["x"]))[0, 0]
            )
            map_prediction_error = np.abs(
                y_arg_map - float(query_stats["map"]["y"][0, 0])
            )

            evaluation["map"] = y_arg_map
            evaluation["regret_map"] = y_arg_map - self.minimum
            evaluation["map_prediction_error"] = map_prediction_error

        self.step += 1
        self.report_step(evaluation, self.step)

        return None, evaluation


class TriesteBO(interaction_loops.Agent):

    def __init__(
        self,
        data: trieste.data.Dataset,
        search_space: trieste.space.SearchSpace,
        acqf: str,
        acqf_options: dict[str, Any],
    ):
        self.data = data
        self.search_space = search_space
        self.step = -1
        self.acqf = trieste_api.create_trieste_acqf(
            acqf, self.search_space, acqf_options
        )
        self.mean_acqf = trieste_api.create_trieste_acqf("mean", self.search_space, {})

    def pick_query(self) -> tuple[Any, dict[str, Any]]:
        self.step += 1

        # Create the model (or return random sample if fails).
        try:
            y_sca, y_mean, y_std = utils.normalize(self.data.observations)
            data_sca = trieste.data.Dataset(self.data.query_points, y_sca)
            model = trieste_api.create_trieste_gp(data_sca, self.search_space)

        except tf.errors.InvalidArgumentError:
            print(
                "WARN: `TriesteBO.pick_query` failed to fit model, returning random sample."
            )
            return self.search_space.sample(1), {}

        # Pick query given model.
        query = trieste_api.optimize_trieste_acqf(
            self.acqf, data_sca, model, self.search_space
        )

        arg_map = trieste_api.optimize_trieste_acqf(
            self.mean_acqf, data_sca, model, self.search_space
        )
        # Un-normalize predicted MAP.
        map_mean = model.predict(arg_map)[0] * y_std + y_mean

        return query, {"map": {"x": np.array(arg_map), "y": np.array(map_mean)}}

    def observe(self, query, feedback, evaluation) -> None:
        del query, evaluation
        self.data = self.data + feedback


if __name__ == "__main__":
    main()
