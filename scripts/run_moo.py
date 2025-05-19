#!/usr/bin/env python

"""Main entry point: runs MOO."""

import argparse
import pickle
import random
from typing import Any

import numpy as np
import tensorflow as tf
import trieste

import wandb
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

    z = exp_params["latent_objectives"]
    z_dim = len(z)
    o_dim = exp_params["o_dim"] - z_dim
    x_dim = exp_params["x_dim"]

    # Create problem and evaluation.
    if exp_params["scalarization_weights"] is None:
        exp_params["scalarization_weights"] = moo.sample_scalarization_weights(
            o_dim + z_dim
        )
    scalarization_weights = tf.convert_to_tensor(
        exp_params["scalarization_weights"], tf.float64
    )

    assert 0.99 < sum(scalarization_weights) < 1.01, "Preference weights must sum to 1"
    assert len(scalarization_weights) == o_dim + z_dim, "Enter `| -o| ` scalars"

    objectives_problem = trieste_api.create_test_function(
        exp_params["problem"], x_dim, o_dim + z_dim
    )
    assert isinstance(
        objectives_problem,
        trieste.objectives.multi_objectives.MultiObjectiveTestProblem,
    )

    problem = Problem(
        objectives_problem, scalarization_weights, exp_params["problem_noise"], z
    )

    report_step = (
        reporting.initiate_and_create_wandb_logger(
            exp_params["wandb"], exp_params, exp_conf
        )
        if exp_params["wandb"]
        else reporting.print_dot
    )
    evaluation = Evaluation(objectives_problem, scalarization_weights, z, report_step)

    # Create agent with (potentially zero) initial data points.
    x_init = objectives_problem.search_space.sample(exp_params["n_init"])
    assert isinstance(x_init, tf.Tensor)
    f_init, _ = problem.give_feedback(x_init)

    if exp_params["type_agent"] == "bo":
        ai: interaction_loops.Agent = trieste_api.BO(
            trieste.data.Dataset(x_init, f_init["y"]),
            objectives_problem.search_space,
            exp_params["acqf"],
            acqf_options=conf.get_entries_with_tag(exp_params, "acqf-option"),
        )

    elif exp_params["type_agent"] == "composite":
        assert z_dim == 0, "composite agent with latent objectives is not supported."

        ai = trieste_api.CompositeBO(
            exp_params["scalarization_weights"],
            trieste.data.Dataset(x_init, f_init["o"]),
            objectives_problem.search_space,
            exp_params["acqf"],
            acqf_options=conf.get_entries_with_tag(exp_params, "acqf-option"),
        )

    elif exp_params["type_agent"] == "utility-learner":
        partial_objective_function = trieste_api.create_partial_moo_problem(
            objectives_problem, o_dim + z_dim, z
        )
        ai = trieste_api.UtilityBO(
            partial_objective_function,
            trieste.data.Dataset(x_init, f_init["o"]),
            trieste.data.Dataset(f_init["o"], f_init["y"]),
            exp_params["acqf"],
            acqf_options=conf.get_entries_with_tag(exp_params, "acqf-option"),
        )

    elif exp_params["type_agent"] == "moo":
        partial_objective_function = trieste_api.create_partial_moo_problem(
            objectives_problem, o_dim + z_dim, z
        )

        ai = trieste_api.MOO(
            x_init,
            f_init["o"],
            f_init["y"],
            partial_objective_function.search_space,
            exp_params["acqf"],
            acqf_options=conf.get_entries_with_tag(exp_params, "acqf-option"),
        )

    elif exp_params["type_agent"] == "random":
        ai = trieste_api.RandomAgent(objectives_problem.search_space)

    else:
        raise ValueError(f"{exp_params['type_agent']} is not supported")

    print(f"Running experiment for {path}")
    res = interaction_loops.basic_loop(ai, problem, evaluation, exp_params["budget"])

    # Post-process data for easy visualization later.
    res["conf"] = exp_params
    res["conf"]["experiment_type"] = "moo"

    map_y = np.array(
        [[i["map"]] if "map" in i else [np.nan] for i in res["evaluation_stats"]]
    )
    map_x = np.array(
        [
            (
                i["map"]["x"][0]
                if "map" in i
                else np.full(objectives_problem.dim, np.nan)
            )
            for i in res["query_stats"]
        ]
    )
    map_o = [
        (i["o_map"][0] if "o_map" in i else np.full(exp_params["o_dim"], np.nan))
        for i in res["evaluation_stats"]
    ]

    res["results"] = {
        "data_init": {
            "x": np.array(x_init),
            "o": np.array(f_init["o"]),
            "y": np.array(f_init["y"]),
        },
        "queries": np.array(res["query"]),
        "observations": np.array([f["y"] for f in res["feedback"]]),
        "o": np.array([r["o"] for r in res["feedback"]]),
        "o_all": np.array([r["o_all"] for r in res["feedback"]]),
        "y_min": np.array([d["y_min"] for d in res["evaluation_stats"]]),
        "map": {"arg_max": map_x, "max": map_y, "obj": map_o},
    }

    if z_dim > 0:
        res["results"]["data_init"]["z"] = np.array(f_init["z"])
        res["results"]["z"] = (np.array([r["z"] for r in res["feedback"]]),)

    with open(path, "wb") as f:
        pickle.dump(res, f)

    print(f"Done experiments, saved results in {path}")


class Problem(interaction_loops.Problem):
    """The 'problem' in MOO, represented by test and scalar functions."""

    def __init__(
        self,
        objectives_problem: trieste.objectives.multi_objectives.MultiObjectiveTestProblem,
        scalarization_weights: tf.Tensor,
        problem_noise: list[float] | None,
        latent_objectives: list[int],
    ):
        n_obj = len(scalarization_weights)
        for o in latent_objectives:
            assert 0 <= o < n_obj

        self.observer = trieste_api.create_observer(
            objectives_problem.objective, noise_stdev=problem_noise
        )
        self.scalarization_weights = scalarization_weights
        self.observed_objectives = list(set(range(n_obj)) - set(latent_objectives))
        self.latent_objectives = latent_objectives

    def give_feedback(self, query) -> tuple[Any, dict[str, Any]]:
        data_points = self.observer(query)
        assert isinstance(data_points, trieste.data.Dataset)

        objectives = data_points.observations
        assert isinstance(objectives, tf.Tensor)

        # NOTE: we have no noise in the utility function.
        y = moo.scalarize_objectives(objectives, self.scalarization_weights)

        # Split objectives into latent and observed set.
        observed = tf.gather(objectives, self.observed_objectives, axis=-1)

        if len(self.latent_objectives) > 0:
            latent = tf.gather(objectives, self.latent_objectives, axis=-1)
        else:
            # One way of creating a tensor of expected shape.
            latent = tf.reshape((), (len(query), 0))

        return {"x": query, "o_all": objectives, "o": observed, "z": latent, "y": y}, {}

    def observe(self, query, feedback, evaluation) -> None:
        del query, feedback, evaluation


class Evaluation(interaction_loops.Evaluation):
    """Evaluation of MOO problem, mostly about recording true values without noise."""

    def __init__(
        self,
        problem: trieste.objectives.multi_objectives.MultiObjectiveTestProblem,
        scalarization_weights: tf.Tensor,
        latent_objectives: list[int],
        report_step: reporting.StepReport,
    ):
        self.problem = problem
        self.latent_objectives = latent_objectives
        self.observed_objectives = list(
            set(range(len(scalarization_weights))) - set(latent_objectives)
        )

        self.scalarization_weights = scalarization_weights
        self.observed_objectives_weights = tf.gather(
            scalarization_weights, self.observed_objectives, axis=-1
        ).numpy()

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
        del feedback_stats, kwargs
        self.step += 1

        y_observed = feedback["y"].numpy()[0, 0]
        self.obs_min = min(self.obs_min, y_observed)

        o_true = self.problem.objective(query)
        assert isinstance(o_true, tf.Tensor)

        y_true = np.array(moo.scalarize_objectives(o_true, self.scalarization_weights))[
            0, 0
        ]
        self.y_min = min(self.y_min, y_true)

        evaluation = {
            "obs_min": self.obs_min,
            "o_true": np.array(o_true)[0],
            "y_min": self.y_min,
        }

        if "map" in query_stats:
            o_map = self.problem.objective(query_stats["map"]["x"])
            y_arg_map = np.array(
                tf.matmul(
                    o_map,
                    tf.reshape(self.scalarization_weights, (-1, 1)),
                )
            )
            map_prediction_error = np.abs(y_arg_map - query_stats["map"]["y"])

            evaluation["map"] = float(y_arg_map[0, 0])
            evaluation["o_map"] = np.array(o_map)
            evaluation["map_prediction_error"] = float(map_prediction_error[0, 0])

        report = dict(evaluation)
        if "observation_noise" in query_stats:
            report["query_observation_noise"] = query_stats["observation_noise"]
        if "weight_posterior" in query_stats:
            post_centered = (
                query_stats["weight_posterior"] - self.observed_objectives_weights
            )
            report["weight_posterior"] = {
                i: wandb.Histogram(post_centered[:, i])
                for i in range(post_centered.shape[-1])
            }

            report["weight_msre"] = np.sqrt(post_centered**2).mean()
            report["weight_variance"] = np.var(post_centered, axis=0).mean()
            report["weight_variance_variance"] = np.var(np.var(post_centered, axis=0))

        if "weight_map" in query_stats:
            report["weight_map_msre"] = np.mean(
                (self.observed_objectives_weights - query_stats["weight_map"]) ** 2
            )

        self.report_step(report, self.step)

        return None, evaluation


if __name__ == "__main__":
    main()
