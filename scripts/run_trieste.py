#!/usr/bin/env python

"""Main entry point: runs Trieste BO."""

import argparse
import pickle
from typing import Any

import numpy as np
import tensorflow as tf
import trieste
import trieste.logging

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
    parser.add_argument(
        "--tensorboard",
        help="Log results to `save_dir/tensorboard/experiment_name`",
        action="store_true",
    )
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

    if exp_params["tensorboard"]:
        trieste.logging.set_summary_filter(lambda _: True)
        trieste.logging.set_tensorboard_writer(
            tf.summary.create_file_writer(
                exp_params["save_dir"] + "/tensorboard/" + experiment_name
            )
        )

    # Create problem and evaluation.
    trieste_problem = test_functions.create_trieste_test_function(exp_params["problem"])
    assert isinstance(
        trieste_problem, trieste.objectives.single_objectives.SingleObjectiveTestProblem
    )
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

    ai = TriesteBO(
        data_init,
        trieste_problem.search_space,
        exp_params["acqf"],
        acqf_options=conf.get_entries_with_tag(exp_params, "acqf-option"),
    )
    # TODO: support noise.
    problem = Problem(observer)

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
        "y_max": np.stack([d["y_max"] for d in res["evaluation_stats"]]),
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

        self.obs_max = -np.inf
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
        # TODO: support noise (record true observation).

        assert isinstance(feedback, trieste.data.Dataset)

        y_observed = np.array(feedback.observations)[0, 0]

        self.obs_max = tf.maximum(self.obs_max, y_observed).numpy()

        evaluation = {
            "y_max": self.obs_max,
            "regret_obs": self.obs_max - self.minimum,
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
        self.acqf = core.create_trieste_acqf_rule(acqf, self.search_space, acqf_options)

        self.ask_tell: (
            trieste.ask_tell_optimization.AskTellOptimizerNoTraining | None
        ) = None

    def pick_query(self) -> tuple[Any, dict[str, Any]]:
        self.step += 1
        trieste.logging.set_step_number(self.step)

        # Verify this class is used appropriately:
        # We expect that in between each `pick_query` call, the `observe` method is called.
        # This method will set `self.ask_tell` to `None`. If this does not happen, we crash here.
        assert self.ask_tell is None

        # Here we do the main optimization step.
        # For this, we use `Trieste` "AskTell" interface:
        # (https://secondmind-labs.github.io/trieste/3.1.0/notebooks/ask_tell_optimization.html)

        # The real important steps are the usual, though: (1) get posterior, (2) get acquisition optimization, (3) run it.

        # 1. Create the model (or return random sample if fails).
        try:
            model = core.create_trieste_gp(self.data, self.search_space)

        except tf.errors.InvalidArgumentError:
            print(
                "WARN: `TriesteBO.pick_query` failed to fit model, returning random sample."
            )
            return self.search_space.sample(1), {}

        # 2. Create the acquisition optimizer.
        acqf_rule: trieste.acquisition.rule.AcquisitionRule = (
            trieste.acquisition.rule.EfficientGlobalOptimization(self.acqf)
        )

        # 3. Optimize.
        self.ask_tell = trieste.ask_tell_optimization.AskTellOptimizerNoTraining(
            self.search_space, self.data, model, acquisition_rule=acqf_rule
        )
        query = self.ask_tell.ask()

        # For statistics, we may be interested in the maximum a posterior: the mean of the posterior.
        mean_rule: trieste.acquisition.rule.AcquisitionRule = (
            trieste.acquisition.rule.EfficientGlobalOptimization(
                trieste.acquisition.function.function.NegativePredictiveMean()
            )
        )
        arg_map = mean_rule.acquire_single(self.search_space, model, self.data)
        map_mean, _ = model.predict(arg_map)

        return query, {"map": {"x": np.array(arg_map), "y": np.array(map_mean)}}

    def observe(self, query, feedback, evaluation) -> None:
        del query, evaluation
        self.data = self.data + feedback

        # We will `tell` our observations if we used `self.ask_tell` to get the queries.
        # We then set `self.ask_tell` to None, to make sure any mistaken use of this class is avoided.
        if self.ask_tell is not None:
            self.ask_tell.tell(feedback)
            self.ask_tell = None


if __name__ == "__main__":
    main()
