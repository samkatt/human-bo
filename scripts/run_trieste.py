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

    # FIX: probably remove `pt` suffix? Unless it is pickled?
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

    res["results"] = {
        "data_init": {"x": np.array(x_init), "y": np.array(data_init.observations)},
        "queries": np.stack(res["query"]),
        "observations": np.stack([f.observations for f in res["feedback"]]),
        "y_max": np.stack([d["y_max"] for d in res["evaluation_stats"]]),
    }

    if "map_arg_max" in res["query_stats"][0]:
        breakpoint()  # TODO: support MAP.
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
        del query, query_stats, feedback_stats, kwargs
        # TODO: support noise (record true observation).
        # TODO: support MAP.

        assert isinstance(feedback, trieste.data.Dataset)

        y_observed = np.array(feedback.observations)[0, 0]
        self.y_max = tf.maximum(self.y_max, y_observed).numpy()

        evaluation = {"y_max": self.y_max, "regret": self.y_max - self.minimum}

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
        self.acqf = acqf
        self.acqf_options = acqf_options

        self.ask_tell: (
            trieste.ask_tell_optimization.AskTellOptimizerNoTraining | None
        ) = None

    def pick_query(self) -> tuple[Any, dict[str, Any]]:
        self.step += 1

        # Verify this class is used appropriately:
        # We expect that in between each `pick_query` call, the `observe` method is called.
        # This method will set `self.ask_tell` to `None`. If this does not happen, we crash here.
        assert self.ask_tell is None

        # TODO: improve when to sample random queries.
        # XXX: why is `self.data` potentially `None`?
        if len(self.data) < 2:
            print(
                "WARN (TriesteBO.pick_query): not enough data, returning random sample."
            )
            return self.search_space.sample(1), {}

        trieste.logging.set_step_number(self.step)

        # XXX: update `model`?
        model = trieste.models.gpflow.models.GaussianProcessRegression(
            trieste.models.gpflow.builders.build_gpr(self.data, self.search_space)
        )

        acqf_rule = core.create_trieste_acqf_rule(
            self.acqf, self.search_space, self.acqf_options
        )

        self.ask_tell = trieste.ask_tell_optimization.AskTellOptimizerNoTraining(
            self.search_space, self.data, model, acquisition_rule=acqf_rule
        )

        query = self.ask_tell.ask()

        # TODO: support reporting MAP.
        return query, {}

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
