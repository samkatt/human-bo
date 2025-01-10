#!/usr/bin/env python

import argparse
import os
from typing import Any

import torch

import human_bo.conf as conf
from human_bo import core
from human_bo.factories import pick_test_function
from human_bo.joint_optimization.human_suggests_second import (
    PlainJointAI,
    create_test_both_queries_problem,
)
from human_bo.test_functions import sample_initial_points

if __name__ == "__main__":
    torch.set_default_dtype(torch.double)

    parser = argparse.ArgumentParser(description="Command description.")
    for arg, values in conf.CONFIG.items():
        parser.add_argument(
            "-" + values["shorthand"],
            "--" + arg,
            help=values["help"],
            type=values["type"],
            **values["parser-arguments"],
        )

    parser.add_argument("-f", "--save_path", help="Name of saving directory.", type=str)
    args = parser.parse_args()
    exp_conf = conf.from_ns(args)

    experiment_name = (
        "_".join(
            [
                str(v)
                for k, v in exp_conf.items()
                if "experiment-parameter" in conf.CONFIG[k]["tags"]
            ]
        )
        + "_"
        + str(exp_conf["seed"])
    )
    path = args.save_path + "/" + experiment_name + ".pt"

    if os.path.isfile(path):
        print(f"File {path} already exists, aborting run!")
        exit()
    if not os.path.isdir(args.save_path):
        print(f"Save path {args.save_path} is not an existing directory")
        exit()

    # Create problem.
    f = pick_test_function(exp_conf["problem"])
    problem = create_test_both_queries_problem(f, exp_conf["problem_noise"])

    # Create AI agent.
    bo = core.PlainBO(exp_conf["kernel"], exp_conf["acqf"], f._bounds)
    train_x, train_y, _ = sample_initial_points(
        f, f._bounds, exp_conf["n_init"], exp_conf["problem_noise"]
    )
    ai = PlainJointAI(bo.pick_queries, train_x, train_y)

    # Create human agent (pretend it is also BO).
    bo_human = core.PlainBO(exp_conf["kernel"], exp_conf["acqf"], f._bounds)
    train_x_human, train_y_human, _ = sample_initial_points(
        f, f._bounds, exp_conf["n_init"], exp_conf["problem_noise"]
    )
    human = PlainJointAI(bo_human.pick_queries, train_x_human, train_y_human)
    # human = lambda history, ai_action: (
    #     core.random_query(f._bounds).unsqueeze(0),
    #     {"descr": "Human uses random query and has no stats."},
    # )

    # Run actual experiment.
    print(f"Running experiment for {path}", end="", flush=True)

    res: dict[str, Any] = core.ai_then_human_optimization_experiment(
        ai.pick_queries,
        lambda hist, ai_action: human.pick_queries(hist),
        problem,
        exp_conf["seed"],
        exp_conf["budget"],
    )
    res["conf"] = exp_conf

    torch.save(res, path)
