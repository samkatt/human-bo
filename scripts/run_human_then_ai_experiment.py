#!/usr/bin/env python

import argparse
from typing import Any

import torch

from human_bo import conf, core, reporting, utils
from human_bo.factories import pick_test_function
from human_bo.joint_optimization import human_suggests_second
from human_bo.test_functions import sample_initial_points


def main():
    torch.set_default_dtype(torch.double)
    human_suggests_second.update_config()

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
    parser.add_argument("--wandb", help="Wandb configuration file.", type=str)

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

    utils.exit_if_exists(path)
    utils.exit_if_exists(args.save_path, negate=True)

    # Create problem.
    f = pick_test_function(exp_conf["problem"], exp_conf["problem_noise"])

    # Create AI agent.
    bo = core.PlainBO(exp_conf["kernel"], exp_conf["acqf"], f._bounds)
    x_init, y_init = sample_initial_points(f, f._bounds, exp_conf["n_init"])
    ai = human_suggests_second.PlainJointAI(bo.pick_queries, x_init, y_init)

    human = human_suggests_second.create_user(exp_conf, f)

    # Create result reporting
    report_step = (
        reporting.initiate_and_create_wandb_logger(args.wandb, exp_conf)
        if args.wandb
        else reporting.print_dot
    )

    # Run actual experiment.
    print(f"Running experiment for {path}")

    res: dict[str, Any] = human_suggests_second.ai_then_human_optimization_experiment(
        ai.pick_queries,
        human,
        f,
        report_step,
        exp_conf["seed"],
        exp_conf["budget"],
    )
    res["conf"] = exp_conf

    torch.save(res, path)


if __name__ == "__main__":
    main()
