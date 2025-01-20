#!/usr/bin/env python

"""Main entry point: runs collaborative BO."""

import argparse
from typing import Any

import torch

from human_bo import conf, core, reporting, utils
from human_bo.factories import pick_test_function
from human_bo.joint_optimization import human_suggests_second
from human_bo.test_functions import sample_initial_points


def main():
    """Main entry human-then-AI experiments."""
    torch.set_default_dtype(torch.double)

    exp_conf = conf.CONFIG
    exp_conf.update(human_suggests_second.CONFIG)

    parser = argparse.ArgumentParser(description="Command description.")
    for arg, values in exp_conf.items():
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
    exp_params = conf.from_ns(args)

    experiment_name = (
        "_".join(
            [
                str(v)
                for k, v in exp_params.items()
                if "experiment-parameter" in exp_conf[k]["tags"]
            ]
        )
        + "_"
        + str(exp_params["seed"])
    )
    path = args.save_path + "/" + experiment_name + ".pt"

    utils.exit_if_exists(path)
    utils.exit_if_exists(args.save_path, negate=True)

    # Create problem.
    f = pick_test_function(exp_params["problem"], exp_params["problem_noise"])

    # Create AI agent.
    bo = core.PlainBO(exp_params["kernel"], exp_params["acqf"], f._bounds)
    x_init, y_init = sample_initial_points(f, f._bounds, exp_params["n_init"])
    ai_agent = human_suggests_second.PlainJointAI(bo.pick_queries, x_init, y_init)

    def ai(x, y, _hist):
        return ai_agent.pick_queries(x, y)

    human = human_suggests_second.create_user(exp_params, f)

    # Create result reporting
    report_step = (
        reporting.initiate_and_create_wandb_logger(args.wandb, exp_params)
        if args.wandb
        else reporting.print_dot
    )

    # Run actual experiment.
    print(f"Running experiment for {path}")

    res: dict[str, Any] = human_suggests_second.ai_then_human_optimization_experiment(
        ai,
        human,
        f,
        report_step,
        exp_params["seed"],
        exp_params["budget"],
    )
    res["conf"] = exp_params

    torch.save(res, path)


if __name__ == "__main__":
    main()
