#!/usr/bin/env python

"""Main entry point: runs typical BO with (potentially) human giving the feedback."""

import argparse

import torch

from human_bo import conf, core, human_feedback_experiments, reporting, utils


def main():
    """Main entry human-feedback experiments."""
    torch.set_default_dtype(torch.double)
    human_feedback_experiments.update_config()

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

    # Create result reporting
    report_step = (
        reporting.initiate_and_create_wandb_logger(args.wandb, exp_conf)
        if args.wandb
        else reporting.print_dot
    )

    print(f"Running experiment for {path}")
    res = core.human_feedback_experiment(**exp_conf, report_step=report_step)
    res["conf"] = exp_conf
    res["conf"] = exp_conf

    torch.save(res, path)


if __name__ == "__main__":
    main()
