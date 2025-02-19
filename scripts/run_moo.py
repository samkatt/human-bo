"""Entry-point for subjective multi-objective optimization."""

import argparse

import torch

from human_bo import conf, test_functions, utils, reporting
from human_bo.moo import moo_scratchpad, moo_core


def main():
    """Main entry MOO experiments."""
    torch.set_default_dtype(torch.double)

    exp_conf = conf.CONFIG
    exp_conf.update(moo_scratchpad.CONFIG)

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
        "-f", "--save_path", help="Name of saving directory.", type=str, required=True
    )
    parser.add_argument("--wandb", help="Wandb configuration file.", type=str)
    args = parser.parse_args()
    exp_params = conf.from_ns(args)

    experiment_name = "_".join(
        conf.get_values_with_tag(exp_params, "experiment-parameter", exp_conf)
        + [str(exp_params["seed"])]
    )

    path = args.save_path + "/" + experiment_name + ".pt"

    utils.exit_if_exists(path)
    utils.create_directory_if_does_not_exist(args.save_path)

    torch.manual_seed(exp_params["seed"])

    # TODO: add MOO to `test_functions`.
    # TODO: check for handling multiple values for `problem_noise`?
    moo_function = test_functions.pick_test_function(
        exp_params["problem"], exp_params["problem_noise"]
    )
    report_step = (
        reporting.initiate_and_create_wandb_logger(args.wandb, exp_params, exp_conf)
        if args.wandb
        else reporting.print_dot
    )

    # TODO: finish this.
    evaluation = moo_scratchpad.MOOEvaluation(moo_function, utility_function, report_step)

if __name__ == "__main__":
    torch.manual_seed(0)

    res = moo_scratchpad.moo_learn_subjective_function()
    res["conf"] = {""}
    print("Saving!")
    torch.save(res, "./tmp.pt")
