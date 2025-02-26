"""Entry-point for subjective multi-objective optimization."""

import argparse

import torch

from human_bo import conf, interaction_loops, reporting, test_functions, utils
from human_bo.moo import moo_core


def main():
    """Main entry MOO experiments."""
    torch.set_default_dtype(torch.double)

    exp_conf = conf.CONFIG
    exp_conf.update(moo_core.CONFIG)

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

    path = exp_params["save_dir"] + "/" + experiment_name + ".pt"

    utils.exit_if_exists(path)
    utils.create_directory_if_does_not_exist(exp_params["save_dir"])

    assert exp_params["n_init"] == 0, "There is no support for initial points in MOO"

    torch.manual_seed(exp_params["seed"])

    # TODO: check for handling multiple values for `problem_noise`?
    moo_function = test_functions.pick_moo_test_function(
        exp_params["problem"], exp_params["problem_noise"]
    )
    utility_function = moo_core.create_utility_function(
        exp_params["preference_weights"]
    )

    report_step = (
        reporting.initiate_and_create_wandb_logger(
            exp_params["wandb"], exp_params, exp_conf
        )
        if exp_params["wandb"]
        else reporting.print_dot
    )

    evaluation = moo_core.MOOEvaluation(moo_function, utility_function, report_step)
    agent = moo_core.create_AI(
        moo_function, exp_params["algorithm"], exp_params["kernel"], exp_params["acqf"]
    )
    problem = moo_core.MOOProblem(moo_function, utility_function)

    print(f"Running experiment for {path}")
    res = interaction_loops.basic_loop(agent, problem, evaluation, exp_params["budget"])

    res["conf"] = exp_params

    torch.save(res, path)
    print(f"Done experiments, saved results in {path}")


if __name__ == "__main__":
    main()
