"""Stuff for reporting results and progress"""

from typing import Any, Callable

import yaml

import wandb
from human_bo import conf, utils

StepData = dict[str, Any]
StepReport = Callable[[StepData, int], None]


def print_dot(step_data: StepData, step: int) -> None:
    """Prints a dot to the terminal, can be used to track progress."""
    del step_data, step
    print(".", end="", flush=True)


def initiate_and_create_wandb_logger(
    path_to_conf_file: str,
    exp_params: dict[str, Any],
    exp_conf: dict[str, dict[str, Any]] | None = None,
) -> StepReport:
    """Wakes up wandb and returns a function that will log `step_data`."""
    assert path_to_conf_file

    if exp_conf is None:
        exp_conf = conf.CONFIG

    with open(path_to_conf_file) as f:
        wandb_conf = yaml.safe_load(f)

    dir_wandb = "./wandb/" + "_".join(
        conf.get_values_with_tag(exp_params, "experiment-parameter", exp_conf)
    )
    utils.create_directory_if_does_not_exist(dir_wandb)

    wandb_conf["dir"] = dir_wandb

    # breakpoint()

    wandb.init(config=exp_params, **wandb_conf)
    return lambda step_data, step: wandb.log(step_data, step=step)


def print_step_data(step_data: StepData, step: int) -> None:
    """Prints a dot to the terminal, can be used to track progress."""
    del step
    print(step_data)
