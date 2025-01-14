"""Stuff for reporting results and progress"""

from typing import Any, Callable

import wandb
import yaml

StepData = dict[str, Any]
StepReport = Callable[[StepData, int], None]


def print_dot(step_data: StepData, step: int) -> None:
    print(".", end="", flush=True)


def initiate_and_create_wandb_logger(
    path_to_conf_file: str, exp_conf: dict[str, Any]
) -> StepReport:
    """Creates a"""
    assert path_to_conf_file

    with open(path_to_conf_file) as f:
        wandb_conf = yaml.safe_load(f)

    wandb.init(config=exp_conf, **wandb_conf)
    return lambda step_data, step: wandb.log(step_data, step=step)
