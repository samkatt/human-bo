import os
import torch
from human_bo import core
import argparse

import human_bo.conf as conf

import traceback
import warnings
import sys


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    """Custom function to print (warning) traces properly

    Taken from https://stackoverflow.com/questions/22373927/get-traceback-of-warnings.

    Really, just for debugging and figuring out where the warnings are generated from.
    """
    log = file if hasattr(file, "write") else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


if __name__ == "__main__":
    torch.set_default_dtype(torch.double)
    warnings.showwarning = warn_with_traceback

    parser = argparse.ArgumentParser(description="Command description.")
    for arg, values in conf.CONFIG.items():
        parser.add_argument(
            "-" + values["shorthand"],
            "--" + arg,
            help=values["help"],
            type=values["type"],
        )

    parser.add_argument("-p", "--save_path", help="Name of saving directory.", type=str)
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

    res = core.eval_model(**exp_conf)
    res["conf"] = exp_conf

    torch.save(res, path)
