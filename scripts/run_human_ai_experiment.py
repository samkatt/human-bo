#!/usr/bin/env python

import argparse
import os
import sys

import torch

from human_bo import conf, core

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
        sys.exit()
    if not os.path.isdir(args.save_path):
        print(f"Save path {args.save_path} is not an existing directory")
        sys.exit()

    print(f"Running experiment for {path}", end="", flush=True)
    res = core.human_feedback_experiment(**exp_conf)
    res["conf"] = exp_conf

    torch.save(res, path)
