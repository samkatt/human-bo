import os
import torch
from human_bo import core
import argparse

import human_bo.conf as conf

torch.set_default_dtype(torch.double)


if __name__ == "__main__":
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
