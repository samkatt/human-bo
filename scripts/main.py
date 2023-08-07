import functools
import multiprocessing as mp
import torch
from human_bo import factories, core
import argparse

torch.set_default_dtype(torch.double)


def parser_bo():
    """Parser used to run the algorithm from an already known crn.

    - Output:
        * parser: ArgumentParser object.
    """

    parser = argparse.ArgumentParser(description="Command description.")

    parser.add_argument(
        "-n", "--N_REP", help="int, number of reps for stds", type=int, default=1
    )
    parser.add_argument(
        "-ni", "--N_INIT", help="int, size of initial dataset", type=int, default=1
    )
    parser.add_argument(
        "-se", "--seed", default=None, help="int, random seed", type=int
    )
    parser.add_argument(
        "-s", "--savefolder", default=None, type=str, help="Name of saving directory."
    )
    parser.add_argument(
        "-b",
        "--budget",
        help="BO Budget",
        default=10,
        type=int,
    )
    parser.add_argument(
        "-k",
        "--kernels",
        nargs="*",
        type=str,
        default=["RBF"],
        help="list of kernels to try.",
    )
    parser.add_argument(
        "-a",
        "--acqfs",
        nargs="*",
        type=str,
        default=["MES"],
        help="list of BO acquisition function to try.",
    )
    parser.add_argument(
        "-e",
        "--experiments",
        nargs="*",
        type=str,
        default=["Forrester"],
        help="list of test functions to optimize.",
    )
    return parser


def main(N_REP, N_INIT, budget, kernels, acqfs, experiments, seed, savefolder):
    """Main function that is called with arguments parsed by `parser_bo`"""

    combi = factories.build_combinations(
        N_REP, experiments, kernels, acqfs, N_INIT, seed
    )
    with mp.Pool() as p:
        p.map(
            functools.partial(core.parallel_eval, combi, budget, savefolder),
            range(len(combi)),
        )
    p.close()


if __name__ == "__main__":
    parser = parser_bo()
    main(**vars(parser.parse_args()))
