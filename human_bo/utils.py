import argparse

import matplotlib as mpl
from botorch.test_functions import Branin, Hartmann, Rosenbrock
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel

from human_bo.test_functions import Zhou


def set_matplotlib_params():
    """Set matplotlib params."""

    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rc("font", family="serif")
    mpl.rcParams.update(
        {
            "font.size": 24,
            "lines.linewidth": 2,
            "axes.labelsize": 24,  # fontsize for x and y labels
            "axes.titlesize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "axes.linewidth": 2,
            "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
            "text.usetex": True,  # use LaTeX to write all text
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.grid": True,
        }
    )


def adapt_save_fig(fig, filename="test.pdf"):
    """Remove right and top spines, set bbox_inches and dpi."""

    for ax in fig.get_axes():
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    fig.savefig(filename, bbox_inches="tight", dpi=300)


def parser_bo():
    """
    Parser used to run the algorithm from an already known crn.
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


def pick_kernel(ker, dim):
    "Instantiate the given kernel."

    # ScaleKernel adds the amplitude hyperparameter
    if ker == "RBF":
        K = ScaleKernel(RBFKernel(ard_num_dims=dim))
    elif ker == "Matern":
        K = ScaleKernel(MaternKernel(ard_num_dims=dim))
    return K


def pick_test_function(func):
    "Instantiate the given function to optimize."

    if func == "Zhou":
        testfunc = Zhou()
    elif func == "Hartmann":
        testfunc = Hartmann(negate=True)
    elif func == "Branin":
        testfunc = Branin(negate=True)
    elif func == "Rosenbrock":
        testfunc = Rosenbrock(dim=2, negate=True, bounds=[(-5.0, 5.0), (-5.0, 5.0)])
    return testfunc
