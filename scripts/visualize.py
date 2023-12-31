"""Simple script to visualize the results in a folder generated by the main script"""

import warnings
from botorch import settings
from botorch.fit import fit_gpytorch_model
import numpy as np
from gpytorch.mlls import ExactMarginalLogLikelihood
import math
from botorch.models import SingleTaskGP
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch
import argparse
from human_bo import utils
from human_bo.conf import CONFIG
from human_bo.factories import pick_acqf, pick_kernel, pick_user_model, pick_test_function

from human_bo.visualization import set_matplotlib_params
from human_bo.utils import recursively_filter_dict


def compare_regrets_over_time(files: list[str]) -> None:
    """Visualizes (plots) the regrets in of `files`

    Asserts that the results being compared have the same hyper parameter.

    :files: list of file paths (string)
    """
    experiment_config = torch.load(files[0])["conf"]

    results = {}
    for file in files:
        # Load the file and initiated configurations and results.
        new_results = torch.load(file)
        new_conf = new_results["conf"]

        n_init, budget = new_conf["n_init"], new_conf["budget"]
        optimal_value = pick_test_function(new_conf["function"]).optimal_value
        y = new_results["true_Y"]

        regrets = torch.tensor(
            [optimal_value - y[: n_init + i].max() for i in range(budget + 1)]
        )

        # Make sure experiment shares the same parameters.
        for k, c in CONFIG.items():
            if "experiment-hyper-parameter" not in c["tags"]:
                continue  # We only check for "experiment-hyper-parameter" configurations

            assert (
                experiment_config[k] == new_conf[k]
            ), f"Different {k} parameters used: {experiment_config[k]} and {new_conf[k]}"

        # Store results.
        current_results_for_experiment = results
        for k, c in CONFIG.items():
            if "experiment-parameter" not in c["tags"]:
                continue  # We store results per "experiment-parameter" combination.

            # dig into results
            try:
                current_results_for_experiment = current_results_for_experiment[
                    new_conf[k]
                ]
            except KeyError:
                current_results_for_experiment[new_conf[k]] = {}
                current_results_for_experiment = current_results_for_experiment[
                    new_conf[k]
                ]

        # `r` is now a bunch of results (one per seed)
        if current_results_for_experiment:
            current_results_for_experiment["regrets"] = torch.cat(
                (current_results_for_experiment["regrets"], regrets.unsqueeze(-1)),
                dim=1,
            )
        else:
            # First time looking at this particular experimental setup!
            current_results_for_experiment["conf"] = new_conf
            current_results_for_experiment["regrets"] = regrets.unsqueeze(-1)

    # Go over all results and compute (and store) their mean and standard error.
    for e in recursively_filter_dict(
        results, lambda _, v: isinstance(v, dict) and "regrets" in v
    ):
        r = e["regrets"]
        e["mean_regret"] = r.mean(axis=1)
        e["std_regret"] = 1.96 * r.std(axis=1) / math.sqrt(r.shape[1])

    fig = plt.figure(figsize=(8, 6))

    for e in recursively_filter_dict(
        results, lambda _, v: isinstance(v, dict) and "regrets" in v
    ):
        conf = e["conf"]
        mean = e["mean_regret"]
        std = e["std_regret"]

        plt.plot(
            range(len(mean)),
            mean,
            label=" ".join(
                [
                    v
                    for k, v in conf.items()
                    if "experiment-parameter" in CONFIG[k]["tags"]
                ]
            ),
            linestyle="--",
        )

        plt.fill_between(
            range(len(mean)),
            mean - std,
            mean + std,
            alpha=0.2,
        )  # careful with the std bands when plotting in log scale (not symmetric)
        plt.yscale("log")

    plt.xlabel("Budget (Iterations)")
    plt.ylabel("Simple regret")
    fig.legend(shadow=True)
    fig.set_tight_layout({"pad": 0})

    plt.show()


def visualize_trajectory(file: str, *, plot_user_model=True) -> None:
    """Visualizes the end result (approximation, sample data, etc)

    :file: file path (string)
    :returns: None
    """

    # Pre-compute some constants
    candidate_test_functions = [
        f for f, c in CONFIG["function"]["choices"].items() if c["dims"] == 1
    ]

    # Load the file.
    new_results = torch.load(file)
    conf = new_results["conf"]

    if conf["function"] not in candidate_test_functions:
        raise ValueError(
            f"{file} is not an 1-dimensional experiment (in {candidate_test_functions}) and thus is excluded"
        )

    # Load configurations and results.
    budget, n_init = conf["budget"], conf["n_init"]
    problem = pick_test_function(conf["function"])

    x_min, x_max = problem._bounds[0]
    x_linspace = torch.linspace(x_min, x_max, 101).reshape(-1, 1)
    y_truth = problem(x_linspace)
    queries, observations = new_results["train_X"], new_results["train_Y"]

    # Get "global" (across all time steps) values.
    optimal_xs = CONFIG["function"]["choices"][conf["function"]]["optimal_x"]
    user_models = [
        pick_user_model(conf["user_model"], optimal_x, problem)(x_linspace, y_truth)
        for optimal_x in optimal_xs
    ]

    # Process results for each "time step"
    results = []
    for b in range(budget + 1):
        x, y = queries[: n_init + b], observations[: n_init + b]

        with settings.validate_input_scaling(False):
            # We do not want to scale the GPR predictions to align them with real data points
            gpr = SingleTaskGP(x, y, covar_module=pick_kernel(conf["kernel"], 1))

        mll = ExactMarginalLogLikelihood(gpr.likelihood, gpr)
        fit_gpytorch_model(mll)

        gpr_post_mean = gpr.likelihood(gpr(x_linspace)).mean.detach().numpy()
        gpr_post_var = np.sqrt(gpr.likelihood(gpr(x_linspace)).variance.detach().numpy())

        acqf = pick_acqf(conf["acqf"], y, gpr, torch.tensor(problem._bounds).T)
        acqf_eval = acqf(x_linspace[:, None, :]).detach().numpy()

        results.append(
            {
                "gpr_post_mean": gpr_post_mean,
                "gpr_post_var": gpr_post_var,
                "x": x,
                "y": y,
                "acqf": acqf_eval,
            }
        )

    # Create our plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca()

    def draw_results(b: int):
        """Actually draws the results in our figure"""
        ax.clear()

        # Grab results for time step `b`
        r = results[b]
        m, var, x, y, acqf = r["gpr_post_mean"], r["gpr_post_var"], r["x"], r["y"], r["acqf"]

        # Plot global
        if plot_user_model:
            for user_model in user_models:
                __import__('pdb').set_trace()
                ax.plot(x_linspace, user_model, "g", label="User model")

        ax.plot(x_linspace, y_truth, label="Ground Truth")

        # Plot results
        ax.plot(x_linspace, m, "b", label="GP Mean function")
        ax.fill_between(
            x_linspace.squeeze(),
            m - var,
            m + var,
            alpha=0.2,
            color="b",
        )
        ax.scatter(x, y, alpha=0.5, color="black", marker="x", s=100)

        if b < budget:
            ax.scatter(
                queries[n_init + b], observations[n_init + b], color="r", label="Next"
            )

        ax.plot(x_linspace, acqf, linestyle="dotted", color="orange", linewidth=3, label=conf["acqf"])

        # Basic plotting style
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(shadow=True)
        ax.set_title(
            " ".join(
                [
                    v
                    for k, v in conf.items()
                    if "experiment-parameter" in CONFIG[k]["tags"]
                ]
                + [str(conf["seed"])]
            )
        )

        fig.canvas.draw_idle()

    fig.subplots_adjust(bottom=0.2)
    ax_budget = fig.add_axes([0.2, 0.05, 0.6, 0.03])

    slider_budget = Slider(
        ax_budget,
        "b",
        0,
        budget,
        valinit=budget,
        valstep=[b for b in range(budget + 1)],
    )

    slider_budget.on_changed(draw_results)

    draw_results(budget)
    plt.show()


if __name__ == "__main__":
    warnings.showwarning = utils.warn_with_traceback
    parser = argparse.ArgumentParser(description="Command description.")
    parser.add_argument(
        "-t",
        "--type",
        default="regrets",
        type=str,
        help="Type of visualization",
        choices=["regrets", "trajectory"],
    )
    parser.add_argument(
        "-f",
        "--files",
        nargs="*",
        type=str,
        help="All files that need to be processed",
    )
    parser.add_argument(
        "--plot-user-model",
        type=bool,
        help="Whether to display user model when plotting type=trajectory",
        default=True,
        action=argparse.BooleanOptionalAction
    )

    args = parser.parse_args()

    # Basic setup for all visualizations.
    torch.set_default_dtype(torch.double)
    set_matplotlib_params()

    if args.type == "regrets":
        compare_regrets_over_time(args.files)

    if args.type == "trajectory":
        if len(args.files) != 1:
            raise ValueError("Please only provide 1 file when plotting trajectory")

        visualize_trajectory(args.files[0], plot_user_model=args.plot_user_model)
