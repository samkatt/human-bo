#!/usr/bin/env python

"""Simple script to visualize the results in a folder generated by the main script"""

import argparse
import math
import warnings
from typing import Any

import matplotlib.pyplot as plt
import torch
from matplotlib.widgets import Slider

from human_bo import conf, core, test_functions, utils, visualization
from human_bo.moo import moo_core


def get_init_points(res):
    x, y = torch.Tensor(), torch.Tensor()

    if "initial_points" in res:
        x = torch.cat((x, res["initial_points"]["x"]))
        y = torch.cat((y, res["initial_points"]["y"]))

    if "ai_conf" in res and "initial_points" in res["ai_conf"]:
        x = torch.cat((x, res["ai_conf"]["initial_points"]["x"]))
        y = torch.cat((y, res["ai_conf"]["initial_points"]["y"]))

    if "user_conf" in res and "initial_points" in res["user_conf"]:
        x = torch.cat((x, res["user_conf"]["initial_points"]["x"]))
        y = torch.cat((y, res["user_conf"]["initial_points"]["y"]))

    return x, y


def compare_ymax_over_time(files: list[str]) -> None:
    """Visualizes (plots) the regrets in of `files`

    Asserts that the results being compared have the same hyper parameter.

    :files: list of file paths (string)
    """
    experiment_config = torch.load(files[0], weights_only=True)["conf"]

    results: dict[str, Any] = {}
    for file in files:
        # Load the file and initiated configurations and results.
        new_results = torch.load(file, weights_only=True)
        new_conf = new_results["conf"]
        y_max = torch.Tensor([x["y_max"] for x in new_results["evaluation_stats"]])

        # Make sure experiment shares the same parameters.
        for k, c in conf.CONFIG.items():
            if "experiment-hyper-parameter" not in c["tags"]:
                continue  # We only check for "experiment-hyper-parameter" configurations

            assert (
                experiment_config[k] == new_conf[k]
            ), f"Different {k} parameters used: {experiment_config[k]} and {new_conf[k]}"

        # Store results.
        current_results_for_experiment = results
        for k, c in conf.CONFIG.items():
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
            current_results_for_experiment["y_max"] = torch.cat(
                (current_results_for_experiment["y_max"], y_max.unsqueeze(-1)),
                dim=1,
            )
        else:
            # First time looking at this particular experimental setup!
            current_results_for_experiment["conf"] = new_conf
            current_results_for_experiment["y_max"] = y_max.unsqueeze(-1)

    # Go over all results and compute (and store) their mean and standard error.
    for e in utils.recursively_filter_dict(
        results, lambda _, v: isinstance(v, dict) and "y_max" in v
    ):
        r = e["y_max"]
        e["mean_y_max"] = r.mean(axis=1)
        e["std_y_max"] = 1.96 * r.std(axis=1) / math.sqrt(r.shape[1])

    fig = plt.figure(figsize=(8, 6))

    for e in utils.recursively_filter_dict(
        results, lambda _, v: isinstance(v, dict) and "y_max" in v
    ):
        exp_params = e["conf"]
        mean = e["mean_y_max"]
        std = e["std_y_max"]

        plt.plot(
            range(len(mean)),
            mean,
            label=" ".join(
                conf.get_values_with_tag(exp_params, "experiment-parameter")
            ),
            ls="--",
        )

        plt.fill_between(
            range(len(mean)),
            mean - std,
            mean + std,
            alpha=0.2,
        )

        # Careful with the std bands when plotting in log scale (not symmetric)
        # plt.yscale("log")

    plt.xlabel("Budget (Iterations)")
    plt.ylabel("max y")
    fig.legend(shadow=True)
    fig.tight_layout(pad=0)

    plt.show()


def visualize_trajectory_1D(results) -> None:
    """Visualizes the end result (approximation, sample data, etc)

    :file: file path (string)
    :returns: None
    """

    # Load configurations and results.
    exp_params = results["conf"]
    problem = test_functions.pick_test_function(exp_params["problem"], noise=0.0)

    bounds = torch.tensor(problem._bounds).T
    x_min, x_max = bounds.squeeze().tolist()

    x_linspace = torch.linspace(x_min, x_max, 101).reshape(-1, 1)
    y_truth = problem(x_linspace)

    x_init, y_init = get_init_points(results)
    queries, observations = torch.cat(results["query"]), torch.cat(results["feedback"])

    n = len(observations)

    # Process results for each "time step"
    results = []
    for b in range(-1, n):
        x = torch.cat((x_init, queries[: b + 1]))
        y = torch.cat((y_init, observations[: b + 1]))

        if len(x) == 0:

            # Little hack: very rarely we start experiments with no initial points.
            # In this case, we _cannot_ compute any of the things we want to do below.
            # So we just return zero for all of them.
            results.append(
                {
                    "gpr_post_mean": torch.zeros(len(x_linspace)),
                    "gpr_post_var": torch.zeros(len(x_linspace)),
                    "queries": queries[: b + 1],
                    "x_init": x_init,
                    "observations": observations[: b + 1],
                    "y_init": y_init,
                    "acqf": torch.zeros(len(x_linspace)),
                }
            )

            continue

        gpr = core.fit_gp(
            x, y, core.pick_kernel(exp_params["kernel"], 1), input_bounds=bounds
        )

        posteriors = gpr.posterior(x_linspace)
        gpr_post_mean = posteriors.mean.squeeze().detach().numpy()
        gpr_post_var = posteriors.variance.squeeze().detach().numpy()

        acqf = core.create_acqf(
            exp_params["acqf"],
            x,
            gpr,
            bounds,
        )
        acqf_eval = acqf(x_linspace[:, None, :]).detach().numpy()

        results.append(
            {
                "gpr_post_mean": gpr_post_mean,
                "gpr_post_var": gpr_post_var,
                "queries": queries[: b + 1],
                "x_init": x_init,
                "observations": observations[: b + 1],
                "y_init": y_init,
                "acqf": acqf_eval,
            }
        )

    # Create our plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca()

    def draw_results(slider_input: float) -> int:
        """Actually draws the results in our figure"""
        ax.clear()

        b = int(slider_input)

        # Grab results for time step `b`
        r = results[b]
        m, var, queries_at_b, x_init, observations_at_b, y_init, acqf = (
            r["gpr_post_mean"],
            r["gpr_post_var"],
            r["queries"],
            r["x_init"],
            r["observations"],
            r["y_init"],
            r["acqf"],
        )

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
        ax.scatter(
            x_init,
            y_init,
            alpha=0.5,
            color="black",
            marker="x",
            s=100,
            label="init points",
        )
        ax.scatter(
            queries_at_b,
            observations_at_b,
            alpha=0.5,
            color="green",
            marker="x",
            s=100,
            label="observations",
        )

        if b < n:
            ax.scatter(queries[b], observations[b], color="r", label="Next")

        ax.plot(
            x_linspace,
            acqf,
            ls="dotted",
            color="orange",
            linewidth=3,
            label=exp_params["acqf"],
        )

        # Basic plotting style
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(shadow=True)
        ax.set_title(
            "_".join(
                conf.get_values_with_tag(exp_params, "experiment-parameter")
                + [str(exp_params["seed"])]
            )
        )

        fig.canvas.draw_idle()

        return 0

    fig.subplots_adjust(bottom=0.2)
    ax_budget = fig.add_axes((0.2, 0.05, 0.6, 0.03))

    slider_budget = Slider(
        ax_budget,
        "b",
        0,
        n,
        valinit=n,
        valstep=list(range(n + 1)),
    )

    slider_budget.on_changed(draw_results)

    draw_results(n)
    plt.show()


def visualize_trajectory_2D(result_file_content) -> None:
    """Visualizes the end result (approximation, sample data, etc)

    :file: file path (string)
    :returns: None
    """

    # Load configurations and results.
    exp_params = result_file_content["conf"]
    problem = test_functions.pick_test_function(exp_params["problem"], noise=0.0)

    # Pre-compute global variables.
    bounds = problem.bounds
    x1_min, x1_max, x2_min, x2_max = bounds.T.flatten()
    x1 = torch.linspace(x1_min, x1_max, 100)
    x2 = torch.linspace(x2_min, x2_max, 100)
    X1, X2 = torch.meshgrid(x1, x2, indexing="xy")
    X = torch.stack((X1, X2), dim=2)
    Y = problem(torch.stack((X1, X2), dim=2))

    x_init, y_init = get_init_points(result_file_content)
    queries, observations = torch.cat(result_file_content["query"]), torch.cat(
        result_file_content["feedback"]
    )

    n = len(observations)

    # Generate results for each step.
    results = []
    for b in range(-1, n):

        x = torch.cat((x_init, queries[: b + 1]))
        y = torch.cat((y_init, observations[: b + 1]))

        if len(x) == 0:
            # Weird corner case: there is nothing to plot.
            results.append(
                {
                    "gpr_post_mean": torch.zeros_like(Y),
                    "gpr_post_mean_dist": torch.zeros_like(Y),
                    "gpr_post_var": torch.zeros_like(Y),
                    "queries": queries[: b + 1],
                    "x_init": x_init,
                    "observations": observations[: b + 1],
                    "y_init": y_init,
                    "acqf": torch.zeros_like(Y),
                }
            )

            continue

        gpr = core.fit_gp(
            x, y, core.pick_kernel(exp_params["kernel"], 1), input_bounds=bounds
        )

        posteriors = gpr.posterior(X)
        gpr_post_mean = posteriors.mean.squeeze()
        gpr_post_var = posteriors.variance.squeeze()

        acqf = core.create_acqf(
            exp_params["acqf"],
            x,
            gpr,
            bounds,
        )
        acqf_eval = acqf(X.reshape(-1, 2)[:, None, :]).reshape(gpr_post_mean.shape)

        results.append(
            {
                "gpr_post_mean": gpr_post_mean.detach().numpy(),
                "gpr_post_mean_dist": (gpr_post_mean - Y).detach().numpy(),
                "gpr_post_var": gpr_post_var.detach().numpy(),
                "queries": queries[: b + 1],
                "x_init": x_init,
                "observations": observations[: b + 1],
                "y_init": y_init,
                "acqf": acqf_eval.detach().numpy(),
            }
        )

    # Setup figures
    fig = plt.figure(figsize=(10, 8))
    surface_kwargs = {"rcount": 3, "ccount": 3, "lw": 0.5, "alpha": 0.3}
    contour_vals = {
        "var": "gpr_post_var",
        "mean_dist": "gpr_post_mean_dist",
        "acqf": "acqf",
    }
    axs = {
        "var": fig.add_subplot(2, 2, 1),
        "mean_dist": fig.add_subplot(2, 2, 2),
        "acqf": fig.add_subplot(2, 2, 3),
        "ax_3d": fig.add_subplot(2, 2, 4, projection="3d"),
    }
    contours = {
        k: axs[k].contourf(x1, x2, results[-1][v], cmap="cividis")
        for k, v in contour_vals.items()
    }
    cbars = {k: plt.colorbar(contours[k]) for k in contour_vals}

    for k, ax in axs.items():
        ax.set_title(k)

    def draw_results(slider_input: float) -> int:
        """Draw results for budget=slider_input"""
        b = int(slider_input)

        r = results[b]
        x_init, y_init = r["x_init"], r["y_init"]
        x, y = r["queries"], r["observations"]

        for k, r_key in contour_vals.items():
            axs[k].contourf(x1, x2, r[r_key], cmap="cividis")
            cbars[k].set_ticklabels(
                [
                    f"{number:.2f}"
                    for number in torch.linspace(
                        r[r_key].min(), r[r_key].max(), len(cbars[k].get_ticks())
                    ).tolist()
                ]
            )

            axs[k].scatter(x[:, 0], x[:, 1], color="green", label="observations")
            axs[k].scatter(
                x_init[:, 0], x_init[:, 1], color="purple", label="initial points"
            )

        axs["ax_3d"].clear()

        axs["ax_3d"].plot_surface(  # type: ignore
            X1.numpy(),
            X2.numpy(),
            Y.numpy(),
            color="green",
            edgecolor="green",
            label="f(x)",
            **surface_kwargs,
        )

        axs["ax_3d"].plot_surface(  # type: ignore
            X1.numpy(),
            X2.numpy(),
            r["gpr_post_mean"],
            color="blue",
            edgecolor="blue",
            label="GP mean",
            **surface_kwargs,
        )
        axs["ax_3d"].scatter(x[:, 0], x[:, 1], y, color="black", label="observations")
        axs["ax_3d"].scatter(
            x_init[:, 0], x_init[:, 1], y_init, color="blue", label="initial points"
        )

        if b < n:
            [next_x_1, next_x_2], next_y = queries[b], observations[b]

            for k in contour_vals:
                axs[k].scatter(next_x_1, next_x_2, color="orange")

            axs["ax_3d"].scatter(
                next_x_1, next_x_2, next_y, color="orange", label="next"
            )

        axs["ax_3d"].legend()
        fig.canvas.draw_idle()

        return 0

    fig.subplots_adjust(bottom=0.2)
    ax_budget = fig.add_axes((0.2, 0.05, 0.6, 0.03))

    slider_budget = Slider(
        ax_budget,
        "b",
        0,
        n,
        valinit=n,
        valstep=list(range(n + 1)),
    )

    slider_budget.on_changed(draw_results)

    draw_results(n)
    plt.show()


def visualize_moo(results):
    # Re-create problem and its dimensions.
    exp_params = results["conf"]
    problem = test_functions.pick_moo_test_function(exp_params["problem"], noise=None)
    utility_function = moo_core.create_utility_function(
        exp_params["preference_weights"]
    )

    dim = problem.dim
    num_objs = problem.num_objectives

    x_lims = [(x[0], x[1]) for x in problem._bounds]
    x_linspaces = [torch.linspace(x_min, x_max, 100) for x_min, x_max in x_lims]
    X_mesh = torch.meshgrid(*x_linspaces, indexing="xy")
    O_x = problem(torch.stack(X_mesh, dim=2))
    o_lims = [
        (O_x[..., o].min().item(), O_x[..., o].max().item()) for o in range(num_objs)
    ]
    o_linspaces = [torch.linspace(o_min, o_max, 100) for o_min, o_max in o_lims]
    O_mesh = torch.meshgrid(*o_linspaces, indexing="xy")

    # Get data from file.
    queries = torch.cat(results["query"])
    utilities = torch.cat([r["utility"] for r in results["feedback"]])
    objectives = torch.cat(
        [
            torch.tensor([list(q.values()) for q in r["objectives"].values()])
            for r in results["feedback"]
        ]
    )

    # TODO: plot true utilities as line and plot observations as points.

    assert len(queries) == len(utilities) == len(objectives)

    # Populate data for each time step.
    n = len(queries)

    results = [
        {
            "x": queries[:t],
            "u": utilities[:t],
            "o": objectives[:t],
            "next_x": queries[t] if t < n else None,
            "next_u": utilities[t] if t < n else None,
            "next_o": objectives[t] if t < n else None,
        }
        for t in range(0, n + 1)
    ]

    # Set up figures.
    fig = plt.figure()
    ax_u = fig.add_subplot(221)
    ax_o = fig.add_subplot(222) if num_objs == 2 else None
    ax_x = fig.add_subplot(223) if dim == 2 else None

    # Utility plot.
    ax_u.plot(utilities, label="Utility", color="black")
    (lines_u_next,) = ax_u.plot(utilities[-1], n, "ro", label="Next")
    ax_u.set_xlim(0, n)
    ax_u.set_xlabel("budget")
    ax_u.set_ylabel("u(x)")
    ax_u.set_title("Utility")
    ax_u.legend()

    # Objectives plot.
    if ax_o:
        U_o = utility_function(torch.stack(O_mesh, dim=2))
        contourf_o = ax_o.contourf(*o_linspaces, U_o, cmap="cividis")
        plt.colorbar(contourf_o, ax=ax_o)

        (scatter_observed_o,) = ax_o.plot(
            objectives[:, 0],
            objectives[:, 1],
            "ko",
            label="Observations",
        )
        (scatter_next_o,) = ax_o.plot(
            objectives[-1, 0], objectives[-1, 1], "ro", label="Next"
        )

        ax_o.set_xlabel("o1")
        ax_o.set_ylabel("o2")
        ax_o.set_title("Objectives")
        ax_o.legend()
    else:
        scatter_observed_o, scatter_next_o = None, None

    # Query plot.
    if ax_x:
        U_x = utility_function(problem(torch.stack(X_mesh, dim=2)))
        contourf_x = ax_x.contourf(*x_linspaces, U_x, cmap="cividis")
        plt.colorbar(contourf_x, ax=ax_x)

        (scattered_x,) = ax_x.plot(queries[:, 0], queries[:, 1], "ko")

        (scattered_next_x,) = ax_x.plot(
            queries[-1, 0], queries[-1, 1], "ro", label="Next"
        )
        ax_x.set_xlabel("x1")
        ax_x.set_ylabel("x2")
        ax_x.legend()
    else:
        scattered_x = None
        scattered_next_x = None

    def draw_results(slider_input: float) -> int:
        """Populate figure for time step `slider_input`"""
        b = int(slider_input)

        r = results[b]

        x, o = r["x"], r["o"]
        next_x, next_u, next_o = r["next_x"], r["next_u"], r["next_o"]

        # Update observations.
        if scatter_observed_o:
            scatter_observed_o.set_data(o[:, 0], o[:, 1])
        if scattered_x:
            scattered_x.set_data(x[:, 0], x[:, 1])

        # Set "next" data.
        if b < n:
            lines_u_next.set_data([b], [next_u.item()])
            if scatter_next_o:
                scatter_next_o.set_data([next_o[0]], [next_o[1]])
            if scattered_next_x:
                scattered_next_x.set_data([next_x[0]], [next_x[1]])
        else:
            lines_u_next.set_data([], [])
            if scatter_next_o:
                scatter_next_o.set_data([], [])
            if scattered_next_x:
                scattered_next_x.set_data([], [])

        return 0

    fig.subplots_adjust(bottom=0.1)
    ax_budget = fig.add_axes((0.2, 0.025, 0.6, 0.03))
    slider_budget = Slider(ax_budget, "b", 0, n, valinit=n, valstep=range(n + 1))
    slider_budget.on_changed(draw_results)

    draw_results(n)
    plt.show()


if __name__ == "__main__":
    warnings.showwarning = utils.warn_with_traceback

    parser = argparse.ArgumentParser(description="Command description.")
    parser.add_argument(
        "-t",
        "--type",
        default="regrets",
        type=str,
        help="Type of visualization.",
        choices=["y_max", "trajectory"],
    )
    parser.add_argument(
        "-f",
        "--files",
        nargs="*",
        type=str,
        help="All files that need to be processed.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        help="For which budget to plot",
        default=True,
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()

    # Basic setup for all visualizations.
    torch.set_default_dtype(torch.double)
    visualization.set_matplotlib_params()

    if args.type == "y_max":
        compare_ymax_over_time(args.files)

    if args.type == "trajectory":
        if len(args.files) != 1:
            raise ValueError("Please only provide 1 file when plotting trajectory")

        file_content = torch.load(args.files[0], weights_only=True)

        x_dim = conf.CONFIG["problem"]["parser-arguments"]["choices"][
            file_content["conf"]["problem"]
        ]["dims"]

        is_moo = (
            "num_objectives"
            in conf.CONFIG["problem"]["parser-arguments"]["choices"][
                file_content["conf"]["problem"]
            ]
        )

        if is_moo:
            visualize_moo(file_content)
        elif x_dim == 1:
            visualize_trajectory_1D(file_content)
        elif x_dim == 2:
            visualize_trajectory_2D(file_content)
        else:
            raise ValueError(
                f"Experiment on {file_content['conf']['problem']} is too high-dimensional ({x_dim}) to visualize"
            )
