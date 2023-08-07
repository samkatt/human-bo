import matplotlib as mpl


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
