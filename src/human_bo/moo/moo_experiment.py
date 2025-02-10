"""Main functionality for multi-objective optimization."""

from botorch.test_functions import multi_objective as moo_test_functions
from botorch.utils import sampling as botorch_sampling

def run_moo():
    """Basic, very simple multi-objective optimization.

    Inspired by https://botorch.org/docs/tutorials/multi_objective_bo/,
    this is mostly a tutorial for typical MOO: finding pareto front.
    """
    noise_std = [0.2, 0.4]
    test_func = moo_test_functions.BraninCurrin
    n_init = 6


    f = test_func(noise_std)
    train_x = botorch_sampling.draw_sobol_samples(bounds=f.bounds, n=n_init, q=1).squeeze(1)

