"""Testing and example code for multi-objective optimization."""

from botorch.acquisition import objective
from botorch.test_functions import multi_objective as moo_test_functions

from human_bo.moo import moo_core
from human_bo import core, interaction_loops, reporting


def moo_learn_subjective_function():
    """Experiment on multi-objective optimization where the only unknown is one (unmeasurable) objective function."""
    # Parameters.
    noise_std = [0.2, 0.4]
    test_function = moo_test_functions.BraninCurrin
    budget = 20
    # unknown_output_idx = 0

    # n_init = 6
    # num_restarts_acqf = 8
    # raw_samples_acqf = 128

    # Setup problem.
    moo_function = test_function(noise_std)

    preference_weights = core.random_queries(
        [(0, 1) for _ in range(moo_function.dim)]
    ).squeeze()
    preference_weights = preference_weights / preference_weights.sum()

    utility_function = objective.LinearMCObjective(preference_weights)

    # Setup interaction components.
    agent = core.RandomAgent(moo_function.bounds)
    problem = moo_core.MOOProblem(moo_function, utility_function)
    evaluation = moo_core.MOOEvaluation(
        moo_function, utility_function, reporting.print_dot
    )

    print("Starting subjective MOO loop...")
    res = interaction_loops.basic_loop(agent, problem, evaluation, budget)
    print("Ended subjective MOO loop!")

    return res
