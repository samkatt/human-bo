"""Code for integration with Trieste."""

from typing import Any

import tensorflow as tf
import trieste

from human_bo import interaction_loops, test_functions


def create_trieste_acqf(
    acqf: str,
    search_space: trieste.space.SearchSpace,
    acqf_options: dict[str, Any],
) -> trieste.acquisition.interface.SingleModelAcquisitionBuilder:
    """Creates an acquisition "rule" for Trieste to give to optimizers.

    Note: the result is *state-full*, so please make sure you re-create this every time you optimize.
    """
    if acqf == "EI":
        return trieste.acquisition.function.function.AugmentedExpectedImprovement()
    if acqf == "UCB":
        return trieste.acquisition.function.function.NegativeLowerConfidenceBound(
            acqf_options["ucb_beta"]
        )
    if acqf == "MES":
        return trieste.acquisition.function.entropy.MinValueEntropySearch(search_space)
    if acqf == "mean":
        return trieste.acquisition.function.function.NegativePredictiveMean()

    raise ValueError(f"{acqf} is not an accepted acquisition function")


def optimize_trieste_acqf(
    trieste_acqf: trieste.acquisition.interface.SingleModelAcquisitionBuilder,
    dataset: trieste.data.Dataset,
    trieste_model: trieste.models.interfaces.ProbabilisticModel,
    search_space: trieste.space.SearchSpace,
):
    """A very short helper function to get from an acquisition function builder to query.

    In practice, it really just creates an acquisition Trieste "rule" based on EGO and
    optimize `trieste_acqf` based on the rest of the input.

    Does no additional fancy stuff such as normalization or otherwise.

    See `create_trieste_acqf` for how to create `trieste_acqf`.
    """
    trieste_rule: trieste.acquisition.rule.AcquisitionRule = (
        trieste.acquisition.rule.EfficientGlobalOptimization(trieste_acqf)
    )
    return trieste_rule.acquire_single(search_space, trieste_model, dataset)


def create_trieste_gp(
    data: trieste.data.Dataset, search_space: trieste.space.SearchSpace
):
    """Factory function for creating (Trieste) posterior models.

    Note: will call `optimize` on the model before returning.
    """
    gp = trieste.models.gpflow.models.GaussianProcessRegression(
        trieste.models.gpflow.builders.build_gpr(
            data, search_space, trainable_likelihood=True
        )
    )
    gp.optimize(data)

    return gp


def create_trieste_test_function(
    func: str,
) -> trieste.objectives.single_objectives.ObjectiveTestProblem:
    if func == "Levy1D":
        return trieste.objectives.single_objectives.SingleObjectiveTestProblem(
            name="Levy 1",
            objective=lambda x: trieste.objectives.single_objectives.levy(x, 1),
            search_space=trieste.space.Box([0.0], [1.0]),
            minimizers=tf.convert_to_tensor([[11 / 20]]),
            minimum=tf.convert_to_tensor([0]),
        )
    if func == "Zhou":
        return trieste.objectives.single_objectives.SingleObjectiveTestProblem(
            name="Zhou",
            objective=lambda x: tf.reshape(
                -test_functions.zhou(x, tf.experimental.numpy.pi, tf.exp), [-1, 1]
            ),
            search_space=trieste.space.Box([0.0], [1.0]),
            minimizers=tf.convert_to_tensor([[1 / 3], [2 / 3]]),
            minimum=tf.convert_to_tensor([-2.002595246981888]),
        )
    if func == "Forrester":
        return trieste.objectives.single_objectives.SingleObjectiveTestProblem(
            name="Forrester",
            objective=lambda x: tf.reshape(
                -test_functions.forrester(x, tf.sin), [-1, 1]
            ),
            search_space=trieste.space.Box([0.0], [1.0]),
            minimizers=tf.convert_to_tensor([[0.7572]]),
            minimum=tf.convert_to_tensor([-6.020738786441099]),
        )
    if func == "Branin":
        return trieste.objectives.single_objectives.Branin

    raise ValueError(f"{func} is not an accepted Trieste test function")


def create_trieste_observer(f, noise_stdev) -> trieste.observer.Observer:
    """Makes `f` noisey (with stdev `noise`) and make a Trieste observer out of it."""

    def noisey_f(X):
        y = f(X)
        noise = tf.random.normal(y.shape, stddev=noise_stdev, dtype=y.dtype)
        return y + noise

    return trieste.objectives.utils.mk_observer(noisey_f)


class RandomAgent(interaction_loops.Agent):

    def __init__(self, search_space: trieste.space.SearchSpace):
        self.search_space = search_space

    def pick_query(self) -> tuple[Any, dict[str, Any]]:
        return self.search_space.sample(1), {}

    def observe(self, query, feedback, evaluation) -> None:
        del query, feedback, evaluation
