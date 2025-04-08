"""Code for integration with Trieste."""

from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import trieste

from human_bo import interaction_loops, moo, test_functions, utils


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
    x_dim: int | None = None,
    o_dim: int | None = None,
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
    if func == "InverseBranin":
        branin = trieste.objectives.single_objectives.Branin
        return trieste.objectives.single_objectives.SingleObjectiveTestProblem(
            name="InverseBranin",
            objective=lambda x: -branin.objective(x),
            search_space=branin.search_space,
            minimizers=None,  # XXX: I do not know these.
            minimum=None,  # XXX: I do not know these.
        )

    # It is MOO from here on out!
    if func == "DTLZ2":
        assert x_dim is not None and x_dim > 0
        assert o_dim is not None and o_dim > 0
        return trieste.objectives.multi_objectives.DTLZ2(x_dim, o_dim)

    if func == "VLMOP2":
        assert x_dim is not None and x_dim > 0
        return trieste.objectives.multi_objectives.VLMOP2(x_dim)

    if func == "BraninCurrin":
        search_space = trieste.space.Box([0.0], [1.0]) ** 2

        def obj(x):
            return tf.concat(
                (
                    trieste.objectives.single_objectives.branin(x),
                    test_functions.currin(x, tf.pow, tf.exp),
                ),
                axis=-1,
            )

        return trieste.objectives.multi_objectives.MultiObjectiveTestProblem(
            name="BraninCurrin",
            objective=obj,
            search_space=search_space,
            gen_pareto_optimal_points=lambda n, seed=None: tf.stack(
                generate_pareto_optimal_points(n, obj, search_space), axis=-1
            ),
        )
    if func == "InverseBraninCurrin":
        search_space = trieste.space.Box([0.0], [1.0]) ** 2

        def obj(x):
            return tf.concat(
                (
                    -trieste.objectives.single_objectives.branin(x),
                    -test_functions.currin(x, tf.pow, tf.exp),
                ),
                axis=-1,
            )

        return trieste.objectives.multi_objectives.MultiObjectiveTestProblem(
            name="BraninCurrin",
            objective=obj,
            search_space=search_space,
            gen_pareto_optimal_points=lambda n, seed=None: tf.stack(
                generate_pareto_optimal_points(n, obj, search_space), axis=-1
            ),
        )

    raise ValueError(f"{func} is not an accepted Trieste test function")


def create_trieste_observer(
    f, noise_stdev: list[float] | None
) -> trieste.observer.Observer:
    """Makes `f` noisey (with deviation `noise_stdev`) and a Trieste observer out of it.

    If `noise_stdev` is `None`, this will return a noiseless problem `f`.
    """

    if noise_stdev is None:
        print("WARN:creating observer without noise - your problem has no noise.")
        return trieste.objectives.utils.mk_observer(f)

    mvn = tfp.distributions.MultivariateNormalDiag(
        scale_diag=tf.convert_to_tensor(noise_stdev, tf.float64)
    )

    def noisey_f(X):
        y = f(X)
        noise = mvn.sample(len(y))

        assert y.shape == noise.shape
        return y + noise

    return trieste.objectives.utils.mk_observer(noisey_f)


def generate_pareto_optimal_points(n: int, objective, space: trieste.space.SearchSpace):
    def gen():
        x: trieste.types.TensorType = space.sample(1)
        y: trieste.types.TensorType = objective(x)

        return (x, y)

    def comp(
        d1: tuple[trieste.types.TensorType, trieste.types.TensorType],
        d2: tuple[trieste.types.TensorType, trieste.types.TensorType],
    ) -> int:
        y1, y2 = d1[1], d2[1]
        if tf.reduce_all(tf.greater(y1, y2)):
            return 1
        if tf.reduce_all(tf.less(y1, y2)):
            return -1

        return 0

    pareto_points = moo.generate_front(n, gen, comp)

    return [d[0] for d in pareto_points]


class RandomAgent(interaction_loops.Agent):

    def __init__(self, search_space: trieste.space.SearchSpace):
        self.search_space = search_space

    def pick_query(self) -> tuple[Any, dict[str, Any]]:
        return self.search_space.sample(1), {}

    def observe(self, query, feedback, evaluation) -> None:
        del query, feedback, evaluation


class TriesteBO(interaction_loops.Agent):

    def __init__(
        self,
        data: trieste.data.Dataset,
        search_space: trieste.space.SearchSpace,
        acqf: str,
        acqf_options: dict[str, Any],
    ):
        self.data = data
        self.search_space = search_space
        self.step = -1
        self.acqf = create_trieste_acqf(acqf, self.search_space, acqf_options)
        self.mean_acqf = create_trieste_acqf("mean", self.search_space, {})

    def pick_query(self) -> tuple[Any, dict[str, Any]]:
        self.step += 1

        # Create the model (or return random sample if fails).
        try:
            y_sca, y_mean, y_std = utils.normalize(self.data.observations)
            data_sca = trieste.data.Dataset(self.data.query_points, y_sca)
            model = create_trieste_gp(data_sca, self.search_space)

        except tf.errors.InvalidArgumentError:
            print(
                "WARN: `TriesteBO.pick_query` failed to fit model, returning random sample."
            )
            return self.search_space.sample(1), {}

        # Pick query given model.
        query = optimize_trieste_acqf(self.acqf, data_sca, model, self.search_space)

        arg_map = optimize_trieste_acqf(
            self.mean_acqf, data_sca, model, self.search_space
        )
        # Un-normalize predicted MAP.
        map_mean = model.predict(arg_map)[0] * y_std + y_mean

        return query, {"map": {"x": np.array(arg_map), "y": np.array(map_mean)}}

    def observe(self, query, feedback, evaluation) -> None:
        del query, evaluation
        self.data = self.data + feedback


def compute_utility(objectives: tf.Tensor, preference_weights: tf.Tensor) -> tf.Tensor:
    """Calculates (linear) utility of `objectives` given `preference_weights`.

    In practice, returns matrix multiplication `objectives * preference_weights`.

    Will cast `objectives` into [..., o_dim] to do the multiplication.
    """
    assert preference_weights.ndim is not None and preference_weights.ndim <= 2
    assert objectives.ndim == 2 and objectives.shape[-1] == preference_weights.shape[0]

    return tf.matmul(objectives, tf.reshape(preference_weights, (-1, 1)))
