"""Code for integration with Trieste."""

from collections.abc import Callable
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import trieste

from human_bo import interaction_loops, moo, posteriors, test_functions, utils


def create_acqf(
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
        assert "ucb_beta" in acqf_options, "UCB expects a `ucb_value`."
        assert isinstance(acqf_options["ucb_beta"], float), "`ucb_value` must be float."
        assert acqf_options["ucb_beta"] >= 0, "`ucb_value` must be positive."

        return trieste.acquisition.function.function.NegativeLowerConfidenceBound(
            acqf_options["ucb_beta"]
        )
    if acqf == "MES":
        return trieste.acquisition.function.entropy.MinValueEntropySearch(search_space)
    if acqf == "mean":
        return trieste.acquisition.function.function.NegativePredictiveMean()

    raise ValueError(f"{acqf} is not an accepted acquisition function")


def optimize_acqf(
    trieste_acqf: trieste.acquisition.interface.SingleModelAcquisitionBuilder,
    dataset: trieste.data.Dataset,
    trieste_model: trieste.models.interfaces.ProbabilisticModel,
    search_space: trieste.space.SearchSpace,
) -> trieste.types.TensorType:
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


def create_test_function(
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
    if func == "Currin":
        return trieste.objectives.single_objectives.SingleObjectiveTestProblem(
            name="Currin",
            objective=lambda x: test_functions.currin(x, tf.pow, tf.exp),
            search_space=trieste.space.Box([0.0], [1.0]) ** 2,
            minimizers=tf.convert_to_tensor([[0.0, 1.0]]),
            minimum=tf.convert_to_tensor([1.1804080208620997]),
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

        def bc(x: trieste.types.TensorType) -> trieste.types.TensorType:
            return tf.concat(
                (
                    trieste.objectives.single_objectives.branin(x),
                    test_functions.currin(x, tf.pow, tf.exp),
                ),
                axis=-1,
            )

        return trieste.objectives.multi_objectives.MultiObjectiveTestProblem(
            name="BraninCurrin",
            objective=bc,
            search_space=search_space,
            # XXX: this is wrong.
            gen_pareto_optimal_points=lambda n, seed=None: tf.stack(
                generate_pareto_optimal_points(n, bc, search_space), axis=-1
            ),
        )

    raise ValueError(f"{func} is not an accepted Trieste test function")


def create_noisey_f(
    f, noise_stdev
) -> Callable[[trieste.types.TensorType], trieste.types.TensorType]:
    """Makes `f` noisey (with deviation `noise_stdev`).

    If `noise_stdev` is `None`, this will return a noiseless problem `f`.
    """

    mvn = tfp.distributions.MultivariateNormalDiag(
        scale_diag=tf.convert_to_tensor(noise_stdev, tf.float64)
    )

    def noisey_f(X: trieste.types.TensorType):
        y = f(X)
        noise = mvn.sample(len(y))

        assert y.shape == noise.shape
        return y + noise

    return noisey_f


def create_observer(f, noise_stdev: list[float] | None) -> trieste.observer.Observer:
    """Makes `f` noisey (with deviation `noise_stdev`) and a Trieste observer out of it.

    If `noise_stdev` is `None`, this will return a noiseless problem `f`.
    """

    if noise_stdev is None:
        print("Creating observer without noise - your problem has no noise.")
        return trieste.objectives.utils.mk_observer(f)

    return trieste.objectives.utils.mk_observer(create_noisey_f(f, noise_stdev))


def create_partial_moo_problem(
    moo_problem: trieste.objectives.multi_objectives.MultiObjectiveTestProblem,
    o_dim: int,
    z: list[int],
) -> trieste.objectives.multi_objectives.MultiObjectiveTestProblem:
    assert o_dim > 0
    for z_i in z:
        assert 0 <= z_i < o_dim

    o_dims_observed = list(set(range(o_dim)) - set(z))

    def partial_objective_function(X: trieste.types.TensorType):
        o = moo_problem.objective(X)
        return tf.gather(o, o_dims_observed, axis=-1)

    return trieste.objectives.multi_objectives.MultiObjectiveTestProblem(
        f"{moo_problem.name}-latent-{z}",
        partial_objective_function,
        moo_problem.search_space,
        moo_problem.gen_pareto_optimal_points,  # XXX: this is wrong.
    )


def generate_pareto_optimal_points(n: int, objective, space: trieste.space.SearchSpace):
    """A very dumb basic random sampling function.

    Creates a random sampler, and keeps those that are not dominated.
    """

    def gen():
        """Our super simple random sampler of `(x, y)` data."""
        x: trieste.types.TensorType = space.sample(1)
        y: trieste.types.TensorType = objective(x)

        return (x, y)

    def comp(
        d1: tuple[trieste.types.TensorType, trieste.types.TensorType],
        d2: tuple[trieste.types.TensorType, trieste.types.TensorType],
    ) -> int:
        """Our comparison function over two tensors.

        If `y_1 < y_2` -> 1,  if `y_1 > y_2` -> -1, else returns 0.
        """

        y1, y2 = d1[1], d2[1]
        if tf.reduce_all(tf.less(y1, y2)):
            return 1
        if tf.reduce_all(tf.greater(y1, y2)):
            return -1

        return 0

    # Here we filter out the non-dominating.
    pareto_points = moo.generate_front(n, gen, comp)

    return [d[0] for d in pareto_points]


class RandomAgent(interaction_loops.Agent):

    def __init__(self, search_space: trieste.space.SearchSpace):
        self.search_space = search_space

    def pick_query(self) -> tuple[Any, dict[str, Any]]:
        return self.search_space.sample(1), {}

    def observe(self, query, feedback, evaluation) -> None:
        del query, feedback, evaluation


class BO(interaction_loops.Agent):

    def __init__(
        self,
        data: trieste.data.Dataset,
        search_space: trieste.space.SearchSpace,
        acqf: str,
        acqf_options: dict[str, Any],
    ):
        self.data = data
        self.search_space = search_space
        self.acqf = create_acqf(acqf, self.search_space, acqf_options)
        self.mean_acqf = create_acqf("mean", self.search_space, {})

    def pick_query(self) -> tuple[Any, dict[str, Any]]:
        query_stats: dict[str, Any] = {"optimization_fails": 0}

        # Create the model (or return random sample if fails).
        try:
            y_sca, y_mean, y_std = utils.normalize(self.data.observations)
            data_sca = trieste.data.Dataset(self.data.query_points, y_sca)
            model = posteriors.create_gp(data_sca, self.search_space)

        except tf.errors.InvalidArgumentError as e:
            print(
                "`TriesteBO.pick_query` failed to fit model, returning random sample.",
                e,
            )
            query_stats["optimization_fails"] += 1
            return self.search_space.sample(1), query_stats

        # Pick query given model (or return random if fails).
        try:
            query = optimize_acqf(self.acqf, data_sca, model, self.search_space)
        except trieste.acquisition.optimizer.FailedOptimizationError as e:
            print(
                "`TriesteBO.pick_query` failed to optimize, returning random sample.",
                e,
            )
            query_stats["optimization_fails"] += 1
            query = self.search_space.sample(1)

        # Get MAP (for reporting statistics).
        try:
            arg_map = optimize_acqf(self.mean_acqf, data_sca, model, self.search_space)
            # Un-normalize predicted MAP.
            map_mean = model.predict(arg_map)[0] * y_std + y_mean

            query_stats["map"] = {"x": np.array(arg_map), "y": np.array(map_mean)}

        except trieste.acquisition.optimizer.FailedOptimizationError as e:
            print("`CompositeBO.pick_query` failed to find MAP.", e)
            query_stats["optimization_fails"] += 1

        query_stats["observation_noise"] = np.array(
            model.get_observation_noise()
        ) * tf.pow(y_std, 2)

        return query, query_stats

    def observe(self, query, feedback, evaluation) -> None:
        del evaluation
        self.data += trieste.data.Dataset(query, feedback["y"])


class CompositeBO(interaction_loops.Agent):
    """Multi-objective optimization agent that *knows* the scalarization weights."""

    def __init__(
        self,
        composition_weights: list[float],
        data: trieste.data.Dataset,
        search_space: trieste.space.SearchSpace,
        acqf: str,
        acqf_options: dict[str, Any],
    ):
        """Initiates an agent that performs BO on scalarized MOO problem.

        This agents knows the scalarization weights `composition_weights`,
        but builds surrogate models for the objectives.

        The optimization uses the surrogate models, in combination with scalar weights,
        to maximize the `acqf`.
        """
        self.weights = composition_weights
        self.data = data
        self.search_space = search_space

        self.acqf = create_acqf(acqf, self.search_space, acqf_options)
        self.mean_acqf = create_acqf("mean", self.search_space, {})

    def pick_query(self) -> tuple[Any, dict[str, Any]]:
        query_stats: dict[str, Any] = {"optimization_fails": 0}

        # Create the model (or return random sample if fails).
        try:
            model = posteriors.CompositeGP(self.data, self.search_space, self.weights)

        except tf.errors.InvalidArgumentError as e:
            print(
                "`CompositeBO.pick_query` failed to fit model, returning random sample.",
                e,
            )
            query_stats["optimization_fails"] += 1
            return self.search_space.sample(1), query_stats

        # Pick query given model.
        try:
            query = optimize_acqf(self.acqf, self.data, model, self.search_space)
        except trieste.acquisition.optimizer.FailedOptimizationError as e:
            print(
                "`CompositeBO.pick_query` failed to optimize, returning random sample.",
                e,
            )
            query_stats["optimization_fails"] += 1
            query = self.search_space.sample(1)

        try:
            arg_map = optimize_acqf(self.mean_acqf, self.data, model, self.search_space)
            map_mean = model.predict(arg_map)[0]

            query_stats["map"] = {"x": np.array(arg_map), "y": np.array(map_mean)}
        except trieste.acquisition.optimizer.FailedOptimizationError as e:
            print("`CompositeBO.pick_query` failed to find MAP.", e)
            query_stats["optimization_fails"] += 1

        query_stats["observation_noise"] = np.array(model.get_observation_noise())

        return query, query_stats

    def observe(self, query, feedback, evaluation) -> None:
        del evaluation
        self.data += trieste.data.Dataset(query, feedback["o"])


class UtilityBO(interaction_loops.Agent):
    """Multi-objective optimization agent that *knows* the objective functions.

    This BO agent does *not* know the utility weights and, hence, tracks a posterior over those.
    """

    def __init__(
        self,
        X: tf.Tensor,
        O: tf.Tensor,
        Y: tf.Tensor,
        objectives: trieste.objectives.multi_objectives.MultiObjectiveTestProblem,
        acqf: str,
        acqf_options: dict[str, Any],
    ):
        """Initiates an agent that performs BO on scalarized MOO problem.

        This agents knows the objective functions `objectives`,
        but builds a posterior over the utility weights.

        The optimization uses this posterior, in combination with `objectives`,
        to maximize the `acqf`.

        - `data_objectives` is supposed to contain `x -> o`.
        - `data_costs` is supposed to contain `o -> u`, from which we then infer the weights.
        """
        self.X, self.O, self.Y = X, O, Y
        self.objectives = objectives

        self.acqf = create_acqf(acqf, self.objectives.search_space, acqf_options)
        self.mean_acqf = create_acqf("mean", self.objectives.search_space, {})

    def pick_query(self) -> tuple[Any, dict[str, Any]]:
        # Useful for typing warnings in the rest of the function
        assert isinstance(self.X, tf.Tensor)
        assert isinstance(self.Y, tf.Tensor)
        assert isinstance(self.O, tf.Tensor)

        query_stats: dict[str, Any] = {"optimization_fails": 0}

        # Create the model (or return random sample if fails).
        try:
            model = posteriors.UtilityDistribution(
                self.O, self.Y, self.objectives.objective
            )

            # Report weight distribution.
            query_stats["weight_posterior"] = (
                model.weight_posterior.weighted_particles.sample([100]).numpy()
            )

            query_stats["weight_map"] = (
                model.weight_posterior.weighted_particles.map().numpy()
            )

        except (tf.errors.InvalidArgumentError, ValueError) as e:
            print(
                "`UtilityBO.pick_query` failed to fit model, returning random sample.",
                e,
            )
            query_stats["optimization_fails"] += 1
            return self.objectives.search_space.sample(1), query_stats

        # For acquisition functions which may use existing data,
        # `acqf_data` is x -> y data for reference.
        acqf_data = trieste.data.Dataset(self.X, self.Y)

        # Pick query given model.
        try:
            query = optimize_acqf(
                self.acqf, acqf_data, model, self.objectives.search_space
            )
        except trieste.acquisition.optimizer.FailedOptimizationError as e:
            print(
                "`UtilityBO.pick_query` failed to optimize, returning random sample.",
                e,
            )
            query_stats["optimization_fails"] += 1
            query = self.objectives.search_space.sample(1)

        try:
            arg_map = optimize_acqf(
                self.mean_acqf,
                acqf_data,
                model,
                self.objectives.search_space,
            )
            map_mean = model.predict(arg_map)[0]

            query_stats["map"] = {"x": np.array(arg_map), "y": np.array(map_mean)}
        except trieste.acquisition.optimizer.FailedOptimizationError as e:
            print("`UtilityBO.pick_query` failed to find MAP.", e)
            query_stats["optimization_fails"] += 1

        query_stats["observation_noise"] = np.array(model.get_observation_noise())

        return query, query_stats

    def observe(self, query, feedback, evaluation) -> None:
        del evaluation
        self.X = tf.concat([self.X, query], 0)
        self.O = tf.concat([self.O, feedback["o"]], 0)
        self.Y = tf.concat([self.Y, feedback["y"]], 0)


class MOO(interaction_loops.Agent):
    """Multi-objective optimization agent that ignores latent objectives."""

    def __init__(
        self,
        X: tf.Tensor,
        O: tf.Tensor,
        Y: tf.Tensor,
        search_space: trieste.space.SearchSpace,
        acqf: str,
        acqf_options: dict[str, Any],
        utility_noise: float = 0.1,
    ):
        """Initiates a MOO agent with GPs for each objective weight posterior for utility.

        This agent maintains a GP for each objective, bar the latent which is ignored,
        and a posterior over weights of the (linear) utility function.

        The optimization uses the surrogate models, in combination with weights posterior,
        to maximize the `acqf`.
        """
        self.X = X
        self.O = O
        self.Y = Y

        self.search_space = search_space
        self.o_dim = self.O.shape[-1]
        self.utility_noise = tf.convert_to_tensor(utility_noise, tf.float64)

        assert isinstance(self.o_dim, int)

        self.n_particles = 20**self.o_dim  # number of weight samples.
        self.n_predictions = 100  # number of samples used to `self.predict`.

        self.acqf = create_acqf(acqf, self.search_space, acqf_options)
        self.mean_acqf = create_acqf("mean", self.search_space, {})

    def pick_query(self) -> tuple[Any, dict[str, Any]]:
        query_stats: dict[str, Any] = {"optimization_fails": 0}

        assert (
            isinstance(self.X, tf.Tensor)
            and isinstance(self.O, tf.Tensor)
            and isinstance(self.Y, tf.Tensor)
        )

        data = trieste.data.Dataset(self.X, self.Y)

        # Create the model (or return random sample if fails).
        try:
            model = posteriors.MOOPosterior(self.X, self.O, self.Y, self.search_space)

        except (tf.errors.InvalidArgumentError, ValueError) as e:
            print(
                "`MOO.pick_query` failed to fit model, returning random sample.",
                e,
            )
            query_stats["optimization_fails"] += 1
            return self.search_space.sample(1), query_stats

        # Pick query given model.
        try:
            query = optimize_acqf(
                self.acqf,
                data,
                model,
                self.search_space,
            )
        except trieste.acquisition.optimizer.FailedOptimizationError as e:
            print(
                "`MOO.pick_query` failed to optimize, returning random sample.",
                e,
            )
            query_stats["optimization_fails"] += 1
            query = self.search_space.sample(1)

        try:
            arg_map = optimize_acqf(self.mean_acqf, data, model, self.search_space)
            map_mean = model.predict(arg_map)[0]

            query_stats["map"] = {"x": np.array(arg_map), "y": np.array(map_mean)}
        except trieste.acquisition.optimizer.FailedOptimizationError as e:
            print("`MOO.pick_query` failed to find MAP.", e)
            query_stats["optimization_fails"] += 1

        query_stats["observation_noise"] = np.array(model.get_observation_noise())

        return query, query_stats

    def observe(self, query, feedback, evaluation) -> None:
        del evaluation
        self.X = tf.concat([self.X, query], 0)
        self.O = tf.concat([self.O, feedback["o"]], 0)
        self.Y = tf.concat([self.Y, feedback["y"]], 0)
