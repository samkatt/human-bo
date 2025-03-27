"""Test functions that are not implemented in BoTorch."""

import tensorflow as tf
import torch
import trieste
from botorch import test_functions
from botorch.test_functions import base


def zhou(X, pi, exp):
    def phi_zou(X):
        return (2 * pi) ** (-0.5) * exp(-0.5 * X**2)

    part1 = 10 * (X[..., 0] - 1 / 3)
    part2 = 10 * (X[..., 0] - 2 / 3)
    return 5 * (phi_zou(part1) + phi_zou(part2))


# TODO: rename to ZhouBotorch
class Zhou(test_functions.SyntheticTestFunction):
    """The Zhou (1-dimensional) function (https://www.sfu.ca/~ssurjano/zhou98.html)"""

    dim = 1
    _bounds = [(-0.0, 1.0)]
    _optimizers = [tuple([1 / 3]), tuple([2 / 3])]
    _optimal_value = 2.002595246981888

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        return zhou(X, torch.pi, torch.exp)


def forrester(X, sin):
    return -((6 * X[..., 0] - 2) ** 2) * sin(12 * X[..., 0] - 4)


# TODO: rename to ForresterBotorch
class Forrester(test_functions.SyntheticTestFunction):
    """The Forrester (1-dimensional) function (https://www.sfu.ca/~ssurjano/forretal08.html)"""

    dim = 1
    _bounds = [(-0.0, 1.0)]
    _optimizers = [tuple([0.7572])]
    _optimal_value = 6.020738786441099

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        return forrester(X, torch.sin)


# TODO: rename to `create_`.
def pick_test_function(func: str, noise: float) -> test_functions.SyntheticTestFunction:
    """Instantiate the given function to optimize.

    :func: string description of the test function to return
    :noise: standard deviation of the noise
    """

    if func == "Forrester":
        return Forrester(noise_std=noise)
    if func == "Zhou":
        return Zhou(noise_std=noise)
    if func == "Hartmann":
        return test_functions.Hartmann(negate=True, noise_std=noise)
    if func == "Branin":
        return test_functions.Branin(negate=True, noise_std=noise)
    if func == "Rosenbrock2D":
        return test_functions.Rosenbrock(
            dim=2, negate=True, bounds=[(-5.0, 5.0), (-5.0, 5.0)], noise_std=noise
        )
    if func == "Ackley1D":
        return test_functions.Ackley(dim=1, noise_std=noise, negate=True)
    if func == "DixonPrice1D":
        return test_functions.DixonPrice(dim=1, noise_std=noise, negate=True)
    if func == "Griewank1D":
        return test_functions.Griewank(dim=1, noise_std=noise, negate=True)
    if func == "Levy1D":
        return test_functions.Levy(dim=1, noise_std=noise, negate=True)
    if func == "Rastrigin1D":
        return test_functions.Rastrigin(dim=1, noise_std=noise, negate=True)
    if func == "StyblinskiTang1D":
        return test_functions.StyblinskiTang(dim=1, noise_std=noise, negate=True)

    raise ValueError(f"{func} is not an accepted (single objective) test function")


# TODO: rename to `create_`.
def pick_moo_test_function(
    func: str, noise: list[float] | None
) -> base.MultiObjectiveTestProblem:
    """Instantiate the given multi-objective function to optimize.

    :func: string description of the test function to return
    :noise: standard deviations of the noise, None means no noise.
    """

    if func == "BraninCurrin":
        return test_functions.BraninCurrin(noise_std=noise)

    raise ValueError(f"{func} is not an accepted MOO test function")


# TODO: rename to `create_`.
def pick_trieste_test_function(
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
                -zhou(x, tf.experimental.numpy.pi, tf.exp), [-1, 1]
            ),
            search_space=trieste.space.Box([0.0], [1.0]),
            minimizers=tf.convert_to_tensor([[1 / 3], [2 / 3]]),
            minimum=tf.convert_to_tensor([-2.002595246981888]),
        )
    if func == "Forrester":
        return trieste.objectives.single_objectives.SingleObjectiveTestProblem(
            name="Forrester",
            objective=lambda x: tf.reshape(-forrester(x, tf.sin), [-1, 1]),
            search_space=trieste.space.Box([0.0], [1.0]),
            minimizers=tf.convert_to_tensor([[0.7572]]),
            minimum=tf.convert_to_tensor([-6.020738786441099]),
        )
    if func == "Branin":
        return trieste.objectives.single_objectives.Branin

    raise ValueError(f"{func} is not an accepted Trieste test function")
