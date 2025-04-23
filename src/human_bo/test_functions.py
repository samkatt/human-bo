"""Test functions not implemented in libraries directly."""


def zhou(X, pi, exp):
    def phi_zou(X):
        return (2 * pi) ** (-0.5) * exp(-0.5 * X**2)

    part1 = 10 * (X[..., 0] - 1 / 3)
    part2 = 10 * (X[..., 0] - 2 / 3)
    return 5 * (phi_zou(part1) + phi_zou(part2))


def forrester(X, sin):
    return -((6 * X[..., 0] - 2) ** 2) * sin(12 * X[..., 0] - 4)


def currin(X, power, exp):
    """Currin function as described most often in BO.

    Approximates:
    - max: x = [.2166, 0], y = 13.79872184813862
    - min: x = [0, 1] , y = 1.1804080208620997
    """
    x_0 = X[..., :1]
    x_1 = X[..., 1:]
    factor1 = 1 - exp(-1 / (2 * x_1))
    numer = 2300 * power(x_0, 3) + 1900 * power(x_0, 2) + 2092 * x_0 + 60
    denom = 100 * power(x_0, 3) + 500 * power(x_0, 2) + 4 * x_0 + 20

    return factor1 * numer / denom
