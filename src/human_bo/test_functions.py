"""Test functions not implemented in libraries directly."""


def zhou(X, pi, exp):
    def phi_zou(X):
        return (2 * pi) ** (-0.5) * exp(-0.5 * X**2)

    part1 = 10 * (X[..., 0] - 1 / 3)
    part2 = 10 * (X[..., 0] - 2 / 3)
    return 5 * (phi_zou(part1) + phi_zou(part2))


def forrester(X, sin):
    return -((6 * X[..., 0] - 2) ** 2) * sin(12 * X[..., 0] - 4)
