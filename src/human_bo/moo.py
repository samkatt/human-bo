
def sample_preference_weights(o_dim: int):
    assert o_dim > 1

    weights = [random.uniform(0, 1) for _ in range(o_dim)]
    total = sum(weights)

    return [w / total for w in weights]
