def laplace_smoothing(num, denom, alpha=1):
    """
    Apply Laplace smoothing to the given numerator and denominator.

    :param num: The numerator. Shape should be (n_features,).
    :param denom: The denominator. Shape should be (n_features,), or a scalar.
    :param alpha: The smoothing parameter.
    :return: Smoothed probabilities."""
    return (num + alpha) / (denom + alpha * len(num))

def gini(x):
    return 1 - ((x.value_counts() / len(x)) ** 2).sum()

def mse(x):
    return ((x - x.mean()) ** 2).sum()
