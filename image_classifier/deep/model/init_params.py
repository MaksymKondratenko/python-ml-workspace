import numpy as np


def init_params(dimensions):
    """
    Function to initalize weight parameters
        and bias terms for all layers

    :param dimensions: tuple with numbers of nodes
        in each layer from input to output
    :return: weight parameters and biases for level 'l' as a dict
            ...
            Wl - nl x nl-1
            bl - nl x 1
            ...
    """
    np.random.seed(1)
    params = {}

    L = len(dimensions)
    for l in range(1, L):
        n_curr = dimensions[l]
        n_prev = dimensions[l-1]
        params["W" + str(l)] = np.random.randn(n_curr, n_prev) * 0.01
        params["b" + str(l)] = np.zeros((n_curr, 1))

    return params
