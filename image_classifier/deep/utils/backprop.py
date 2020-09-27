import numpy as np


def relu(dA, Z):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    Z -- propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    return dZ


def sigmoid(dA, Z):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    Z -- backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    return dZ
