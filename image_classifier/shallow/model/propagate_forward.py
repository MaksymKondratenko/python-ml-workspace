import numpy as np


def propagate_forward(A_prev, W, b, activation):
    """
    Forward propagation linear function

    :param A_prev: Vector of node activations for a previous layer 'l' (nl-1 x 1)
    :param W: Matrix of a connection weights between
        current layer and previous layer nodes (nl x nl-1)
    :param b: Vector of bias terms for a current layer (nl x nl-1)
    :param activation: a type of activation to select
        either 'relu' or 'sigmoid'
    :return: Vector of node linear activations for a current layer
    """

    Z = linear_forward(A_prev, W, b)

    if activation == 'relu':
        A = np.max(0, Z)
    elif activation == 'sigmoid':
        A = 1 / (1 + np.exp(-Z))

    return A


def linear_forward(A, W, b):
    return np.dot(W, A) + b