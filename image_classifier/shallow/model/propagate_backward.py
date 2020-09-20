import numpy as np

from image_classifier.shallow.utils.backprop import relu, sigmoid


def propagate_backward(dA, A_prev, Z, W, activation):
    """
    Function to the activation loss backwards

    :param dA: Vector of the activation losses
        for the current level nodes
    :param A_prev: Vector of the activations
        of the previous level nodes
    :param Z: Vector of the pre-activations
        of the current level nodes
    :param W: Matrix of the weights for connections
        between current level and prev level nodes
    :param activation: a type of activation to select
        either 'relu' or 'sigmoid'
    :return:
        dA - Vector of the activation losses
            for the previous level nodes
        dW - Matrix of the weight deltas for connections
            between current level and prev level nodes
        db - Vector of bias deltas for current level nodes
    """

    if activation == 'relu':
        dZ = relu(dA, Z)
    if activation == 'sigmoid':
        dZ = sigmoid(dA, Z)

    return linear_backward(dZ, A_prev, W)


def linear_backward(dZ, A_prev, W):
    q = 1 / len(A_prev)

    dW = q * np.dot(dZ, A_prev.T)
    db = q * np.sum(dZ, axis=1, keepdims=True)
    dA = np.dot(W.T, dZ)

    return dA, dW, db
