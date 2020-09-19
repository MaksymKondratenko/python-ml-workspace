import numpy as np


def init_params(nx, nh, ny):
    """
    Function to initalize weight parameters
        and bias terms for all layers

    :param nx: size of the input layer
    :param nh: size of the hidden layer
    :param ny: size of the output layer

    :return: weight parameters and biases
            (W1, d1, W2, d2)
            W1 - nh x nx
            b1 - nh x 1
            W2 - ny x nh
            b2 - ny x 1
    """
    np.random.seed(1)

    W1 = np.random.randn(nh, nx) * 0.01
    d1 = np.zeros((nh, 1))
    W2 = np.random.randn(ny, nh) * 0.01
    d2 = np.zeros((ny, 1))

    params = {"W1": W1,
              "d1": d1,
              "W2": W2,
              "d2": d2}
    return params
