import numpy as np

def compute_cost (AL, Y):
    q = - 1 / len(Y)
    cost = q * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    return cost