def update_params(params, grads, learning_rate):
    """
    Function to update params with back-propagated deltas

    :param params: parameters to update, weights and biases
    :param grads: parameter deltas
    :param learning_rate: a scalar
    :return: parameters after update
    """

    params['W1'] = params['W1'] + learning_rate * grads['W1']
    params['b1'] = params['b1'] + learning_rate * grads['b1']
    params['W2'] = params['W2'] + learning_rate * grads['W2']
    params['b2'] = params['b2'] + learning_rate * grads['b2']

    return params