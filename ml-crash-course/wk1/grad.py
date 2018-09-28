def gradient(C, w, b):
    """
    Calculate the gradient for a cost function C based on the weight vector
    w and the bias b.
    """
    return list(partial_deriv(C, b)) + list(partial_deriv(C, wi) for wi in w)

def delta_v(C, w, b):
    return -learning_rate * gradient(C, w, b)
