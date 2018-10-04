import numpy as np

# Cost function (squared error).
c = lambda (a, y): return np.sum((y - a)**2) / 2

# Activation function (sigmoid).
sigma = lambda z: return 1 / (1 + np.exp(-z))

# Derivative of the cost function (squared error) with respect to the activations in a layer.
nabla_c = lambda (a, y): return a - y

# Derivative of the activation function (sigmoid) with respect to the weighted inputs to a layer.
ddz_sigma = lambda z: return np.exp(-z) / ((np.exp(-x) + 1)**2)


def backprop(features, labels, weights, biases):
    # Store weighted inputs and activations.
    z, a = [], []

    # 1. Set the initial activation.
    aL = features

    # 2. Feedforward, computing the weighted inputs and activations.
    for w, b in zip(weights, biases):
        zl = np.dot(aL, w) + b
        al = sigma(zl)

        z.append(zl)
        a.append(al)

        aL = al

    # 3. Compute the output error.
    output_error = nabla_c(aL, labels) * numpy.vectorize(ddz_sigma)(aL)  # BP1

    # Initialize gradient for output.
    delta_w, delta_b = np.zeros(weights.shape), np.zeros(biases.shape)

    # Set the gradient for the output layer.
    delta_b[-1] = output_error  # BP3
    delta_w[-1] = aL * output_error # BP4

    # 4. and 5. Backpropagate the error through the network, constructing the
    # gradient.
    for i in range(len(z)-2, -1, -1):
        wl, zl, al = weights[i], z[i], a[i]

        output_error = np.dot(np.transpose(wl), output_error) * ddz_sigma(zl)  # BP 2

        delta_b[i] = output_error  # BP3
        delta_w[i] = al * output_error  # BP4

    return delta_w, delta_b
