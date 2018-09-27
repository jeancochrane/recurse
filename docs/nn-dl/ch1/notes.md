# Notes on Chapter 1: Using neural nets to recognize handwritten digits

([Link](http://neuralnetworksanddeeplearning.com/chap1.html))

## Introduction

- Key concepts to cover in this chapter:
    - Two types of articifial neurons
        - Perceptrons
        - Sigmoid neurons
    - Stochastic gradient descent (standard learning algorithm)

## Perceptrons

- Older model of artificial neuron; less commonly used today
    - Good foundation

- Basic idea: Take several binary inputs and produce a single binary output:

```
x1
  \
x2 -> () -> output
  /
x3
```

- Simple rule to compute outputs: Take **weighted sum of inputs**
    - If weighted sum > threshold value, return 1; otherwise, return 0
    - Note that this is also the _dot product_ of the weight and input vectors

```python
def activation(input, weights, threshold):
    """
    Summation version of the perceptron activation function.
    """
    return 1 if sum(w*x for w, x in zip(weights, input)) >= threshold else 0

def activation(x, w, b):
    """
    Simplified version of the activation function using matrix algebra and bias.
    """
    return 1 if numpy.dot(w, x) + b >= 0 else 0
```

- Alternate conception: **device that makes decisions by weighing evidence**
    - Inputs are evidence; weight indicates "how important" each type of
      evidence is

- Model variables: weights and thresholds
    - Threshold is also known as _bias_ when it is moved to the left-hand side
      of the activation equality
        - Alternate way of thinking about it: How easy is it to get the neuron
          to fire?
        - Intuition: High bias -> easier for neuron to fire (since threshold
          will be very small/negative); low bias -> harder for neuron to fire

- Output of one perceptron (or set of perceptrons) can "feed" into another; we
  call these assemblages _layers_

### Perceptrons as universal computers

- Perceptrons can act as NAND gates, which are functionally complete (can
  represent any logic)
    - Two inputs with weight -2, bias of 3
        - `00` -> `-2*0 + -2*0 + 3 = 3` -> `1`
        - `01` -> `-2*0 + -2*1 + 3 = 1` -> `0`
        - `11` -> `-2*1 + -2*1 + 3 = -1` -> `0`

- I don't get why NAND gates are universal computers...?

- Often conventional to draw _input layers_ of perceptrons -- these have no
  inputs, just configured to output a single value
    - Could we say that the value of the output is the bias of the input layer...?

- Perceptrons are more than just NAND gates -- because learning algorithms allow
  them to _learn their weights and biases_, they can automatically generate
  solutions to problems!

## Sigmoid neurons

- Intuition for learning algorithms:
    - We want small changes to weights/biases to correspond to small changes in
      outputs
        - Unfortunately, perceptrons are very sensitive to small changes!

- Sigmoids address the problem of perceptrons being too sensitive to small
  changes
    - Instead of binary inputs, inputs can take any value _between 0 and 1_
    - Instead of binary outputs, output is `sigmoid(np.dot(x, w) + b)`
        - Also known as the _logistic_ function

```python
def sigmoid(z):
    """
    Sigmoid function.
    """
    return 1 / (1 + (math.e ** -z))

def sigmoid_activation(w, x, b):
    """
    Sigmoid neuron activation function.

    Params:
        - w: Weight vector.
        - x: Input vector.
        - b: Bias.
    """
    return sigmoid(numpy.dot(w, x) + b)
```

- To understand sigmoid, look at the extremes:
    - As `z -> infinity`: `e ** -z -> 0`, hence `sigmoid(z) -> 1`
    - As `z -> -infinity`: `e ** -z -> infinity`, hence `sigmoid(z) -> 0`

- Shape of the sigmoid is a _smoothed step function_
    - Step function -> perceptron, so a sigmoid neuron is in effect a smoothed
    perceptron

- With the sigmoid, `delta(output)` is a linear function of the changes `delta(weights)` and
  `delta(bias)`, which makes it easy to choose changes to weights and bias that
  correspond to small changes in the output
    - I don't quite understand the math...? Would appreciate a rundown!
        - Apparently partial derivatives are easy to calculate for exponentials,
          which makes the sigmoid function attractive?
    - Key: The sigmoid isn't particularly special; any function with
      a similar smoothing property and a linear relationship between the output
      and the weights/bias will be an appropriate _activation function_

- Interpreting sigmoid output:
    - When the output needs to be interpretable, we can set a threshold (such as
      0.5) beyond which the output should be interpreted as one class vs. the
      other
