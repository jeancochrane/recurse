# Chapter 3: Improving the way neural networks learn

([Link](http://neuralnetworksanddeeplearning.com/chap3.html))

## Introduction

- This chapter: introduce improvements to basic backprop algorithm
    - Cross-entropy cost function
    - Regularization methods
    - Better weight initialization
    - Heuristics for choosing good hyperparameters

## Cross-entropy cost function

- Learning should happen faster when errors are well-defined, yet this is not
  the case for squared cost

- Size of updates is determined (in part) by the partial derivatives of the cost
  function with respect to the weights and biases, so saying "learning is too
  slow" is equivalent to saying that "the partial derivatives are small"

- Recall that when a sigmoid neuron's output is close to 0 or 1,
  `partial_deriv(C, w)` and `partial_deriv(C, b)` will also be very small
    - This is a key source of learning slowdown

### Introducing the cross-entropy cost function

- Cross-entropy function:

```python
def cross_entropy_cost(x, y):
    return -sum(yi*math.log(xi) + (1-yi)*math.log(1-xi) for xi, yi in zip(x, y)) / len(x)
```

- Why can cross-entropy be called a cost function?
    - Always non-negative
        - Inputs to logs are always in the range `[0, 1]`, so the log terms are
          always negative
        - Minus sign out front converts the sum to positive
        - When `y ~ x`, the cost will be low
            - Assumes `y` is either 0 or 1! (Can you see why?)

- Partial derivative of cross-entropy cost w.r.t. a weight:

```python
def deriv_cross_entropy(x, y, z):
    return sum(xi*(sigma(zi) - yi) for xi, yi, zi in zip(x, y, z)) / len(x)
```

- Nice properties of this partial derivative:
    - Cost learns at a rate controlled by `a-y` and scaled by `x`
    - Avoids learning slowdown caused by the derivative of the sigmoid, since
      that term is no longer part of the equation

- Interesting answer to the question: Why didn't we just adjust the learning
  rate?
    - There's a limit on how well the learning rate can help, since what we're
      after is how to speed up learning at _certain points_ in the cost
      function (here, when the training example is unambiguously wrong)
        - I wonder if a dynamic learning rate would have a similar effect...?

- When using sigmoid activation, cross-entropy is nearly always a better cost
  function
    - Random weights mean that often the initial predictions are very wrong;
      sigmoid means that very wrong guesses will learn slowly with quadratic
      cost 

### What does the cross-entropy mean? Where does it come from?

- Cross-entropy stems from the goal of finding a cost function that produces
  partial derivatives of the following form:

```python
partial_deriv(C, w_j) == x_j * (a - y)

partial_deriv(C, b) == (a - y)
```

- Intuition: Cross-entropy is a measure of _surprise_
    - Think of `a` as the probability that `y = 0`, and `1 - a` as the
      probability that `y = 1`

## Softmax

- Another approach to addressing learning slowdown

- Instead of replacing the cost function: replace the _activation function_
  (sigmoid)

- Rescale weighted inputs and use them to produce a probability distribution:

```python
def softmax(zi, z):
    return np.exp(zi) / sum(np.exp(zl) for zl in z)
```

- Nice properties:
    - Exponentials mean that all activations are positive
    - Sum in the denominator means that all activations in a layer sum to 1
