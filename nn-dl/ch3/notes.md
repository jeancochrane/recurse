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

### Addressing learning slowdown with softmax

- First, define the log-likelihood cost:

```python
def log_likelihood(aLy):
    """
    aLy must be a probability (range [0, 1]).
    """
    return -math.log(aLy)
```

- Partial derivatives w.r.t. biases and weights again avoid the derivative
    - Similar partial derivatives as sigmoid + cross-entropy

## Overfitting and regularization

- **Early stopping**: At the end of each epoch, compute classification accuracy
  on the validation set, and stop once we've reached an acceptable threshold
  (once the accuracy has _saturated_)

### Regularization

- Techniques to reduce overfitting

- L2 Regularization (or _weight decay_): Add a regularization term to the cost
  function to add a penalty for complexity
    - Intuition: Instruct the network to _prefer small weights_

```python
def reg_term(w, x, lam):
    return (lam * sum(wi**2 for wi in w)) / (2 * len(x))
```

- Note that:
    - Scaled by the regularization parameter, lambda
    - Regularization term is averaged (mean squared weights)
    - Doesn't include biases!

- Any cost function can be regularized:

```python
def regularize(C, w, x, y, lam):
    return C(x, y) + reg_term(w, x, lam) 
```

- Large weights will only be considered when the first term of the cost function
  `C0` is greatly improved

- Also known as _weight decay_, because the update equation factors to include
  the term `(1 - (lr * lam) / len(x)) * w`

- Another benefit of regularization: Less likely to get caught in local minima
    - Heuristically: Large weights -> long weight vector; updates are designed
      to produce small changes to weights; hence update won't "explore the
      weight space" fully
        - Not totally clear to me what this means...?

### Why does regularization reduce overfitting?

- Key question: What does it mean to say that "smaller weights" are "less
  complex"?
    - My guess: Weights that are close to 0 are similar to reducing the order of
      the polynomial (hence reducing complexity of the model)
    - Nielsen's explanation: With small weights, small changes to inputs won't change the overall
      behavior much
        - e.g. higher _variance_

- Sidebar: Why can a 9th-order polynomial fit 9 data points exactly...?

- No firm theoretical basis for regularization's benefit; it just works
  empirically
    - Something of a kludge
    - Deeper issue: How to generalize from small amounts of data?
        - Not currently well-formulated
    - Gradient-based learning appears to have a "self-regularization effect",
      although this is not well-understood

- Regularizing bias is less important, since bias is not sensitive to the input
    - (Recall that the primary problem with overfitting is seeing inputs that don't match
      up with the training set)

### Other techniques for regularization

- Wide diversity of possible regularizations!

#### L1 Regularization

- Add the sum of the absolute values of the weights (as opposed to the sum of
  squares)
    - Difference: Weights shrink by a constant amount (as opposed to a function
      of `w`)
        - When `w` is large: L1 shrinks `w` slower
        - When `w` is small: L1 shrinks `w` faster
        - Cumulative effect: Drive "unimportant" weights to 0, concentrate
          weights in the network

#### Dropout

- For each epoch: Randomly weight half the neurons at 0; forward and backward
  pass through the network; repeat 
    - Heuristically, this is similar to training many different neural nets, and
      then having them "vote" on the predicted output
        - Also: Reduces "co-dependency" between neurons
            - Make sure that the model is resistent to loss of individual pieces
              of evidence

#### Artificially expanding the training data

- Intuition: Make small modifications to (classification) inputs to artificially
  produce more training data
    - E.g. Rotate or skew images; add background noise or speed up/slow down sound
