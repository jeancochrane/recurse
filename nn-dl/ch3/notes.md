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

## Weight initialization

- Can we improve on initializing weights using independent normalized Gaussian
  samples?
    - Gaussians have a key problem: with large enough training input, standard
      deviations of the input activations `zL` will be quite large,  causing
      some neurons to saturate quickly

- Claims that choosing different cost functions helps with saturated output
  neurons, but not hidden neurons
    - Why...?

- Instead: Choose standard deviation for weights of `1 / sqrt(len(w))`
    - "Squash" Gaussians
    - Keep biases at 1 (will explain later)
    - Now, `z` has a Gaussian distribution with mean 0 and standard deviation
      `(sqrt(1.5))` (see exercise derivation)

- Empirically: New approach to weight initialization speeds up initial training;
  however, converges to the same weights
    - Only in this case! In ch4 we'll see an example of a network whose
      performance is improved, too

## How to choose a neural network's hyperparameters?

- Heuristics for setting up hyperparameters in a neural network

- Broad strategy:
    1. Try to get _any_ decent learning (i.e. better than chance)
        - Reduce training data to a much simpler case (e.g. 0s and 1s)
        - Strip network down to simplest architecture (fewest hidden neurons)
        - Increase frequency of monitoring/logging (e.g. by lowering batch size
          per epoch, or by reporting metrics in fewer epochs)

### Learning rate

- I don't quite understand the sidebar about gradient descent...?
    - In particular:

> Briefly, a more complete explanation is as follows: gradient descent uses
> a first-order approximation to the cost function as a guide to how to decrease
> the cost. For large η, higher-order terms in the cost function become more
> important, and may dominate the behaviour, causing gradient descent to break
> down. This is especially likely as we approach minima and quasi-minima of the
> cost function, since near such points the gradient becomes small, making it
> easier for higher-order terms to dominate behaviour.

- Start by finding a **threshold value** for `η`: The smallest value at which
  the training cost begins to oscillate upwards during the first few epochs
    - This indicates an upper bound of the order of magnitude

- Why set the learning rate by monitoring the cost against the training set,
  and not the classification accuracy of the validation set?
    - Nielsen's (somewhat idiosyncratic) preference: Learning rate is meant to
      affect speed of convergence, not necessarily accuracy; hence it seems
      harmless to use training cost as a metric

### Use early stopping to determine the number of training epochs

- In brief:
    1. At the end of each epoch, compute validation accuracy
    2. If the accuracy stops improving, terminate

- Helps determine epochs automatically
    - In the early stages, you might actually want to play with epochs; in
      particular, it can be nice to know if you're overfitting

- Threshold of thumb: Classification hasn't improved in ~10 epochs
    - Not a panacea! Sometimes networks are flat and then start improving again
    - Good for initial experimentation

### Learning rate schedule

- Intuition: Start with a large learning rate; decrease as the network
  approaches convergence
    - Similar approach as early stopping: Halve the learning rate when you
      see no-improvement-in-10; stop at some threshold (e.g. 1/128th of the
      original value)

- Usually only useful when you want to get maximum performance; otherwise, just
  another headache hyperparameter to deal with

### Regularization parameter

- Advice: Figure out learning rate before regularization rate (i.e. stat with 0)
    - Nielsen doesn't have a principled approach to starting learning rate;
      instead, suggests starting at 1.0 and increasing/decreasing by orders of
      magnitude
        - Then, return and reoptimize `η`

### Mini-batch size

- Trade-off:
    - Too small, and you don't take advantage of matrix operations/hardware
    - Too large, and you're not updating weights frequently enough

- Approach: Get good-enough results from other parameters, _then_ experiment
  with mini-batch sizes
    - This works because batch size is independent of the other hyperparameters

- Plot validation accuracy against time
    - Real time, not epochs! Epochs will remain constant; the relevant variable
      is elapsed time in the world 

- Interestingly, we want to scale the learning rate as we increase the
  mini-batch size
    - This is because the update is modeled as `w - (()η/n) * sum(grad for grad in
      gradient))`, so if we scale `η` such that it stays constant as `n`
      increases, it's similar to doing online learning for `n` samples at a time
      (but can take advantage of matrix speedups)
        - Not quite _equivalent_, since the gradient for each sample is
          calculated against the same set of weights (in online learning the
          gradient is calculated against an updated set of weights each time)

### Automated tuning techniques

- Grid search: Search through hyperparameter space for optimal param
  combinations

## Other techniques


