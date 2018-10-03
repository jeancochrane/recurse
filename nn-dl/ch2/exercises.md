# Exercises for Chapter 2: How the backpropagation algorithm works

## 1: Backpropagation with a single modified neuron

### Question

Suppose we modify a single neuron in a feedforward network so that the output
from the neuron is given by `f(z)`, where `f`  is some function other than the
sigmoid. How should we modify the backpropagation algorithm in this case?

### Answer

The backpropagation algorithm needn't be modified in this case; instead, we need
to ensure that we use the derivative of `f` in place of the derivative of the
sigmoid in equations BP1 and BP2. 

## 2: Backpropagation with linear neurons

### Question

Suppose we replace the usual non-linear `σ` function with `σ(z)=z`
throughout the network. Rewrite the backpropagation algorithm for this case.

### Answer

Since `σ'(z)=1`, we don't have to compute `σ'` anymore in order to compute the
output error or to propagate the error backwards through the network. The steps
to the algorithm should remain the same, but the computations will be simpler.
