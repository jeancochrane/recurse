# Notes on Chapter 2: How the backpropogation algorithm works

([Link](http://neuralnetworksanddeeplearning.com/chap2.html))

## Introduction

- Backpropogation = fast algorithm for computing gradients

- Foundation of modern speedups in neural networks, making them viable

- In essence: Expression for the partial derivative of the cost function `C`
  with respect to any weight `w` or bias `b` in the network
    - Two interpretations:
        1. Fast algorithm for learning
        2. Way of determining how changes in weights and biases will affect the
           overall network
            - Intuition: How quickly does the cost change when we change weights or
              biases?

## A fast matrix-based approach to computing the output from a neural net

- Q: Can we use matrices to compute outputs in batches, instead of
  neuron-by-neuron?
    - A: Yes!

- Notational note: `w_ljk` is the weight that affects the connection...
    - Moving from layer `(l-1)` to layer `l`
        - In general, superscript = layer number
    - Outputs to neuron `j` in layer `l`
        - In general, subscript = neuron number
    - Proceeds from neuron `k` in layer `(l-1)
        - In this way, the super/subscripts appear to move "backwards"
            - This is useful because `j, k` will correspond to the row and
              column number of the weight matrix (row of input matrix -> column
              of output matrix)

- Activations in a given layer `l` are related to activations in the previous
  layer `(l-1)`:

```python
def a(l, j):
    """
    Pseudocode function relating the activation at a_lj to the activation at
    a_(l-1)j
    """
    return sigmoid(sum(w(l, j, k) * a(l-1, k) + b(l, j)) for k in l-1) 
```

- Rewrite this in matrix form!
    - Define a weight matrix `w_l` for the layer `l`
        - `w_ljk` will be in row `j` and column `k` of the matrix `w_l`
    - Define a bias vector `b_l` for the layer `l`
        - `b_lj`, one bias for each neuron `j` in `l`
    - Define an activation vector `a_l` for the layer `l`
        - `a_lj`, one activation for each neuron `j` in `l`
    - Define a _vectorized_ version of `sigmoid()` that applies the function to
      each element of a vector `v`
        - Use the same functional notation

```python
    def a(l):
        """
        Pseudocode function relating the activation at a_l-1 to the activation
        at a_l using matrix algebra.
        return v_sigmoid(numpy.dot(w(l), a(l-1)) + b(l)) 
```

- Nielsen claims that the notational quirk `j, k` comes from the fact that if we
  flipped the dimensions, we'd have to use the transpose of the weight matrix
  instead (`w` -> `w.T`)...? I don't see why?

- Input to the sigmoid is often a useful computational unit in itself
    - Called the **weighted input `z_l` to the neurons in layer `l`**

## The two assumptions we need about the cost function

1. Cost function `C` needs to be an average over cost functions `C_x` for
   individual training observations `x`
    - Why? Because backpropogation allows us to compute the partial derivative
      w.r.t weights and biases for _a single training example `x`_, such that
      the partial derivative for the cost `C` can be recovered by
      averaging all partial derivatives 

2. Cost `C` needs to be a function of the outputs of the network, suche that `C
   = C(a_L)`

## The Hadamard product

- One unusual operation we need for backpropogation: element-wise multiplication
  of two vectors, or the _Hadamard product_
    - In NumPy: `numpy.multiply()`
