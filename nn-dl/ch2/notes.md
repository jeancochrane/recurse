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

## The four fundamental equations behind backpropagation

- First, want to be able to measure _error_ for a given neuron
    - Claim: `partial(C, z_lj)` is a decent measure of error
        - To see this, think about behavior when the weighted input `z_lj`
          changes: If `partial(C, z_lj)` is large, then changing the weighted
          input by `delta(z_lj)` can have a big effect; on the other hand if
          `partial(C, z_lj)` is close to zero, then it's difficult to affect the
          cost by changing `z_lj`
            - All of this because the change in `C` w.r.t `z_lj` can be
              represented by `partial(C, z_lj) * delta(z_lj)`

- Four fundamental equations:
    1. Equation for the output layer `error(L)` (i.e. `partial(C, z_lj)`)
    2. Equation for the error `error(L)` in terms of the error in the next
       layer, `error(L+1)`
        - This lets us move "backwards" through the network
    3. Equation for the rate of change of the cost w.r.t. any _bias_ in the
       network
    4. Equation for the rate of change of the cost w.r.t. any _weight_ in the
       network

### The four equations in detail

#### Output error of a neuron

- Intuition: Rate of change of the activation of an output neuron, scaled by the
  "importance" of that neuron to the network. 

```python
def output_error(L, j):
    """
    The error (represented by the partial derivative of C wrt the activation)
    of the jth neuron in the output layer L.
    """
    return partial_deriv(C, activation(z(L, j)))k * deriv(activation(z(L, j)))

def vectorized_output_error(L):
    """
    Vectorized form of the error of the output layer L.
    """
    return partial_deriv(C, activation(z(L))) * deriv(activation(z(L)))
```

#### Error for layer l given error in layer l+1

- Intuition: Move the error backward through the network by multiplying it by
  its transpose weight, then multiplying by the rate of change of the activation
  of the previous neuron (similar to equation above).

```python
def prev_error(l):
    """
    The error of a layer l given the error in the proceeding layer l+1.
    """
    return np.dot(np.transpose(w(l+1)), output_error(l+1)) * deriv(activation(z(L)))
```

#### Rate of change of the cost w.r.t. any bias

- Intuition: Exactly the same as the error!

```python
def partial_deriv_bias(l):
    return vectorized_output_error(l)
```

#### Rate of change of the cost w.r.t. any weight

- Intuition: Output error of the neuron, scaled by the input activation
    - Hence, when the input activation is small, the neuron _learns slowly_

```python
def partial_deriv_weight(l, j, k):
    return activation(l-1, j, k) * output_error(l, j)
```

### Insights from the four equations

- The sigmoid function is near-flat as `sigmoid(z)` approaches either 0 or 1 --
  at these tails, a weight applied to the layer will learn slowly, and we say the
  neuron has _saturated_

- Four fundamental equations hold for any activation function, so we can design
  activation functions with certain properties in mind
    - E.g. Could choose an activation function where the derivative is always
      positive, to avoid saturation

## The backpropagation algorithm

- Intuition: Cost is a function of outputs from the network; to get at the
  change in cost due to weights and biases, working backwards via the chain rule
  makes some sense

1. **Input `x`**
    - Set the activation `a_1`.
2. **Feedforward**
    - For each layer `l`, compute the weighted input `z_l` and the activation
      `a_l`.
3. **Output error**
    - Compute the `output_error` on the output layer `L`.
4. **Backpropagate**
    - Compute the `prev_error` for each layer in `range(2, L)`
5. **Output the gradient**
    - Use equations 3 and 4 to find the partial derivative of the cost w.r.t.
      weights and biases, and construct the gradient

## In what sense is backpropagation a fast algorithm?

- Compute all partial derivatives `partial(C, wj)` with _one_ forward pass
  through the network
    - You could compute this using the definition of a partial derivative, but
      you'd have to pass through the network for each weight!

## The big picture

- Basically: Compute the sum over the rate factor for all paths in the network
  following a change in a weight
