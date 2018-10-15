# Exercises for Chapter 3

## 1. Proving the sigmoid derivative

### Question

Verify that `deriv(sigma(z)) = sigma(z) * (1-sigma(z))`.

### Answer

In lieu of a complete derivation, note that this can be accomplished by applying
the chain rule and the power rule and then simplifying.

## 2. Order of `y` and `a` in cross-entropy cost

### Question

One gotcha with the cross-entropy [cost] is that it can be difficult at first to
remember the respective roles of the `y`s and the `a`s. It's easy to get confused
about whether the right form is `−[y*ln(a) + (1−y)*ln(1−a)]` or `−[a*ln(y) + (1−a)*ln(1−y)]`.
What happens to the second of these expressions when `y = 0` or `y = 1`? Does this problem
afflict the first expression? Why or why not?

### Answer

When `y = 0` or `y = 1` in the first expression, one of the terms will be forced
to evaluate `ln(0)`, which is outside the domain of the natural log. This
problem is avoided in the second expression, where neither `y` is evaluated as
part of the `ln` function.

Interestingly, this implies that the cross-entropy cost will not work if `a_ljk = 0`
or `a_ljk = 1` anywhere in the network.

## 3. 

### Question

In the single-neuron discussion at the start of this section, I argued that the
cross-entropy is small if `σ(z) ≈ y` for all training inputs. The argument relied on
`y` being equal to either 0 or 1. This is usually true in classification problems,
but for other problems (e.g., regression problems) `y` can sometimes take values
intermediate between 0 and 1. Show that the cross-entropy is still minimized
when `σ(z) = y` for all training inputs.

### Answer

I was able to make the expression equal to 0 by taking the partial derivative of the
cost function with respect to `a` and then setting `a = y`. I'm not sure if this
is right though...? Would appreciate a second look!

## 4. Monotonicity of softmax

### Question

 Show that `∂aLj/∂zLk` is positive if `j=k` and negative if `j≠k`. As a consequence,
 increasing `zLj` is guaranteed to increase the corresponding output activation,
 `aLj`, and will decrease all the other output activations. We already saw this
 empirically with the sliders, but this is a rigorous proof.

### Answer

I'm having trouble with this one! Stuck on the differentiation.

## 5. Non-locality of softmax

### Question

A nice thing about sigmoid layers is that the output `aLj` is a function of the
corresponding weighted input, `aLj = σ(zLj)`. Explain why this is not the case for
a softmax layer: any particular output activation `aLj` depends on all the
weighted inputs.

### Answer

The key here is the denominator of the softmax function, which sums over _all_
weighted inputs in the layer.

## 6.

### Question

As discussed above, one way of expanding the MNIST training data is to use small
rotations of training images. What's a problem that might occur if we allow
arbitrarily large rotations of training images?

### Answer

If we allow arbitrarily large rotations of training images, the digits may not
be accurate to their labels anymore! (e.g. "9" rotated 180 degrees should be
classified as "6")

## 7.

### Question

Verify that the standard deviation of `z=∑jwjxj+b` in the paragraph above is
`sqrt(3/2)`. It may help to know that: (a) the variance of a sum of independent
random variables is the sum of the variances of the individual random variables;
and (b) the variance is the square of the standard deviation.

### Answer

See paper derivation. (You also need to know that the variance of a random
variable `Y` scaled by a constant `c` is `c**2 * var(Y)`.)

Shouldn't this be `3/2 * 1000`...?

## 8.

### Question

It's tempting to use gradient descent to try to learn good values for
hyper-parameters such as `λ` and `η`. Can you think of an obstacle to using gradient
descent to determine `λ`? Can you think of an obstacle to using gradient descent
to determine `η`?

### Answer
???
