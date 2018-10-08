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
