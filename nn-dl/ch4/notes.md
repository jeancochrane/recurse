# Notes on Chapter 4: A visual proof that neural nets can compute any function

## Introduction

- "Universality Theorem" holds no matter the number of inputs or outputs

- Implications of the theorem: Most problems can be reframed as function
  computation
    - E.g. music tagging, machine translation
    - Economic planning?

- Many computational models are universal computers; the defining feature of
  neural nets is the attractive learning algorithm

## Two caveats

1. "Compute a function" == "deliver an arbitrarily good approximation"
    - "More hidden layers" typically equates to better approximations
        - Why?

2. Only continuous functions can be approximated in this way
    - This is because neural networks compute continuous functions of the input

- More formal statement: "Neural networks with a single hidden layer can be used
  to approximate any continuous function to any desired precision"
    - Nielsen will present a proof for two hidden layers, and provide guidance
      for how it can be modified for the one-layer case

## Universality with one input and one output

- Start with special case: One input and one output

- Consider step functions instead of sigmoid
    - This makes it easier to reason about the summed output of neurons in the
      network

- Location of the "step" is proportional to `w`, and _inversely_ proportional to
  `b`:

```
s = -b / w
```

- Hence, weighted output from the two neurons is itself a step function
    - Note the similarity here to an `if`/`else` block:

```python
if inpt >= s:
   z += 1
else:
   z += 0 
```

- This technique can be used to get as many peaks of any height as we want

- How to reconcile that weighted output will be combined with a bias and
  transformed by sigmoid?
    - Answer: Assume `b = 0` and look for a weighted output where `z = logit(f(x))` (inverse of the sigmoid)

- This loses me: "In essence, we're using our single-layer neural networks to
  build a lookup table for the function" ?

## Many input variables

- Combine earlier principle to make _tower functions_

- Similar analogy to `if`/`else` block:

```
if combined output from hidden neurons >= threshold:
    output 1
else:
    output 0
```

## Extension beyond sigmoid neurons

- Any type of activation function works following this logic, as long as:
    1. The output is well-defined as `x -> infinity` and as `x -> -infinity`
    2. The limits at infinity and negative infinity need to be different (or
       else the step function will contract into a flat line)

- Note that ReLUs don't satisfy these conditions! Need to find a separate proof
    - Why is this? Is it because as `x -> infinity` the ReLU is not
      well-defined?

## Fixing up the step functions

- Problem: There' still a (small) "window of failure" where the neuron is not
  approximating a step function
    - Nielsen's solution: Shrink the output by half and shift the bumps half
      a width over, then add the two approximations
        - This prevents the "failure windows" from overlapping
            - I don't totally see why?

## Conclusion

- This proof is far from how neural networks actually work
    - It's closer to a proof of NAND gates as universal computers
