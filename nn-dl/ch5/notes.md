# Chapter 5: Why are deep neural networks hard to train?

- More layers allows for more abstraction
    - Why should this be true, formally?
    - Nielsen points to a paper by Bengio, et. al.

- General concept: Early layers learn well, later layers become "stuck"

## The vanishing gradient problem

- To track speed of learning, use the measure of the gradient with respecto the
  bias, `dC/db`
    - Magnitude of these vectors is a (rough) global measure of speed of learning

- Revealed pattern: Magnitudes of gradient vectors grow larger as the layers
  increase
    - Fascinating that the _shape_ of the learning curves is so consistent
      between layers (i.e. learning gets faster or slower with progressive epochs
      following similar patterns), the only difference being the speed at any
      moment in time

- Vanishing gradient problem: In some deep neural networks, the gradient tends
  to shrink as we move backward through the hidden layers
    - The inverse is the _exploding gradient problem_, in which the gradient
      gets larger while moving backward
    - Generalization: The gradient in deep neural networks is _unstable_

## What's causing the vanishing gradient problem? Unstable gradients in deep neural nets

- Fundamental problem isn't about vanishing or exploding -- it's that the
  gradient in the early layers is the product of terms from all the later layers
