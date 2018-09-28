# Week 1: Introduction and Linear Regression

## Framing 

- Supervised learning
    - Combine inputs to make predictions
    - **Labels**: True value that we're trying to predict
    - **Features**: Input variables describing the data (`Xi`)
        - Data representation
    - **Example**: A particular instance of data (`X`)
        - Can be **labelled** (for training) or **unlabelled** (for prediction)
    - **Model**: Function mapping inputs to predicted labels (`Y'`)

### Key ML Terminology

- **Regression**: Predict continuous values

- **Classification**: Predict discrete valueso

## Descending into ML

### Linear regression

- Describes linear relationship between inputs and outputs
    - In Algebra 1: `y = mx + b`
    - In ML: `y' = b + w1*x1`
        - Bias sometimes referred to as `w0`
        - Multivariate, e.g. `y' = b + w1*x1 + w2*x2 + w3*x3`

### Training and loss

- **Training**: Finding good values for the weights and bias based on examples
    - "Empirical risk minimization"
    - Goal: Find combination of weights and bias that minimize loss

- Common loss function is **squared (L2) loss**
    - Loss for the entire training set is _mean square error_
        - A common loss function, but not the only one

## Reducing Loss

- Gradient descent is similar to a game of "Hot and Cold"
    - Nice metaphor!

- Algorithm overview:
    - Start with a guess: Initialize `w` and `b` with either random values, or zero values
    - Compute loss
    - Update parameters based on loss
    - Repeat until the model _converges_
        - Meaning: loss either A) stops changing or B) changes at some
          predefined (very slow) rate 

### Gradient Descent

- Claims that for regression problems, plot of loss vs. weights will always be convex
    - Why is that...?

- Algorithm in depth:
    - Start with a random point (random weights)
    - Calculate the _gradient_ of the loss
        - In one variable, this is the derivative of the curve
        - In multiple variables, this is the vector of partial derivatives with
          respect to the weights

- Learning rate
        - Add a scaled portion of the gradient's magnitude to the starting point
            - _Learning rate_ represents this scale factor
            - Learning rate is an example of a _hyperparameter_, a variable in
              a model that is directly alterable by the programmer
        - Ideal learning rate is related to the "flatness" of the loss function
            - Generally: Inverse of the Hessian (matrix of second partial
              derivatives)

- Why does gradient always point in the direction of the steepest increase in
  the loss function...?

### Optimizing Learning Rate

- 1.60 is the sweet spot!

### Reducing Loss: Stochastic Gradient Descent

- **Batch**: Number of examples you use to calculate the gradient in one
  iteration
    - So far we've assumed the entire dataset; this is often unworkable in
      practice!
        - Large batches often contain redundant data

- **Stochastic gradient descent**: Use only one example per batch
    - Works, but is noisy

- **Mini-batch stochastic gradient descent**: Compromise between full-batch and
  SGD
    - Random sample from the training data
