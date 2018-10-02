# Week 2: Training Models

## Intro to TensorFlow

- What's the distinction between steps/periods/batch sizes...?
    - I think steps = epoch; periods is just a convention for this example

- Enjoy how playing with learning rate during training gives a "feel" for the
  tradeoff!
    - Low learning rate -> RMSE changes too slowly
    - High learning rate -> high variance in RMSE, doesn't converge

## Generalization

- Overfitting: Low training loss, high testing loss
    - Caused by _excessive complexity_

- Central tension of ML: Fit the data well, but also as simply as possible

- Less complex model -> more likely that a good result will generalize

- Theory provides _generalization bounds_: Quantifying capacity to predict new data (in theory)
    - Complexity
    - Training performance

- Empirical performance measures
    - E.g. train/test split

- Assumptions underlying train/test split:
    1. Draw samples _independently and identically_
    2. Distribution is stationary (doesn't change w/in dataset...?)
    3. Draw samples from the same distribution

- Reasons to violate assumptions, e.g.:
    - If the model changes depending on what it has already produced (ads)
    - If the distribution of the behavior changes (seasonal purchases)

## Training and Test Sets

- Test sets must meet two conditions:
    1. Large enough to be statistically significant
    2. Representative of the dataset

- Watch out for testing data leaking into training data!
    - Can cause results to appear better than they are
    - E.g. if there are duplicates between datasets

- Small training samples require low learning rates and batch sizes, or else the
  model risks being unable to find the minimum

- Problem with this approach: We're using test dataset to tweak hyperparameters!
    - We'll want to do something different, like...

## Validation

- To avoid overfitting: Partition data into _three_ subsets
    - Training/testing/validation

- Also not a panacea! Validation and test sets eventually "wear out" after much
  use
    - Hence, try to always continually collect new data

### Programming Exercise

- I see why the sampling is screwed up on the map, but I don't actually think
  the distributions of features are very different between training and
  validation...? Am I seeing something wrong?

- Key point: Debugging in ML is usually _data debugging_, not _code debugging_

- Test performance is actually lower than training performance! Isn't this a bad
  sign though...?
