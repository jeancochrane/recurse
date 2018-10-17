# Week 3: Representing Data

## Representation

### Feature Engineering

- Main focus of machine learning projects: _representation_
    - How to encode the data?

#### Mapping raw data to features

- **Feature engineering**: Turn raw data into numerical vectors

- _Categorical_ features have a discrete set of values, usually not numeric
    - Define _vocabulary_ of possible values (map to integers)
    - Any value not in the vocabulary goes into an _OOV (out-of-vocabulary)
      bucket_

- _One/multi hot encoding_ is a common integer mapping for categorical features
    - Also works for numeric data that should not be weighted directly, e.g. zip
      codes
    - If the vocabulary is large, consider using a _sparse representation_,
      which will only store nonzero values
        - Note that weights will still be learned for every element in the
          vocabulary! So the weight matrix will still be quite large

### Qualities of Good Features

- Categorical features should **appear frequently** (>5 times) in the dataset

- Try to give features **clear and obvious meanings**
    - More so that people can understand you than for better performance

- Don't mix **special values** (like nulls) with actual data
    - Instead, use encodings (e.g. `quality_rating` and
      `is_quality_rating_defined`)

- Don't **tightly couple** to upstream data representations

### Cleaning Data

#### Scaling feature values

- Convert from a natural range to a standardized range (usually `[0, 1]` or
  `[-1, 1]`)

- Advantages:
    - Gradient descent converges faster
        - Why...?
    - Avoids floating-point precision limits ("NaN trap")
    - Makes comparisons between different features more reasonable

#### Dealing with outliers

- Two approaches introduced here:
    - Use logs
    - Clip a maximum value

#### Binning

- Numerical features that don't have direct linear relationships to output (e.g.
  long/lat) can be _binned_
    - Essentially the same as one-hot encoding, just with a numerical range
      instead of a category

#### Scrubbing

- In essence: Removing data that is "wrong" in some degree
    - Features are misspelled or misentered
    - Labels are incorrect
    - Duplicates
    - Missing data

- Duplicates + missing data can be handled programmatically, but incorrect
  features/labels are a bit harder
    - Usually need to do some manual sniffing

## Feature Crosses

### Encoding nonlinearity

- One way to fudge nonlinear boundaries: Create linear combinations of features
  to encode nonlinearity

### Crossing one-hot vectors

- Crossing one-hot vectors is better thought of as _logical conjunctions_
    - How is this represented...?
        - Seems like it's a vector with length `len(A) * len(B)` with
          a 1 corresponding to the intersection between the two vectors

## Regularization

- Note that feature crosses and nonlinear transformations can produce
  overfitting! (Bias/variance tradeoff)

### L2 Regularization

- Intuition: Penalize overly-complicated models

- _Structural risk minimization_: Minimize loss + complexity
    - Optimization algorithm now has two terms: loss term + complexity term

- Two ways to conceive of "complexity":
    - Function of weights
        - i.e. High absolute value weight -> high complexity
    - Function of total number of features with nonzero weights
        - i.e. More features with nonzero weights -> more complexity

- L2 Regularization: Sum of squares of the feature weights

```python
def l2_regularization(w):
    return sum(wi**2 for wi in w)
```

- In this definition, _outlier weights_ can have a disproportionate impact on
  model complexity

- Two effects on a model:
    1. Encourage weight values close to 0
    2. Encourages mean 0 gaussian distribution of weights
        - Why...?

### Lambda

- AKA _regularization rate_: Tuning hyperparameter for complexity loss

- High lambda -> narrower distribution of weights; low lambda -> flatter distribution
