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
    - Instead, use encodings

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
