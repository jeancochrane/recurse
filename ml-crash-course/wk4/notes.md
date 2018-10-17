# Week 4: Classification

## Logistic Regression

### Calculating a Probability

- Method for calculating a probability distribution
    - Can use this either for traditional probability tasks, or for
      binary classification

- Sigmoid function ensures that output always lies between 0 and 1
    - But this a probability distribution does not make...? I suppose  if it's
      for a binary question, then yeah

- `z`: Weighted input, or _log odds_
    - Inverse of the sigmoid: Log of the probability of success divided by
      probability of failure
    - What's the deal here...? I get that log-odds is the inverse of the
      sigmoid, but what does that have to do with `z`?

```python
z = log(P / 1 - P)
```

### Model Training

- Loss function for logistic regression is _log loss_

```python
def log_loss(a, y):
    return -sum(yi*np.log(ai) + (1-yi)*np.log(1-ai) for ai, yi in zip(a, y))
```

- Minimizing log loss -> maximum likelihood estimate

#### Regularization in Logistic Regression

- Important in logistic regression, since asymptote of sigmoid can drive loss
  towards 0 in high dimensions
    - Could we see a rigorous proof of this please...?

## Classification

### Thresholding

- Classification/decision threshold is necessary to make a binary prediction out
  of a logistic regression value
    - This is a hyperparameter, and depends on the problem domain

### True/False, Positive/Negative

- Four fundamental outcomes in a _confusion matrix_:
    1. True positive
    2. False positive
    3. False negative
    4. True positive

### Accuracy

- Fraction of predictions the model gets right

```
accuracy = num_correct / num_predictions
```

- In terms of true/false positives:

```
accuracy = (tp + fp) / (tp + tn + fp + fn)
```

- Accuracy alone is not enough when working with _class-imbalanced data_ (like
  disease screening tests)

### Precision and Recall

- **Precision**: What proportion of the positive identifications were correct?

```
precision = tp / (tp + fp)
```

- **Recall**: What proportion of actual positives were correctly identified?

```
recall = tp / (tp + fn)
```

- There's often a tradeoff between precision and recall
    - Metrics like **F1 score** help balance this tradeoff

### ROC and AUC

- ROC (receiver operating characteristic) curve: Plot performance of model at
  _all classification thresholds_
    - Two variables: true positive rate (recall) vs. false positive rate
    - I don't really get where the classification thresholds come in here...?
      Why are they not a variable? Or: If they are, why is the plot in two
      dimesnions?

- AUC: Area under the ROC curve
    - Generally, are true positives relatively to the right of true negatives?
    - 1 indicates 100% correct predictions
    - 0 indicates 100% wrong predictions
    - Advantages:
        - Scale-invarient
        - Classification-threshold-invariant
            - E.g.: How good is the model, in general (irrespective of the
              particular threshold chosen)?
    - Sometimes these advantages are disadvantages!

### Prediction bias

- How different is the average of the predictions from the average in the
  labels?
    - E.g. if 1% of emails are spam, a model should on average predict that
      about 1% of emails it's shown are spam, too

- Don't just add a calibration layer to fix this! Instead, look at root causes:
    - Not enough features
    - Noisy data
    - Buggy pipeline
    - Biased training sample
    - Overly strong regularization
