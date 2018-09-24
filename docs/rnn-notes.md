# RNN Notes

## [Text Generation with Recurrent Neural Networks](https://blog.paperspace.com/recurrent-neural-networks-part-1-2/)

### Background

- RNNs: family of nets designed specifically for **sequential data**
    - e.g. speech, time series, text, video...

- One approach: _how likely is a given sentence?_
    - Being able to generate these probability distributions will later allow us
      to generate text

### World-level language models

- Model a sentence `S` as a word vector of `T` words:

```python
S = (w1, w2, w3, ..., wT)
```

- Each word `wi` is part of a vocabulary `V`:

```python
V = (v1, v2, v3, ..., vn)
```

- Key insight: Using the chain rule, the probability of a word occurring at time
  `t` can be represented as a function of the previous words in the sentence:

```python
# Chain rule definition
P(A,B) = P(A) * P(B|A) 

# Chain rule applied to word vectors
# (wt is the current word, and w<t is the set of all words previous to time t)
P(S) = product(P(wt|w<t) for wt in S)
```

- Formal definition of  a language model: predict the next word `wt+1` given the
  preceding words `w<t` at each time step `t`
    - What does "at each time step `t`" mean here? As in, for each `t`, we need
      to compute `wt+1`? Seems expensive...


### Recurrent Neural Networks

- Key insight: humans predict words from context, too!
    - e.g. "Paris is the largest city in..."

- "Recurrent": same operation is performed on each element of the sequence, with
  output depending on the combination of current time step and everything that
  came before
    - i.e. Output at time `t` becomes input at time `t+1`

- Architecture diagram of the loop:
    - Variables:
        - `N`: A node of the loop
        - `o`: Output
        - `h`: "Memory state" (records what happened in the network up to the
          current state)
        - `x`: Time step

```
        o1        o2                      on
        ^         ^                       ^ 
        |         |                       |
--h0--> N --h1--> N --h2--> ... --hn-1--> N --hn-->
        ^         ^                       ^
        |         |                       |
        x1        x2                      xn
```

- Combination of inputs and outputs leads to three common applications:
    1. Sequential input -> sequential output (translation, tagging)
    2. Sequential input -> single output (classification, sentiment analysis)
    3. Single input -> sequential output (image captioning)

### Recurrent Language Models

#### Word Embedding

- Problem: How to represent the data?
    - One solution: one-hot encoding
        - Each word in the vocabulary `V` is represented as a sparse binary
          vector `wi`
        - Claims that this minimizes "prior knowledge"... why?

- Input to the model is a sequence of `T-1` one-hot vectors -- multiply each by
  a weight matrix `E` to get a sequence of continuous vectors
    - (Note that since the vectors are sparse, in practice we usually just slice
      the weight matrix instead of performing a full multiplication, for
      performance)
    - What is the weight matrix, and what are the "continuous" vectors...?

```python
# Weight the input vectors to produce a sequence of continous vectors
xj = E.T * wj
```

#### Forward Path

- General algorithm:
    1. Initialize memory vector `h` to zero
    2. Input at first time step is special token `</s>` to denote the start of
       a sentence
    3. Output is the probability of every word in V given `</s>`
    4. Update memory vector `h` and send to the next time step
    5. Repeat!

- Note that the output is a _softmax layer_ (sequence whose sum is 1) and also
  a _probability distribution_ over all `V`, where element `i` represents the
  probability of the word `Vi` being the next word to appear in the sentence

#### Backward path

- Missing piece: Loss function for learning the appropriate weights
    - Negative log probability that the model assigns to the correct output

```python
loss = -log(P(S))

# Apply the chain rule
loss = -log(product((P(wt|w<t) for wt in S)))

# Log of a product is equal to the sum of the logs
# (xt+1 is the actual next word)
loss = -sum(log(P(wt = xt+1)))
```

- Blog post claims:
    - "With the loss defined and given that the whole system is differentiable,
      we can backpropagate the loss through all the previous RNN units and
      embedding matrices and update its weights accordingly."
        - What's going on here...?
    - "We can backpropogate the loss through all the previous RNN"
