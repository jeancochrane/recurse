# RNN Notes

## Text Generation with Recurrent Neural Networks ([Link](https://blog.paperspace.com/recurrent-neural-networks-part-1-2/))

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
    3. Single input -> sequential output (image captioning, text generation...?)

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
    3. Output is the probability of every word in `V` given `</s>`
    4. Update memory vector `h` and send to the next time step
    5. Repeat!

- Note that the output is a _softmax layer_ (sequence whose sum is 1) and also
  a _probability distribution_ over all `V`, where element `i` represents the
  probability of the word `Vi` being the next word to appear in the sentence

#### Backward path

- Missing piece: Loss function for learning the appropriate weights
    - Negative log probability that the model assigns to the correct output:

```python
loss = -log(P(S))

# Apply the chain rule
loss = -log(product((P(wt|w<t) for wt in S)))

# Log of a product is equal to the sum of the logs
# (xt+1 is the actual next word)
loss = -sum(log(P(wt = xt+1)) for wt in S)
```

- Blog post claims:
    - "With the loss defined and given that the whole system is differentiable,
      we can backpropagate the loss through all the previous RNN units and
      embedding matrices and update its weights accordingly."
        - What's going on here...?

### Generalization to Unseen N-grams

- As far as I can tell, "generalize to unseen n-grams" means measuring how the
  model performs on unseen data (validation)

- Think about the model as a composite of two functions `f` and `g`
    - `f`: Map a sequence of preceding `n-1` words to continuous vector space,
      creating memory vector `h`
      (...?)
    - `g`: Map memory vector `h` to probability distribution
        - First, affine transformation
            - Multiply by weight matrix, add to bias vector
        - Then, softmax normalization
            - Convert the output to a valid probability distribution

```python
# How might we model f in code...?

# g:
# (h is the memory vector; U is the weight matrix; c is the bias vector)
g = lambda h: softmax(U.T * h + c)
```

- I don't quite get this...?
    - Since each element of the output vector is a vector multiplication of `h`
      with the corresponding row of `U.T`, "This means that the predicted
      probability of the model for the i-th word of the dictionary is
      proportional to how aligned the i-th column of `U.T` is with the context
      vector `h`."
        - I think perhaps that for both vectors `U.T` and `h`, the i-th element
          in the vector corresponds to the same word in the dictionary...? But
          I still don't get what `U.T` is?

- Note that the equation above implies that if two sequences are usually
  followed by a similar set of words, their context vectors `h1` and `h2` must
  necessarily be similar
    - "Similar" = "maps to nearby points in the context vector space"
    - This property allows the model to generalize well!
        - e.g. Bigram model of "three teams/four teams/four groups", where
          "three groups" will receive a high probability because the context
          word `three` is projected to a point in the context space close to
          `four`
    - Adheres to the "distributional hypothesis of language": Words that occur
      in similar contexts tend to have similar meanings

- How are the "context space" and the "word space" different...?

### Text Generation

- How to create a new sequence? Quick sketch:
    1. Initialize the context vector `h0` randomly
    2. Unroll the RNN; at each time step, sample one likely word from the
       output distribution
    3. Feed the word back into the RNN, rinse, repeat

- Common problem: Corpus often doesn't have enough text
    - One solution: _pre-train_ the model on a generic corpus (e.g. Gutenberg
      corpus) to give it a basic understanding of words and context, then train
      on a more focused corpus

## Understanding LSTM Networks ([Link](https://colah.github.io/posts/2015-08-Understanding-LSTMs/))

### The Problem of Long-Term Dependencies

- Basic problem: Relevant context is sometimes provided far away from the
  current time step of a sequence
    - E.g. if a text is about growing up in France, "I speak fluent..." would
      naturally lead a human to the token `French`, but the relevant token
      `France` may be very far away in the sequence, and the model will have
      forgotten it!
    - RNNs should _theoretically_ be able to deal with this problem, but in
      practice the often don't -- why...?

### LSTM Networks

- "LSTM" = "Long Short Term Memory"
    - Capable of learning long-term dependencies

- Add **three extra layers** to every cell

### Core Idea Behind LSTMs

- **Cell state**: Persists across the chain, with minor linear interactions

- Can remove or add info to cell state, regulated by _gates_
    - Gate: sigmoid neural net layer + pointwise multiplication
        - Sigmoid layer outputs a float in the range `[0, 1]`, which weights how
          much info should get through
        - LSTM has three gates

### Step-by-step Walkthrough

1. Decide what info to throw away from the cell state
    - "Forget gate layer" controls this step
        - e.g. If the subject has changed, forget the component of the cell
          state that stores info about the subject's gender for pronouns

2. Decide what new info to store in the cell state
    - Two steps:
        1. "Input gate layer" (sigmoid) decides which values to update
            - Why is this distinct from the forgetting step...?
        2. tanh layer (variant of sigmoid with the range `[-1, 1]`) produces
           a vector of candidate values `~Ct` that could be added to the state
    - E.g. Add the gender of the new subject to the cell state

3. Update the old cell state `Ct-1` to the new state `Ct`
    - Two steps:
        1. Multiply old state `Ct-1` by `ft` (forget gate layer)
        2. Add `it * ~Ct` to the cell state (new candidate values, scaled by how
           much we want to update each value)
    - This is where the gender gets forgotten and relearned

4. Determine output
    - Filtered version of the cell state:
        1. Apply sigmoid layer to decide which parts of the cell state to output
           (I suppose as distinguished from which parts to continue storing, as
           in steps 1 and 2...?)
        2. Put the cell state `Ct` through a tanh layer to push the values to be
           bewteen -1 and 1, and multiply by the output of the sigmoid gate
    - E.g. If we just saw a subject, output information relevant to a verb (like
      singular or plural) which will probably be the part of speech that follows

### Variations

- Gated Recurrent Units (GRUs) seem relevant to me
    - Combine forgetting and input gates into a single "update" gate, among
      other simplifications
    - Popular as of 2015 

## The Unreasonable Effectiveness of Recurrent Neural Networks ([Link](https://karpathy.github.io/2015/05/21/rnn-effectiveness/))

- "Vanilla" neural networks are usually limited to a single input and a single
  output; this is the primary advantage of RNNs

- Even single inputs can sometimes be modelled sequentially!
    - E.g. Steering attention around an image

- Basic function of an RNN can be represented by a `step` method, e.g.:

```python
rnn = RNN()
y = rnn.step(x)  # x is input vector; y is output vector
```

- A simple, pseudo-implementation of `step`
    - Three parameters, `W_hh`, `W_xh`, `W_hy`
    - Hidden state `self.h`, initialized to the zero vector
    - `np.tanh` squashes activations to the range `[-1, 1]`

```python
class RNN(object):
    def step(self, x):
        # Update hidden state.
        self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
        # Compute the output vector.
        return np.dot(self.W_hy, self.h)
```

### Character-Level Language Models

- I don't quite understand the business with the "hidden layer", which seems to
  be the memory vector...? In the diagram heading up this section, it's not the
  same length as the input vector! What gives?

- Elucidation of training steps:
    - Each output layer is paired with a target
    - Correct target should have high confidence; all others should be lower
        - If not, run backpropogation algorithm + paramter update to adjust
          weights
            - What's the role of the backpropogation algorithm here...? Has to
              do with the chain rule somehow?

## PyTorch Tutorial notes

- Tensors are like matrices, but they can be indexed in more than two dimensions
  (!)
    - Essentially, a generalization of matrices beyond two dimensions
    - Easy way to think about this is in terms of dimensions:
        - Index a vector (1D tensor) -> scalar
        - Index a matrix (2D tensor) -> vector
        - Index a 3D tensor -> matrix

- Two handy operations:
    - `torch.cat`: Concatenate two tensors (defaults to axis `0`, or rows)
    - `tensor.view`: Reshape a tensor

- Why would the embedding dimension (`embedding_dim`) be different than the
  number of embeddings (`num_embeddings`) in `torch.nn.Embedding`?

- Seems like one difference between backpropogation and parameter updates is:
  backpropogation determines the right gradient; parameter update acutally
  implements it...? Is that right?

## Word Embeddings ([Link](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html#sphx-glr-beginner-nlp-word-embeddings-tutorial-py))

- Sparse word vectors (like the ones covered in the tutorial above) have a
  few major drawbacks:
    - Must store and manage potentially enormous vectors (one position for each
      word in the vocabulary)
    - Doesn't store semantic meaning

- Alternative: use _dense vectors_ to store semantic meaning
    - e.g. `mathematician` could be represented as:

```python
embeddings = ['can_run', 'likes_coffee', 'majored_in_physics']
qmath = [2.3, 9.4, -5.5]
```

- Then, comparison can be achieved with cosine distance (so distance ranges
  `[-1, 1]`):

```python
similarity = lambda x1, x2: (embed(x1) * embed(x2)) / (len(embed(x1))
* len(embed(x2)))
```

- One-hot vectors can be considered from the perspective of dense vectors --
  they simply assign each word its own semantic meaning!

- Instead of assigning these manually, we let the model "learn" the embeddings,
  and treat the number of embeddings as a parameter
