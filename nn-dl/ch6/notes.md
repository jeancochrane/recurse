# Chapter 6: Deep Learning

- Things we'll cover in this chapter:
    - Convolutional networks
    - RNNs and LSTMs
    - Deep learning

## Introducing convolutional networks

- Motivation: What if there were a way to avoid fully connecting each layer?

- Three key concepts:
    1. Local receptive fields
    2. Shared weights and biases

### Local receptive fields

- Intuition: Instead of representing input data as a flat array, think of it as
  a 2x2 grid -- then, each neuron in the first hidden layer will be connected to
  a small _region_ of input neurons
    - Move the field over by one neuron to generate the field for the next hidden neuron
        - In this case, the _stride length_ is 1; no reason this has to be
          1 though!

### Shared weights and biases

- Each hidden neuron in a layer uses the same weights and bias (!!)

- Intuition: Each neuron in a layer detects the same _feature_ (e.g. a vertical
  line, or a horizontal curve)
    - "Feature" patterns are likely to be useful in many places in an  image

- Special terminology:
    - Map from the input to the hidden layer: _feature map_
    - Weights defining a feature map: _shared weights_
    - Bias defining a feature map: _shared bias_
    - Shared weights and shared bias together: _kernel_ or _filter_

- A complete convolutional layer can detect multiple features, and so has
  multiple feature maps 
    - e.g. A hidden layer with three feature maps might have dimensions `(3 x 24
      x 24)`, which could detect three features across the entire image

- Advantage: Fewer weights (and biases) to keep track of!

### Pooling layers

- Intuition: Condense information from the hidden layer
    - Is a given feature found anywhere in a certain region of the image? (Exact
      positional information doesn't really matter)
        - Implicitly: Relative position of features is more important than
          absolute position

- One common approach: _max pooling_
    - Take the maximum activation in a given region of the hidden layer
    - Pool each feature map output separately

- Another approach: _L2 pooling_
    - Take the root sum of squares of the activations in a given region

### Putting it all together

- Finally layer is a fully-connected layer mapping to every neuron from the
  max-pooled layer
    - I'm having trouble visualizing these final connections...? Are there weights
      here? What do they look like?

## Convolutional networks in practice

- Why does the second network architecture use a 100-neuron sigmoid layer before
  the softmax layer...?

- What's up with the layer dimensions in the second code example (adding
  a second convolutional-pooling layer)...?

### Using rectified linear units

- Nielsen's experiments show that networks with ReLU activations consistently
  outperform sigmoid-based networks
    - For now, only heuristic arguments for ReLU (that it doesn't saturate in
      the limit of large `z`)

#### A note on dropout

- In convolutional networks, dropout only needs to be applied to fully-connected
  layers!
    - To see why, consider the shared weights: Weights in the convolutional
      layer must be applied across the entire image, which offers a built-in
      resistence to overfitting
