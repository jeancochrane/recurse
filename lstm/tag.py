# Adapted from https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#example-an-lstm-for-part-of-speech-tagging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSTMTagger(nn.Module):
    """
    Class for the part-of-speech tagging module.
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The two parameters correspond to:
        #   first: input dimensions
        #   second: output dimensions (for hidden states)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space.
        # (Not sure what this does...?)
        # (Seems from `forward` like maybe it projects hidden state layer to
        # the dimensions of the tag layer, so that the output is interpretable?
        # Would help explain why outputs are returned in the hidden layer dimensions,
        # rather than the input dimensions.)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        # Initialize hidden state space.
        self.hidden = self.init_hidden()

    def init_hidden(self):
        """
        Initialize or reset the hidden state space.
        """
        # Axes semantics are:
        # (num_layers, minibatch_size, hidden_dim)
        # Still don't quite understand these dimensions...?
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        """
        Feed-forward the LSTM.
        """
        # Retrieve word embeddings.
        embeds = self.word_embeddings(sentence)

        # Feed-forward!
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1),
                                          self.hidden)

        # Project the hidden layer output into tag space.
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))

        # Use softmax to generate a probability distribution for the output.
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def prepare_sequence(seq, to_ix):
    """
    Turn a sequence of words, `seq`, into a tensor using the dictionary index `to_ix`.
    """
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def prepare_prediction(dist, tags):
    """
    Given an output tensor, format and print the model's prediction for the most
    likely parts of speech.
    """
    return tuple(tags[(predicted == max(predicted)).nonzero().item()] for predicted in dist)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

# Create a dictionary for the vocabulary, and map each word to an index.
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

# Create a dictionary for the target tags, mapping each to an index.
TAGS = ["DET", "NN", "V"]
tag_to_ix = {tag:ix for ix, tag in enumerate(TAGS)}

# Set dimensions for the embeddings and hidden state.
# The example says that these will usually more like 32 or 64-dimensional, but
# that keeping them small makes introspecting them easier.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

# Train the model:
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Inspect the scores prior to training.
# Note that torch.no_grad() enforces that gradient descent is never called (requries_grad
# will always be overrided to False, even when set to True) to reduce memory overhead.
# That's appropriate here because we know we won't incur backpropogation,
# since we're not training.
with torch.no_grad():
    print('Testing sentence:')
    print(training_data[0][0])
    print()

    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    print('Initial scores:')
    print(tag_scores)
    print(prepare_prediction(tag_scores, TAGS))
    print()

# Run 300 epochs, since this is toy data.
# (What's an epoch...? Is it different from an iteration of the RNN?)
for epoch in range(300):
    for sentence, tags in training_data:
        # Clear out gradients, since PyTorch accumulates them automatically.
        model.zero_grad()

        # Clear the hidden state of the LSTM, accumulated from the last sentence.
        model.hidden = model.init_hidden()

        # Prepare inputs for the network.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Run the forward pass.
        # (We don't have to call `forward` directly -- nn.Module understands that
        # `forward` should be assigned to the __call__ method of this instance!)
        tag_scores = model(sentence_in)

        # Compute the loss and gradients, and update parameters.
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# Inspect the scores after training.
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print('Scores after training:')
    print(tag_scores)
    print(prepare_prediction(tag_scores, TAGS))
