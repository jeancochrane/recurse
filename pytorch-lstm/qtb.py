# Adapted from https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#example-an-lstm-for-part-of-speech-tagging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMGenerator(nn.Module):
    """
    RNN implementing 8-gram character-level language model.
    """
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super().__init__()
        # Save the hidden size for use in init_hidden.
        self.hidden_size = hidden_size

        # Word embeddings.
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # The two parameters correspond to:
        #   first: input dimensions
        #   second: output dimensions (for hidden states)
        self.lstm = nn.LSTM(embedding_size, hidden_size)

        # The linear layer that maps from hidden state space to output space.
        self.hidden2output = nn.Linear(hidden_size, vocab_size)

        # Initialize hidden state space.
        self.hidden = self.init_hidden()

    def init_hidden(self):
        """
        Initialize or reset the hidden state space.
        """
        # Axes semantics are:
        # (num_layers, minibatch_size, hidden_dim)
        # Still don't quite understand these dimensions...?
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))

    def forward(self, sentence):
        """
        Feed-forward the RNN.
        """
        # Retrieve word embeddings.
        embeds = self.embeddings(sentence)

        # Feed-forward, updating the hidden layer in the process.
        # (Don't quite understand the dimensionality of the embeddings view...?)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1))

        # Project the hidden layer into output space.
        output_space = self.hidden2output(lstm_out.view(len(sentence), -1))

        # Generate a probability distribution for the output.
        output_scores = F.log_softmax(output_space, dim=1)
        return output_scores


def prepare_sequence(seq, to_ix):
    """
    Turn a sequence of words, `seq`, into a tensor using the vocab index `to_ix`.

    TODO: Could probably become a classmethod on a Sequence datatype?
    """
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


if __name__ == '__main__':

    # Hyperparamters.
    EMBEDDING_SIZE = 32
    HIDDEN_SIZE = 32

    # TODO: Prep training data.
    training_data = []
    word_to_ix = {}
    vocab = set(word_to_ix.keys())

    model = LSTMGenerator(len(vocab), EMBEDDING_SIZE, HIDDEN_SIZE)
    loss_func = nn.NLLLoss(len(word_to_ix.keys()))
    optimizer = optim.SGD(model.paramters(), lr=0.1)

    import time
    start_time = time.time()

    losses = []
    for epoch in range(10):
        total_loss = 0
        for sentence, target in training_data:
            # Clear gradients.
            model.zero_grad()

            # Clear hidden state of the LSTM.
            model.hidden = model.init_hidden()

            # Prepare inputs.
            x = prepare_sequence(sentence, word_to_ix)
            y = prepare_sequence(target, word_to_ix)

            # Run the forward pass.
            y_pred = model(x)

            # Compute the loss and gradients, and update params.
            loss = loss_func(y_pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        losses.append(total_loss)

    end_time = time.time()

    print('Training took {secs} seconds'.format(secs=str(start_time-end_time)))
    print('Final loss: {loss}'.format(loss=losses[-1]))
