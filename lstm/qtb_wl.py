# Word-level implementation of an RNN for text generation.
#
# Adapted from:
#   - https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#example-an-lstm-for-part-of-speech-tagging
#   - https://github.com/fastai/fastai/blob/master/courses/dl1/lesson6-rnn.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSTMGenerator(nn.Module):
    """
    Simple LSTM RNN for text generation.
    """
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # Word embeddings.
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        # The two parameters correspond to:
        #   first: input dimensions
        #   second: output dimensions (for hidden states)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size)

        # The linear layer that maps from hidden state space to output space.
        self.hidden2output = nn.Linear(self.hidden_size, self.vocab_size)

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
        embeds = self.embedding(sentence)

        # Feed-forward, updating the hidden layer in the process.
        # (Don't quite understand the dimensionality of the embeddings view...?)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1))

        # Project the hidden layer into output space.
        output_space = self.hidden2output(lstm_out.view(len(sentence), -1))

        # Generate a probability distribution for the output.
        output_scores = F.log_softmax(output_space, dim=1)
        return output_scores


def prepare_train_test(seq):
    """
    Chop a sequence up into 8-gram training pairs.
    """
    stop = len(train_ix)
    return [(train_ix[i:i+8], train_ix[i+1:i+9]) for i in range(0, stop, 8) if i + 9 < stop]


def get_next_ix(seed, model, to_ix):
    """
    Use a model to get the next character for the string `seed`.
    """
    seed_to_ix = torch.tensor([to_ix[char] for char in seed], dtype=torch.long)
    prob_dist = model(seed_to_ix)
    # return torch.argmax(prob_dist[-1]).item()
    return torch.multinomial(prob_dist[-1].exp(), 1).item()


if __name__ == '__main__':

    import time
    import os

    # Hyperparameters.
    EMBEDDING_SIZE = 256
    HIDDEN_SIZE = 256

    # Import training and test data.
    train_fpath = os.path.join('data', 'nietzche', 'train.txt')
    test_fpath = os.path.join('data', 'nietzche', 'test.txt')

    with open(train_fpath) as train_file:
        train_text = train_file.read()
    with open(test_fpath) as test_file:
        test_text = test_file.read()

    train_text = train_text.split()
    test_text = test_text.split()

    # Prepare vocabulary (universe of possible characters).
    vocab = sorted(list(set(train_text + test_text)))
    vocab_to_ix = {c: i for i, c in enumerate(vocab)}

    # Encode the train/test data according to each character's index in the vocabulary.
    train_ix = [vocab_to_ix[char] for char in train_text]
    test_ix = [vocab_to_ix[char] for char in test_text]

    # Format training and test data to include targets.
    training_data = prepare_train_test(train_ix)
    testing_data = prepare_train_test(test_ix)

    # Temporarily restrict the training data to a smaller sample.
    # training_data = training_data[:1000]

    train_step = round(len(training_data)/10)
    training = [training_data[i:i+train_step] for i in range(0, len(training_data), train_step)]

    model = LSTMGenerator(len(vocab), EMBEDDING_SIZE, HIDDEN_SIZE)
    loss_func = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    start_time = time.time()

    for step in range(1):
        losses = []
        for epoch, training_data in enumerate(training):
            total_loss = 0
            for (batch, (sentence, target)) in enumerate(training_data):
                # Clear gradients.
                model.zero_grad()

                # Clear hidden state of the LSTM.
                model.hidden = model.init_hidden()

                # Prepare inputs.
                x = torch.tensor(sentence, dtype=torch.long)
                y = torch.tensor(target, dtype=torch.long)

                # Run the forward pass.
                y_pred = model(x)

                # Compute the loss and gradients, and update params.
                loss = loss_func(y_pred, y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if batch % 100 == 0:
                    print('Epoch %d : Batch %d : Loss %s' % (epoch, batch, loss.item()))

            losses.append(total_loss/len(training_data))

            print('Losses: {losses}'.format(losses=losses))

    end_time = time.time()

    print('Training took {secs} seconds'.format(secs=str(end_time-start_time)))
    print('Final loss: {loss}'.format(loss=losses[-1]))
    print('Losses: {losses}'.format(losses=losses))

    # Generate some text!
    output = seed = ['What', 'is', 'the']
    for i in range(400):
        char_ix = get_next_ix(seed, model, vocab_to_ix)
        char = vocab[char_ix]
        output.append(char)
        seed[1:].append(char)

    print(output)
