import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


torch.manual_seed(1)

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

# Shakespeare's Sonnet 2.
test_sentence = """
    When forty winters shall besiege thy brow,
    And dig deep trenches in thy beauty's field,
    Thy youth's proud livery so gazed on now,
    Will be a totter'd weed of small worth held:
    Then being asked, where all thy beauty lies,
    Where all the treasure of thy lusty days;
    To say, within thine own deep sunken eyes,
    Were an all-eating shame, and thriftless praise.
    How much more praise deserv'd thy beauty's use,
    If thou couldst answer 'This fair child of mine
    Shall sum my count, and make my old excuse,'
    Proving his beauty by succession thine!
    This were to be new made when thou art old,
    And see thy blood warm when thou feel'st it cold.
""".split()

# Build a list of tuples, recording each word with the two words before it
trigrams = [([test_sentence[i], test_sentence[i+1]], test_sentence[i+2])
            for i in range(len(test_sentence) - 2)]

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_layer = nn.Linear(context_size * embedding_dim, 128)
        self.output_layer = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = self.output_layer(F.relu(self.hidden_layer(embeds)))
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_func = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(20):
    total_loss = 0
    for context, target in trigrams:
        # Prep the inputs.
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # Clear out gradients.
        model.zero_grad()

        # Run the forward pass.
        log_probs = model(context_idxs)

        # Compute the loss.
        loss = loss_func(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        # Do backward pass and update gradient.
        # (Why doesn't the optimizer need to know the loss...?)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    losses.append(total_loss)

print(losses)
