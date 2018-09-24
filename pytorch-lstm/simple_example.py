# Adapted from https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

import torch
import torch.nn as nn

torch.manual_seed(1)

lstm = nn.LSTM(3, 3)  # Input/output dimensions are 3
inputs = [torch.randn(1, 3) for _ in range(5)]  # Random sequence

# Hidden state.
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))

for i in inputs:
    # Step through the sequence one element at a time.
    out, hidden = lstm(i.view(1, 1, -1), hidden)

print('Input:')
print(inputs)
print()
print('Output:')
print(out)
print()
print('Hidden state:')
print(hidden)
