# Import required libraries
import torch
import torch.nn.functional as F
from dataset import Xtr, Ytr, Xdev, Ydev  # Training and dev datasets
import os

# Load model weights: resume from saved weights if available, else initialize
if os.path.exists('trained_weights.pt'):
    checkpoint = torch.load('trained_weights.pt')
    C = checkpoint['C']   # Character embedding matrix
    W1 = checkpoint['W1'] # First layer weights
    b1 = checkpoint['b1'] # First layer bias
    W2 = checkpoint['W2'] # Second layer weights
    b2 = checkpoint['b2'] # Second layer bias
    parameters = [C, W1, b1, W2, b2]
else:
    from weights_biases import C, W1, b1, W2, b2, parameters

# Training hyperparameters
max_steps = 500000      # Number of training steps
batch_size = 32         # Minibatch size
lossi = []              # Track log-loss for plotting
stepi = []              # Track steps for plotting
ud = []                 # Track update statistics

# Training loop
for i in range(max_steps):
  # Minibatch construction: sample random indices
  ix = torch.randint(0, Xtr.shape[0], (batch_size,))

  # Forward pass: embed, hidden layer, output logits
  emb = C[Xtr[ix]] # (batch_size, block_size, embedding_dim)
  h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # Hidden layer
  logits = h @ W2 + b2 # Output logits
  loss = F.cross_entropy(logits, Ytr[ix]) # Compute loss

  # Backward pass: zero gradients, compute gradients
  for p in parameters:
    p.grad = None
  loss.backward()

  # Update parameters using gradient descent
  lr = 0.1 * (0.95 ** (i // 10000)) # Learning rate decay
  for p in parameters:
    p.data += -lr * p.grad

  # Print training progress every 50,000 steps
  if i % 50000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  # Track log-loss and update statistics
  lossi.append(loss.log10().item())
  with torch.no_grad():
    ud.append([(lr*p.grad.std() / p.data.std()).log10().item() for p in parameters])

# Evaluate the loss on the training set
emb = C[Xtr]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr)
print("Training split loss:")
print(loss)

# Evaluate the loss on the dev set
emb = C[Xdev]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print("Dev split loss:")
print(loss)

# Save trained weights for future use
torch.save({
    'C': C,
    'W1': W1,
    'b1': b1,
    'W2': W2,
    'b2': b2
}, 'trained_weights.pt')


