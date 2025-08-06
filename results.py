import torch
import torch.nn.functional as F
from dataset import itos  # itos: integer-to-string mapping for characters

# Load trained model weights from file
checkpoint = torch.load('trained_weights.pt')
C = checkpoint['C']   # Character embedding matrix
W1 = checkpoint['W1'] # First layer weights
b1 = checkpoint['b1'] # First layer bias
W2 = checkpoint['W2'] # Second layer weights
b2 = checkpoint['b2'] # Second layer bias

# Set up a random generator for reproducible sampling
g = torch.Generator().manual_seed(2147483647 + 10)
block_size = 3 # Context length (number of previous characters used)

# Generate 20 names using the trained model
for _ in range(20):
  out = []
  context = [0] * block_size # Initialize context with all zeros (start token)
  while True:
    # Forward pass: embed context, pass through layers, get logits
    emb = C[torch.tensor([context])] # (1, block_size, embedding_dim)
    h = torch.tanh(emb.view(1, -1) @ W1 + b1) # Hidden layer
    logits = h @ W2 + b2 # Output logits for next character
    probs = F.softmax(logits, dim=1) # Convert logits to probabilities
    # Sample next character index from probabilities
    ix = torch.multinomial(probs, num_samples=1, generator=g).item()
    # Update context: remove oldest, add new character
    context = context[1:] + [ix]
    out.append(ix)
    # If sampled character is '.', end the name
    if ix == 0:
      break
  # Convert indices to characters and print the generated name
  print(''.join(itos[i] for i in out))