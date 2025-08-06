import torch

# Set up a random generator for reproducibility
g = torch.Generator().manual_seed(2147483647)

# Initialize model parameters
C = torch.randn((27, 10), generator=g)                # Character embedding matrix (27 chars, 10 dims)
W1 = torch.randn((30, 100), generator=g) * ((5/3) / 30**0.5) # First layer weights (input: 30, output: 100)
b1 = torch.randn(100, generator=g) * 0.01             # First layer bias
W2 = torch.randn((100, 27), generator=g) * 0.01       # Second layer weights (input: 100, output: 27)
b2 = torch.randn(27, generator=g) * 0                 # Second layer bias (initialized to zeros)
parameters = [C, W1, b1, W2, b2]                      # List of all trainable parameters

# Set requires_grad=True for all parameters to enable gradient computation
for p in parameters:
  p.requires_grad = True

# Print info only when running this file directly (not on import)
if __name__ == "__main__":
    print("Weights and biases have been reset.")
    print("Number of parameters:", sum(p.numel() for p in parameters))