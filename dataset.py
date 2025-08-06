
import torch

# Load the dataset: each line in names.txt is a name
words = open('names.txt', 'r').read().splitlines()

# Build the vocabulary of characters and mappings to/from integers
def get_vocab(words):
    chars = sorted(list(set(''.join(words)))) # Unique characters in dataset
    stoi = {s:i+1 for i,s in enumerate(chars)} # String to integer mapping (start at 1)
    stoi['.'] = 0 # Special token for end-of-name
    itos = {i:s for s,i in stoi.items()} # Integer to string mapping
    return stoi, itos

# Initialize vocabulary mappings
stoi, itos = get_vocab(words)

# Build the dataset for training/testing
def build_dataset(words):
    block_size = 3 # Context length: how many previous chars to use
    X, Y = [], []  # X: context, Y: next character
    for w in words:
        context = [0] * block_size # Start with all zeros (start tokens)
        for ch in w + '.':         # Iterate over each character plus end token
            ix = stoi[ch]          # Get integer index for character
            X.append(context)      # Store current context
            Y.append(ix)           # Store next character index
            context = context[1:] + [ix] # Slide context window and append new char

    X = torch.tensor(X) # Convert to tensor
    Y = torch.tensor(Y)
    return X, Y

# Split the dataset into train/dev/test sets
def split_dataset(words):
    import random
    random.seed(42)         # Ensure reproducibility
    random.shuffle(words)   # Shuffle names
    n1 = int(0.8*len(words)) # 80% for training
    n2 = int(0.9*len(words)) # 10% for dev, 10% for test

    # Build splits
    Xtr, Ytr = build_dataset(words[:n1])      # Training set
    Xdev, Ydev = build_dataset(words[n1:n2])  # Dev/validation set
    Xte, Yte = build_dataset(words[n2:])      # Test set
    return Xtr, Ytr, Xdev, Ydev, Xte, Yte

# Create the splits and export for use in model.py
Xtr, Ytr, Xdev, Ydev, Xte, Yte = split_dataset(words)