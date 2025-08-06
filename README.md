# MLP-Name-Generator
This project trains a simple neural network to generate names character-by-character using PyTorch.

## Features
- Trains a neural network on a dataset of names
- Generates new names based on learned patterns
- Saves and loads model weights for continued training and sampling

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/Neural-NetProject.git
   cd Neural-NetProject
   ```

2. **Set up a virtual environment (recommended):**
   ```sh
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On Mac/Linux
   ```

3. **Install dependencies:**
   ```sh
   pip install torch matplotlib
   ```

## Usage

1. **Prepare your dataset:**
   - Place a file named `names.txt` in the project directory, with one name per line.

2. **Train the model:**
   - Run the training script:
     ```sh
     python model.py
     ```
   - The script will train the model and save weights to `trained_weights.pt`.

3. **Generate names:**
   - After training, run:
     ```sh
     python results.py
     ```
   - This will sample and print generated names using the trained model.

## Notes
- You can continue training from saved weights by running `model.py` again.
- Make sure `names.txt` is present and formatted correctly.

## Resetting Weights
- To reset the model weights, run:
  ```sh
  python weights_biases.py
  ```
- Then delete `trained_weights.pt` if you want to start training from scratch.



## Requirements
- Python 3.8+
- [PyTorch](https://pytorch.org/)