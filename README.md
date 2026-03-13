# Word2Vec in Pure NumPy

This project implements the Word2Vec skip-gram model with negative sampling from scratch, using only NumPy. No PyTorch, no TensorFlow — just matrix operations and gradient descent by hand.

## Why this project

Word2Vec is one of the most elegant ideas in NLP: train a shallow neural network to predict context words, then throw away the network and keep the weights as word representations. The result is a dense vector space where semantically related words end up close to each other.

Implementing it from scratch forces a deep understanding of the forward pass, the loss function, and the gradient derivations — which is the whole point.

## How it works

The model uses a sliding window over the text to generate (center, context) pairs. For each pair, it also samples random "negative" pairs — words that are not actually neighbors. The model learns to assign high probability to real pairs and low probability to negative ones, which gradually pushes related word vectors together.

## Usage
```bash
pip install numpy
python download_data.py  # Downloads the complete works of Shakespeare
python train.py          # Trains the model (~25 min on CPU)
python evaluate.py       # Shows nearest neighbors for test words
```

## Training details

The model trains on ~900k tokens from Shakespeare with a vocabulary of ~8800 words. Embeddings have dimension 100, window size is 2, and 5 negative samples are drawn per positive pair. Learning rate decays linearly over the full training.