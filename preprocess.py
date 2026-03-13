import re
from collections import Counter

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return tokens

def build_vocab(tokens, min_count=5):
    counts = Counter(tokens)
    vocab = [w for w, c in counts.items() if c >= min_count]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    return word2idx, idx2word

def generate_training_pairs(tokens, word2idx, window_size=2):
    pairs = []
    indexed = [word2idx[w] for w in tokens if w in word2idx]
    
    for i, center in enumerate(indexed):
        start = max(0, i - window_size)
        end = min(len(indexed), i + window_size + 1)
        for j in range(start, end):
            if i != j:
                pairs.append((center, indexed[j]))
    return pairs

if __name__ == "__main__":
    with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokens = preprocess(text)
    word2idx, idx2word = build_vocab(tokens)

    print(f"Total tokens: {len(tokens)}")
    print(f"Vocabulary size: {len(word2idx)}")
    print(f"First 10 words: {tokens[:10]}")
    pairs = generate_training_pairs(tokens, word2idx)
    print(f"Number of training pairs: {len(pairs)}")
    print(f"First 5 pairs: {pairs[:5]}")