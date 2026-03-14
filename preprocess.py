import re
import numpy as np
from collections import Counter

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return tokens

STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "in", "is", "it", "that",
    "this", "was", "he", "she", "his", "her", "my", "your", "our",
    "their", "we", "you", "i", "me", "him", "us", "them", "be", "are",
    "have", "has", "had", "do", "did", "will", "would", "could", "should",
    "may", "might", "shall", "with", "for", "on", "at", "by", "from",
    "as", "or", "but", "not", "so", "if", "up", "out", "no", "what",
    "all", "were", "they", "been", "one", "there", "when", "who", "which"
    "thou", "thy", "thee", "o", "hath", "doth", "art", "hast", "tis", "twas", "ere", "oft", "how"
}

def build_vocab(tokens, min_count=5):
    counts = Counter(tokens)
    vocab = [w for w, c in counts.items() if c >= min_count and w not in STOPWORDS]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    return word2idx, idx2word

def generate_training_pairs(tokens, word2idx, window_size=2, subsample_threshold=1e-3):
    pairs = []
    counts = Counter(tokens)
    total = len(tokens)
    
    # Subsampling probability
    keep_prob = {w: min(1.0, (subsample_threshold / (counts[w] / total)) ** 0.5)
                 for w in word2idx}
    
    indexed = [(word2idx[w], keep_prob[w]) for w in tokens if w in word2idx]
    
    for i, (center, prob) in enumerate(indexed):
        if np.random.random() > prob:
            continue
        start = max(0, i - window_size)
        end = min(len(indexed), i + window_size + 1)
        for j in range(start, end):
            if i != j:
                pairs.append((center, indexed[j][0]))
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