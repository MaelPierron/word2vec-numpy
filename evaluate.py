import numpy as np
from preprocess import preprocess, build_vocab

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def most_similar(word, word2idx, idx2word, embeddings, top_n=5):
    if word not in word2idx:
        print(f"'{word}' not in vocabulary")
        return
    
    idx = word2idx[word]
    v = embeddings[idx]
    
    similarities = []
    for i, u in enumerate(embeddings):
        if i != idx:
            sim = cosine_similarity(v, u)
            similarities.append((idx2word[i], sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

if __name__ == "__main__":
    with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokens = preprocess(text)
    word2idx, idx2word = build_vocab(tokens)
    embeddings = np.load("data/embeddings_epoch5.npy")

    test_words = ["king", "love", "sword", "death", "good"]
    for word in test_words:
        results = most_similar(word, word2idx, idx2word, embeddings)
        if results:
            print(f"\nMost similar to '{word}':")
            for w, sim in results:
                print(f"  {w}: {sim:.4f}")