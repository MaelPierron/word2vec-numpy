import numpy as np
from preprocess import preprocess, build_vocab, generate_training_pairs
from word2vec import Word2Vec

with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokens = preprocess(text)
word2idx, idx2word = build_vocab(tokens)
pairs = generate_training_pairs(tokens, word2idx)

model = Word2Vec(len(word2idx), embedding_dim=100)

# Test sur 10 paires seulement
for i, (center_idx, context_idx) in enumerate(pairs[:10]):
    neg = [np.random.randint(0, len(word2idx)) for _ in range(5)]
    loss1, v, u, u_neg, sp, sn = model.forward(center_idx, context_idx, neg)
    model.backward(center_idx, context_idx, neg, v, u, u_neg, sp, sn, lr=0.001)
    loss2, _, _, _, _, _ = model.forward(center_idx, context_idx, neg)
    print(f"Pair {i}: loss before={loss1:.4f}, loss after={loss2:.4f}, diff={loss1-loss2:.6f}")