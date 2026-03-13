import numpy as np
import time
from preprocess import preprocess, build_vocab, generate_training_pairs
from word2vec import Word2Vec

def get_negative_samples(vocab_size, context_idx, k=5):
    negatives = []
    while len(negatives) < k:
        idx = np.random.randint(0, vocab_size)
        if idx != context_idx:
            negatives.append(idx)
    return negatives

def train(model, pairs, vocab_size, epochs=5, lr=0.0005):
    for epoch in range(epochs):
        total_loss = 0
        np.random.shuffle(pairs)
        start_time = time.time()
        n = len(pairs) * epochs
        offset = epoch * len(pairs)
        
        for i, (center_idx, context_idx) in enumerate(pairs):
            # Learning rate decay global
            current_lr = lr * (1 - (offset + i) / n)
            current_lr = max(current_lr, lr * 0.0001)
            
            negative_indices = get_negative_samples(vocab_size, context_idx)
            
            loss, v_center, u_context, u_negatives, score_pos, scores_neg = model.forward(
                center_idx, context_idx, negative_indices)
            
            model.backward(center_idx, context_idx, negative_indices,
                         v_center, u_context, u_negatives, score_pos, scores_neg, current_lr)
            
            total_loss += loss
            
            if i % 100000 == 0 and i > 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}, step {i}, avg loss (last 100k): {total_loss / 100000:.4f}, time: {elapsed:.0f}s")
                total_loss = 0
        
        print(f"Epoch {epoch+1} complete! Time: {time.time() - start_time:.0f}s")
        np.save(f"data/embeddings_epoch{epoch+1}.npy", model.W_center)
        print(f"Embeddings saved for epoch {epoch+1}!")

if __name__ == "__main__":
    with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokens = preprocess(text)
    word2idx, idx2word = build_vocab(tokens)
    pairs = generate_training_pairs(tokens, word2idx)
    pairs = np.array(pairs)

    vocab_size = len(word2idx)
    model = Word2Vec(vocab_size, embedding_dim=100)

    train(model, pairs, vocab_size, epochs=5, lr=0.0005)

    np.save("data/embeddings.npy", model.W_center)
    print("Embeddings saved!")