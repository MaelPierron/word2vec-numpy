import numpy as np

class Word2Vec:
    def __init__(self, vocab_size, embedding_dim=100):
        self.W_center = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_context = np.random.randn(vocab_size, embedding_dim) * 0.01

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, center_idx, context_idx, negative_indices):
        v_center = self.W_center[center_idx]
        u_context = self.W_context[context_idx]
        u_negatives = self.W_context[negative_indices]
        
        score_pos = self.sigmoid(np.dot(v_center, u_context))
        scores_neg = self.sigmoid(-np.dot(u_negatives, v_center))
        
        loss = -np.log(score_pos) - np.sum(np.log(scores_neg))
        
        return loss, v_center, u_context, u_negatives, score_pos, scores_neg