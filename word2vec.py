import numpy as np

class Word2Vec:
    def __init__(self, vocab_size, embedding_dim=100):
        self.W_center = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_context = np.random.randn(vocab_size, embedding_dim) * 0.01

    def sigmoid(self, x):
        return np.where(x >= 0, 1 / (1 + np.exp(-np.clip(x, -500, 500))),np.exp(np.clip(x, -500, 500)) / (1 + np.exp(np.clip(x, -500, 500))))

    def forward(self, center_idx, context_idx, negative_indices):
        v_center = self.W_center[center_idx]
        u_context = self.W_context[context_idx]
        u_negatives = self.W_context[negative_indices]
        
        score_pos = self.sigmoid(np.dot(v_center, u_context))
        scores_neg = self.sigmoid(-np.dot(u_negatives, v_center))
        
        loss = -np.log(score_pos) - np.sum(np.log(scores_neg))
        
        return loss, v_center, u_context, u_negatives, score_pos, scores_neg

    def backward(self, center_idx, context_idx, negative_indices,
        v_center, u_context, u_negatives, score_pos, scores_neg, lr=0.01):
    
        # Gradients
        grad_v = (score_pos - 1) * u_context - np.sum((1 - scores_neg)[:, None] * u_negatives, axis=0)
        grad_u_pos = (score_pos - 1) * v_center
        grad_u_neg = -(1 - scores_neg)[:, None] * v_center

        # Gradient clipping
        grad_v = np.clip(grad_v, -2, 2)
        grad_u_pos = np.clip(grad_u_pos, -2, 2)
        grad_u_neg = np.clip(grad_u_neg, -2, 2)

        # Updates avec weight decay
        self.W_center[center_idx] -= lr * grad_v + 1e-5 * self.W_center[center_idx]
        self.W_context[context_idx] -= lr * grad_u_pos + 1e-5 * self.W_context[context_idx]
        self.W_context[negative_indices] -= lr * grad_u_neg + 1e-5 * self.W_context[negative_indices]