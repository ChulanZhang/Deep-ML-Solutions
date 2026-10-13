import math
import random

def train(X_train, y_train, X_val, y_val, n_classes):
    """
    Build a simple digit classifier using ONLY math and random.
    """
    n_features = len(X_train[0]) if len(X_train) > 0 else 64
    weights = [[random.gauss(0, 0.1) for _ in range(n_features)] for _ in range(n_classes)]
    bias = [0.0] * n_classes
    
    lr = 0.01
    epochs = 20
    
    def softmax(logits):
        max_v = max(logits)
        exps = [math.exp(l - max_v) for l in logits]
        s = sum(exps)
        return [e / s for e in exps]
        
    for epoch in range(epochs):
        for i in range(len(X_train)):
            x = X_train[i]
            y = y_train[i]
            
            # forward
            logits = [sum(x[j]*weights[c][j] for j in range(n_features)) + bias[c] for c in range(n_classes)]
            probs = softmax(logits)
            
            # true label one-hot
            true_p = [1.0 if c == y else 0.0 for c in range(n_classes)]
            
            # gradients
            for c in range(n_classes):
                error = probs[c] - true_p[c]
                bias[c] -= lr * error
                for j in range(n_features):
                    weights[c][j] -= lr * error * x[j]

    def predict(X):
        preds = []
        for x in X:
            logits = [sum(x[j]*weights[c][j] for j in range(n_features)) + bias[c] for c in range(n_classes)]
            pred_class = max(range(n_classes), key=lambda c: logits[c])
            preds.append(pred_class)
        return preds
        
    return predict
