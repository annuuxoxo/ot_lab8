"""A 1-hidden-layer MLP with backpropagation written by hand in NumPy.

Backpropagation computes the gradient (chain rule, layer by layer, backwards).
The optimizer (SGD, Adam, ...) is a separate thing that USES that gradient to update weights.

Model:  h = tanh(X W1 + b1),  p = sigmoid(h W2 + b2)
Loss:   mean binary cross-entropy + (lam/2) * (||W1||^2 + ||W2||^2)   <- ML = loss + regularizer
"""
import numpy as np


def init_params(d_in, d_hidden, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.standard_normal((d_in, d_hidden)) / np.sqrt(d_in),
        "b1": np.zeros(d_hidden),
        "W2": rng.standard_normal((d_hidden, 1)) / np.sqrt(d_hidden),
        "b2": np.zeros(1),
    }


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(params, X):
    z1 = X @ params["W1"] + params["b1"]
    h = np.tanh(z1)
    z2 = h @ params["W2"] + params["b2"]
    p = sigmoid(z2).ravel()
    return p, (X, h)


def loss(params, X, y, lam=0.0):
    p, _ = forward(params, X)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    bce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    reg = 0.5 * lam * (np.sum(params["W1"] ** 2) + np.sum(params["W2"] ** 2))
    return bce + reg


def backward(params, X, y, lam=0.0):
    """Backpropagation: returns dLoss/dparam for every parameter."""
    p, (X, h) = forward(params, X)
    n = len(y)
    # d(BCE)/dz2 for sigmoid + cross-entropy simplifies to (p - y) / n
    dz2 = ((p - y) / n)[:, None]
    dW2 = h.T @ dz2 + lam * params["W2"]
    db2 = dz2.sum(0)
    # chain rule back through W2 and tanh (tanh' = 1 - tanh^2)
    dh = dz2 @ params["W2"].T
    dz1 = dh * (1 - h ** 2)
    dW1 = X.T @ dz1 + lam * params["W1"]
    db1 = dz1.sum(0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def accuracy(params, X, y):
    p, _ = forward(params, X)
    return np.mean((p > 0.5) == y)


# --- tiny optimizers operating on parameter dicts (same update rules as optimizers.py) ---

def train(params, X, y, method="sgd", lr=0.1, epochs=300, batch=32, lam=0.0, seed=0):
    rng = np.random.default_rng(seed)
    params = {k: v.copy() for k, v in params.items()}
    m = {k: np.zeros_like(v) for k, v in params.items()}
    s = {k: np.zeros_like(v) for k, v in params.items()}
    t = 0
    history = []
    for _ in range(epochs):
        perm = rng.permutation(len(y))
        for i in range(0, len(y), batch):
            idx = perm[i:i + batch]
            g = backward(params, X[idx], y[idx], lam)
            t += 1
            for k in params:
                if method == "sgd":
                    params[k] -= lr * g[k]
                elif method == "momentum":
                    m[k] = 0.9 * m[k] + g[k]
                    params[k] -= lr * m[k]
                elif method == "adam":
                    m[k] = 0.9 * m[k] + 0.1 * g[k]
                    s[k] = 0.999 * s[k] + 0.001 * g[k] ** 2
                    params[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(s[k] / (1 - 0.999 ** t)) + 1e-8)
                else:
                    raise ValueError(method)
        history.append(loss(params, X, y, lam))
    return params, np.array(history)
