"""Shared network utilities: parameter init, forward pass, accuracy."""
import numpy as np
from .activations import tanh

def init_params(sizes, seed=0):
    """Same seed -> identical starting weights for BP and PC (fair comparison)."""
    r = np.random.default_rng(seed)
    W = [r.normal(0, np.sqrt(1 / sizes[i]), (sizes[i + 1], sizes[i])) for i in range(len(sizes) - 1)]
    b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
    return W, b

def forward(W, b, X, act=tanh):
    """Feed-forward pass. Returns activations xs[0..L] and pre-activations As[0..L] (As[0] = None).
    Hidden layers use `act`, the output layer is linear."""
    L = len(W)
    xs, As = [X], [None]
    for l in range(L):
        a = xs[-1] @ W[l].T + b[l]
        As.append(a)
        xs.append(act(a) if l < L - 1 else a)
    return xs, As

def accuracy(W, b, X, y):
    return float((forward(W, b, X)[0][-1].argmax(1) == y).mean())
