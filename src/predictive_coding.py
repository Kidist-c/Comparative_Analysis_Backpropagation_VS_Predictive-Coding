"""Predictive Coding network.

Layers l = 0..L: x_0 = input (clamped), x_1..x_{L-1} = latent states, x_L = output (clamped to label in training).
    a_l  = W_{l-1} x_{l-1} + b_{l-1}        pre-activation
    mu_l = f(a_l)  (identity at output)     prediction of layer l
    e_l  = x_l - mu_l                       layer-wise prediction error
    F_l  = 0.5 * ||e_l||^2                  layer-wise energy
    F    = sum_l F_l                        total energy
Latent update : x_l <- x_l + lr_x * ( -e_l + (e_{l+1} * f'(a_{l+1})) W_l )
Weight update : W_l <- W_l + lr_w * (e_{l+1} * f'(a_{l+1}))^T x_l   (local)
"""
import numpy as np
from .activations import tanh, dtanh
from .common import init_params, forward

class PCNet:
    def __init__(self, sizes, seed=0, lr_x=0.1, T=50):
        self.sizes, self.L = sizes, len(sizes) - 1
        self.W, self.b = init_params(sizes, seed)
        self.lr_x, self.T = lr_x, T

    def _dact(self, A, l):
        """f'(a_l): derivative for hidden layers, 1 for the linear output layer."""
        return dtanh(A[l]) if l < self.L else 1.0

    def compute_errors(self, xs):
        """Given states xs, return pre-acts, predictions, errors e_l and layer-wise energies F_l (l=1..L)."""
        As, mus, es, Fl = [None], [None], [None], [None]
        for l in range(1, self.L + 1):
            a = xs[l - 1] @ self.W[l - 1].T + self.b[l - 1]
            mu = tanh(a) if l < self.L else a
            e = xs[l] - mu
            As.append(a); mus.append(mu); es.append(e)
            Fl.append(0.5 * np.sum(e ** 2) / len(e))
        return As, mus, es, Fl

    def total_energy(self, xs):
        return sum(self.compute_errors(xs)[3][1:])

    def infer(self, X, Y, record=False):
        """Inference phase: clamp x_0=X, x_L=Y; relax hidden states by gradient descent on F.
        Returns xs, As, es, Fl, history (list of [F_1..F_L] per step if record=True)."""
        xs = [x.copy() for x in forward(self.W, self.b, X)[0]]   # feed-forward init
        xs[-1] = Y.copy()
        hist = []
        for _ in range(self.T):
            As, _, es, Fl = self.compute_errors(xs)
            if record: hist.append(Fl[1:])
            for l in range(1, self.L):
                back = (es[l + 1] * self._dact(As, l + 1)) @ self.W[l]
                xs[l] += self.lr_x * (-es[l] + back)
        As, _, es, Fl = self.compute_errors(xs)
        if record: hist.append(Fl[1:])
        return xs, As, es, Fl, hist

    def weight_updates(self, xs, As, es):
        """Local weight/bias updates (descent directions, already batch-averaged)."""
        N = len(xs[0]); dW, db = [], []
        for l in range(self.L):
            g = es[l + 1] * self._dact(As, l + 1)
            dW.append(g.T @ xs[l] / N); db.append(g.sum(0) / N)
        return dW, db

    def step(self, X, Y, lr):
        xs, As, es, Fl, _ = self.infer(X, Y)
        dW, db = self.weight_updates(xs, As, es)
        for l in range(self.L):
            self.W[l] += lr * dW[l]
            self.b[l] += lr * db[l]
        return sum(Fl[1:])                                      # total energy

    def predict(self, X):
        return forward(self.W, self.b, X)[0][-1]
