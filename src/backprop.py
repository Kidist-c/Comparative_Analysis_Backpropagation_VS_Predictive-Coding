"""Standard neural network trained with backpropagation."""
import numpy as np
from .activations import tanh, dtanh
from .common import init_params, forward

class BPNet:
    def __init__(self, sizes, seed=0):
        self.sizes, self.L = sizes, len(sizes) - 1
        self.W, self.b = init_params(sizes, seed)

    def grads(self, X, Y):
        """Returns (dW list, db list, loss). Loss = 0.5 * mean ||out - Y||^2."""
        xs, As = forward(self.W, self.b, X)
        N = len(X)
        d = (xs[-1] - Y) / N                          # dLoss/da_L
        gW, gb = [None] * self.L, [None] * self.L
        for l in range(self.L - 1, -1, -1):
            gW[l] = d.T @ xs[l]
            gb[l] = d.sum(0)
            if l > 0:                                  # global error signal sent backwards
                d = (d @ self.W[l]) * dtanh(As[l])
        return gW, gb, 0.5 * np.sum((xs[-1] - Y) ** 2) / N

    def step(self, X, Y, lr):
        gW, gb, loss = self.grads(X, Y)
        for l in range(self.L):
            self.W[l] -= lr * gW[l]
            self.b[l] -= lr * gb[l]
        return loss
