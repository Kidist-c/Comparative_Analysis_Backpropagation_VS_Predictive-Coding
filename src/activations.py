"""Activation functions and their derivatives."""
import numpy as np

def tanh(a):
    return np.tanh(a)

def dtanh(a):
    """Derivative of tanh w.r.t. its pre-activation a."""
    return 1.0 - np.tanh(a) ** 2
