"""MNIST loader.

Looks for the data in this order:
  1. data/mnist.npz       (Keras format: x_train, y_train, x_test, y_test)  -> fully offline
  2. scikit-learn fetch_openml('mnist_784')  (downloads ~15 MB once, then cached)
Standard split is used: first 60,000 = train, last 10,000 = test. You can subsample for speed.
Pixels are scaled to [0,1] then normalised with the usual MNIST mean/std (0.1307 / 0.3081).
"""
import os
from collections import namedtuple
import numpy as np

Dataset = namedtuple("Dataset", "name Xtr ytr Ytr Xte yte n_in n_out")
MEAN, STD = 0.1307, 0.3081
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def _read_raw(data_dir):
    npz = os.path.join(data_dir, "mnist.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    try:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
    except Exception as e:
        raise RuntimeError(
            "Could not load MNIST. Either (a) put mnist.npz (Keras format) in bp_vs_pc/data/, or "
            "(b) run with internet access so scikit-learn can download it.\n"
            "Get the file with:  from tensorflow.keras.datasets import mnist  (or torchvision) and np.savez.\n"
            f"Original error: {e}")
    return X[:60000], y[:60000].astype(int), X[60000:], y[60000:].astype(int)

def load_mnist(n_train=5000, n_test=1000, data_dir=_DATA_DIR, seed=0):
    """Returns Dataset(Xtr, ytr, Ytr one-hot, Xte, yte, n_in=784, n_out=10). Use n_train=None for all 60,000."""
    xtr, ytr, xte, yte = _read_raw(data_dir)
    r = np.random.default_rng(seed)
    itr = r.permutation(len(ytr))[:n_train]; ite = r.permutation(len(yte))[:n_test]
    prep = lambda x: ((x.reshape(len(x), -1).astype(np.float64) / 255.0) - MEAN) / STD
    Xtr, ytr, Xte, yte = prep(xtr[itr]), ytr[itr].astype(int), prep(xte[ite]), yte[ite].astype(int)
    return Dataset("mnist", Xtr, ytr, np.eye(10)[ytr], Xte, yte, 784, 10)
