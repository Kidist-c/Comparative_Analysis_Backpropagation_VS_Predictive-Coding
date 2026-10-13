"""Training loop and gradient-equivalence check."""
import time
import numpy as np
from .common import accuracy, forward
from .backprop import BPNet
from .predictive_coding import PCNet

def train(net, Xtr, ytr, Ytr, Xte, yte, epochs=150, lr=0.3, batch_size=50, seed=0, verbose=0):
    """Works for BPNet and PCNet (both expose .step(X, Y, lr)). Returns a log dict."""
    rng = np.random.default_rng(seed)
    log = {"objective": [], "train_acc": [], "test_acc": [], "time": 0.0}
    t0, n = time.time(), len(Xtr)
    for ep in range(epochs):
        p, tot = rng.permutation(n), 0.0
        for i in range(0, n, batch_size):
            idx = p[i:i + batch_size]
            tot += net.step(Xtr[idx], Ytr[idx], lr) * len(idx)
        log["objective"].append(tot / n)
        log["train_acc"].append(accuracy(net.W, net.b, Xtr, ytr))
        log["test_acc"].append(accuracy(net.W, net.b, Xte, yte))
        if verbose and (ep + 1) % verbose == 0:
            print(f"epoch {ep+1:4d}  objective {log['objective'][-1]:.4f}  test acc {log['test_acc'][-1]:.3f}")
    log["time"] = time.time() - t0
    return log

def pc_vs_bp_gradient_similarity(sizes, X, Y, seed=7, nudge=0.01, T=400, lr_x=0.05):
    """Cosine similarity between PC weight updates and -BP gradients when the output is only slightly nudged
    toward the label (small-error regime where PC approximates BP). Returns list per layer."""
    pc, bp = PCNet(sizes, seed, lr_x=lr_x, T=T), BPNet(sizes, seed)
    Yn = forward(pc.W, pc.b, X)[0][-1]
    Yn = Yn + nudge * (Y - Yn)
    xs, As, es, _, _ = pc.infer(X, Yn)
    dW, _ = pc.weight_updates(xs, As, es)
    gW, _, _ = bp.grads(X, Yn)
    return [float(np.sum(d * -g) / (np.linalg.norm(d) * np.linalg.norm(g) + 1e-12)) for d, g in zip(dW, gW)]
