"""Plot helpers (matplotlib)."""
import numpy as np
import matplotlib.pyplot as plt
from .common import forward

def plot_samples(ds, n=10):
    """First n training images (undoes the normalisation for display)."""
    fig, ax = plt.subplots(1, n, figsize=(1.2 * n, 1.6))
    for i, a in enumerate(ax):
        a.imshow(ds.Xtr[i].reshape(28, 28), cmap="gray"); a.set_title(int(ds.ytr[i])); a.axis("off")
    plt.tight_layout(); return fig

def plot_training(log_bp, log_pc):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(log_bp["objective"], label="BP loss"); ax[0].plot(log_pc["objective"], label="PC total energy")
    ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_title("Training objective"); ax[0].legend()
    ax[1].plot(log_bp["test_acc"], label="BP"); ax[1].plot(log_pc["test_acc"], label="PC")
    ax[1].set_xlabel("epoch"); ax[1].set_title("Test accuracy"); ax[1].legend()
    plt.tight_layout(); return fig

def plot_layer_energy(hist):
    """hist: array (T+1, L) of layer-wise energies F_1..F_L during inference."""
    hist = np.asarray(hist)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for k in range(hist.shape[1]):
        ax.plot(hist[:, k], label=f"F{k+1} " + ("(output)" if k == hist.shape[1] - 1 else f"(hidden {k+1})"))
    ax.plot(hist.sum(1), "k--", label="total F")
    ax.set_xlabel("inference step"); ax.set_ylabel("energy"); ax.set_title("Layer-wise energy during inference"); ax.legend()
    plt.tight_layout(); return fig

def plot_predictions(bp, pc, ds, n=10):
    """Test images with BP and PC predicted labels (red = wrong)."""
    pb = forward(bp.W, bp.b, ds.Xte[:n])[0][-1].argmax(1); pp = forward(pc.W, pc.b, ds.Xte[:n])[0][-1].argmax(1)
    fig, ax = plt.subplots(1, n, figsize=(1.3 * n, 2))
    for i, a in enumerate(ax):
        a.imshow(ds.Xte[i].reshape(28, 28), cmap="gray"); a.axis("off")
        bad = pb[i] != ds.yte[i] or pp[i] != ds.yte[i]
        a.set_title(f"true {ds.yte[i]}\nBP {pb[i]} PC {pp[i]}", fontsize=8, color="red" if bad else "black")
    plt.tight_layout(); return fig
