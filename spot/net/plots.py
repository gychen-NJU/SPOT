"""spot.net.plots — plotting helpers used by training (live curves) and evaluation (final figures)."""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import DEPTH_PARAMS


def plot_training_curves(
    history: Dict[str, list],
    save_path: str,
    best_epoch: Optional[int] = None,
) -> str:
    """Plot train/val loss on a log scale alongside the learning-rate history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["train_loss"], label="Train", alpha=0.85)
    ax1.plot(history["val_loss"], label="Val", alpha=0.85)
    if best_epoch:
        ax1.axvline(x=best_epoch, color="g", ls="--", alpha=0.5, label="best")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax1.set_title("Training curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(history["lr"], alpha=0.85)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("LR")
    ax2.set_yscale("log")
    ax2.set_title("Learning rate (warmup + cosine)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_per_param_curves(history: Dict[str, list], save_path: str) -> str:
    """Plot per-parameter validation MSE curves (log scale), one panel per parameter."""
    keys = [k for k in history if "val_mse_" in k]
    if not keys:
        return save_path
    n = len(keys)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, k in enumerate(keys):
        axes[i].plot(history[k], label=k.replace("val_mse_", ""))
        axes[i].set_yscale("log")
        axes[i].set_title(k.replace("val_mse_", ""))
        axes[i].grid(True, alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_scatter(
    pred: np.ndarray,
    true: np.ndarray,
    title: str,
    save_path: str,
    corr: Optional[float] = None,
    log_scale: bool = False,
    max_points: int = 200_000,
) -> str:
    p, t = pred.ravel(), true.ravel()
    if len(p) > max_points:
        sel = np.random.RandomState(0).choice(len(p), max_points, replace=False)
        p, t = p[sel], t[sel]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(t, p, s=0.8, alpha=0.25, rasterized=True)
    lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
    ax.plot([lo, hi], [lo, hi], "r--", alpha=0.6, lw=1)
    ax.set_xlabel("True")
    ax.set_ylabel("Pred")
    label = title if corr is None else f"{title}  r={corr:.4f}"
    ax.set_title(label)
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_depth_profiles(
    pred: Dict[str, np.ndarray],
    true: Dict[str, np.ndarray],
    save_path: str,
    n_samples: int = 5,
    seed: int = 42,
) -> str:
    """Compare true vs predicted depth profiles for a few random test samples."""
    depth_params = [k for k in DEPTH_PARAMS if k != "f"]          # f is the periodic azimuth, plotted separately
    rng = np.random.RandomState(seed)
    n_test = true[depth_params[0]].shape[0]
    samples = rng.choice(n_test, min(n_samples, n_test), replace=False)

    fig, axes = plt.subplots(
        len(depth_params), len(samples),
        figsize=(3.0 * len(samples), 2.6 * len(depth_params)),
    )
    axes = np.atleast_2d(axes)
    for i, k in enumerate(depth_params):
        for j, sidx in enumerate(samples):
            ax = axes[i, j]
            ax.plot(true[k][sidx], "b-", label="True", alpha=0.85, lw=1.2)
            ax.plot(pred[k][sidx], "r--", label="Pred", alpha=0.85, lw=1.2)
            ax.set_title(f"{k} #{sidx}", fontsize=8)
            if j == 0:
                ax.set_ylabel(k, fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
