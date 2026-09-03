# -*- coding: utf-8 -*-
"""
spot.visualization — plotting helpers for inversion results
=============================================================
Provides :func:`plot_convergence`, which draws the chi2 history of an
:class:`spot.inversion.InversionResult`.  Every node cycle is drawn as its
own line segment, cycle boundaries are marked with vertical dashed lines,
and each segment is annotated with its node configuration.

Only depends on matplotlib; all heavy computation stays in the spot core
modules.
"""

import numpy as np

import matplotlib.pyplot as plt

__all__ = ["plot_convergence"]


def _nodes_label(nodes):
    """Compact label of a node config, e.g. 'T:5  B:3  vlos:3'."""
    return "  ".join(f"{q}:{n}" for q, n in nodes.items() if n)


def plot_convergence(result, figsize=(11, 5), logscale=True,
                     show_rejected=True, ax=None):
    """
    Plot the chi2 convergence of an inversion.

    Every node cycle is drawn as its own line segment; cycle boundaries
    are marked by vertical dashed lines with the node configuration
    printed above them.  Rejected iterations (steps that did not lower
    chi2) are optionally marked with crosses.

    Parameters
    ----------
    result : InversionResult
        Result whose ``history`` contains the per-iteration log.
    figsize : tuple, optional
        Figure size (width, height).
    logscale : bool, optional
        Use a logarithmic y axis (recommended: chi2 spans decades).
    show_rejected : bool, optional
        Mark rejected iterations with 'x'.
    ax : matplotlib.axes.Axes, optional
        Draw onto an existing axes instead of creating a new figure.

    Returns
    -------
    (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    hist = result.history or []
    if not hist:
        ax.text(0.5, 0.5, "no history recorded", ha="center",
                transform=ax.transAxes)
        return fig, ax

    x = [h["iteration"] for h in hist]
    y = [h["chi2"] for h in hist]
    accepted = [h["accepted"] for h in hist]

    # cycle boundaries: a new cycle starts when the node config changes
    starts = [0]
    for i in range(1, len(hist)):
        if hist[i]["nodes"] != hist[i - 1]["nodes"]:
            starts.append(i)
    n_cyc = len(starts)

    # draw each cycle segment with its own color
    cmap = plt.get_cmap("tab10")
    for c, i0 in enumerate(starts):
        i1 = starts[c + 1] if c + 1 < n_cyc else len(hist)
        ax.plot(x[i0:i1], y[i0:i1], "-o", ms=3.5, lw=1.5,
                color=cmap(c % 10), label=f"cycle {c + 1}")

    # cycle separator lines (dashed)
    for c in range(1, n_cyc):
        ax.axvline(x[starts[c]], ls="--", color="0.35", lw=1.2, zorder=0)

    # node-config labels: above the tallest chi2 of each cycle segment
    for c, i0 in enumerate(starts):
        i1 = starts[c + 1] if c + 1 < n_cyc else len(hist)
        label = _nodes_label(hist[i0]["nodes"])
        if label:
            seg_max = max(y[i0:i1])
            xpos = x[i0] + (x[i1 - 1] - x[i0]) / 2
            ypos = seg_max * (1.25 if logscale else 1.04)
            ax.text(xpos, ypos, label, ha="center", va="bottom",
                    fontsize=8, color="0.25", rotation=0)

    if show_rejected:
        rx = [xi for xi, a in zip(x, accepted) if not a]
        ry = [yi for yi, a in zip(y, accepted) if not a]
        if rx:
            ax.plot(rx, ry, "x", color="0.4", ms=5, mew=1.2, alpha=0.8,
                    label="rejected step")

    ax.set_xlabel("LM iteration")
    ax.set_ylabel("chi2")
    if logscale:
        ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Inversion convergence (chi2 per iteration, "
                 f"runtime {result.runtime:.1f}s)")
    return fig, ax
