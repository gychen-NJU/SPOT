# -*- coding: utf-8 -*-
"""
spot.utils.interpolation — 1-D interpolation with extrapolation
================================================================

Provides :func:`interp_to_grid`, a numpy-based interpolator with
(optional) linear extrapolation beyond both ends of the source grid.
It is used to (a) resample preset atmosphere models onto a
user-supplied ``ltau`` grid and (b) map node values back onto the full
depth grid during inversion.

The function is numpy on purpose: the model resampling happens once, on
the host, before tensors are moved to the device.  The node-to-grid
step of the inversion is performed in torch and lives in
:mod:`spot.inversion.nodes`.
"""

import numpy as np

__all__ = ["interp_to_grid"]


def interp_to_grid(x_source, y_source, x_target, method="linear"):
    """
    Interpolate ``y_source(x_source)`` onto ``x_target``, with linear
    extrapolation beyond the source grid (both ends).

    Parameters
    ----------
    x_source : (Ns,) array-like
        Source abscissae (strictly increasing).
    y_source : (Ns,) or (..., Ns) array-like
        Values at the source grid; the last axis is interpolated.
    x_target : (Nt,) array-like
        Target abscissae (any order).
    method : str
        'linear' or 'cubic' (natural cubic spline).

    Returns
    -------
    ndarray of shape y.shape[:-1] + (Nt,)
    """
    x_s = np.asarray(x_source, dtype=float)
    x_t = np.asarray(x_target, dtype=float)
    y_s = np.asarray(y_source, dtype=float)
    if x_s.ndim != 1 or x_t.ndim != 1:
        raise ValueError("x_source and x_target must be 1-D")
    if y_s.shape[-1] != x_s.shape[0]:
        raise ValueError("last axis of y_source must match len(x_source)")
    if x_s.shape[0] < 2:
        raise ValueError("need at least 2 source points")

    if method == "linear":
        # np.interp only accepts 1-D values; flatten the leading dims of
        # y (any shape (..., Ns)), interpolate every row, reshape back.
        lead = y_s.shape[:-1]
        flat = y_s.reshape(-1, y_s.shape[-1])
        out = np.stack([np.interp(x_t, x_s, flat[i], left=None, right=None)
                        for i in range(flat.shape[0])])
        return out.reshape(*lead, x_t.shape[0])

    if method == "cubic":
        return _cubic_with_extrapolation(x_s, y_s, x_t)

    raise ValueError(f"method must be 'linear' or 'cubic', got {method!r}")


def _cubic_with_extrapolation(x_s, y_s, x_t):
    """Natural cubic spline; linear extrapolation outside [x_s[0], x_s[-1]]."""
    n = x_s.shape[0]
    # Solve the natural cubic spline system for second derivatives.
    #   h_i = x_{i+1} - x_i ;  system:  h_{i-1} M_{i-1} + 2(h_{i-1}+h_i) M_i
    #   + h_i M_{i+1} = 6 (d_{i+1} - d_i)     (M_0 = M_{n-1} = 0)
    h = np.diff(x_s)
    if np.any(h <= 0):
        raise ValueError("x_source must be strictly increasing")
    d = np.diff(y_s, axis=-1) / h

    diag = np.zeros(n)
    off = np.zeros(n - 1)
    rhs = np.zeros((*y_s.shape[:-1], n))
    diag[0] = 1.0
    diag[-1] = 1.0
    off[0] = 0.0
    for i in range(1, n - 1):
        diag[i] = 2.0 * (h[i - 1] + h[i])
        off[i - 1] = h[i - 1]
        off[i] = h[i]
        rhs[..., i] = 6.0 * (d[..., i] - d[..., i - 1])

    m = _solve_tridiagonal(diag, off, rhs)          # (..., n) second derivatives

    # Evaluate the spline on x_t (only inside the source range)
    idx = np.searchsorted(x_s, x_t) - 1
    idx = np.clip(idx, 0, n - 2)
    x0 = x_s[idx]
    x1 = x_s[idx + 1]
    t = (x_t - x0) / (x1 - x0)
    y0 = np.take_along_axis(y_s, np.broadcast_to(idx, x_t.shape), axis=-1)
    y1 = np.take_along_axis(y_s, np.broadcast_to(idx + 1, x_t.shape), axis=-1)
    m0 = np.take_along_axis(m, np.broadcast_to(idx, x_t.shape), axis=-1)
    m1 = np.take_along_axis(m, np.broadcast_to(idx + 1, x_t.shape), axis=-1)
    hh = x1 - x0
    y = ((m0 * (x1 - x_t) ** 3 + m1 * (x_t - x0) ** 3) / (6.0 * hh)
         + (y0 - m0 * hh * hh / 6.0) * (x1 - x_t) / hh
         + (y1 - m1 * hh * hh / 6.0) * (x_t - x0) / hh)

    # Linear extrapolation beyond both ends
    slope_left = (y_s[..., 1] - y_s[..., 0]) / (x_s[1] - x_s[0])
    slope_right = (y_s[..., -1] - y_s[..., -2]) / (x_s[-1] - x_s[-2])
    left = x_t < x_s[0]
    right = x_t > x_s[-1]
    if np.any(left):
        y = np.where(left, y_s[..., :1] + slope_left[..., None] * (x_t - x_s[0]), y)
    if np.any(right):
        y = np.where(right, y_s[..., -1:] + slope_right[..., None] * (x_t - x_s[-1]), y)
    return y


def _solve_tridiagonal(diag, off, rhs):
    """Thomas algorithm for a symmetric tridiagonal system."""
    n = diag.shape[0]
    out = np.array(rhs, dtype=float)
    c = np.zeros(n)
    c[0] = off[0] / diag[0]
    for i in range(1, n - 1):
        denom = diag[i] - off[i - 1] * c[i - 1]
        c[i] = off[i] / denom
    # forward substitution
    out[..., 0] = out[..., 0] / diag[0]
    for i in range(1, n):
        out[..., i] = (out[..., i] - off[i - 1] * out[..., i - 1]) / (
            diag[i] - off[i - 1] * c[i - 1])
    # back substitution
    for i in range(n - 2, -1, -1):
        out[..., i] = out[..., i] - c[i] * out[..., i + 1]
    return out
