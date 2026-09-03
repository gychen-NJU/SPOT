# -*- coding: utf-8 -*-
"""
spot.physics.voigt — Voigt and Faraday (Voigt-Faraday) profiles
================================================================

A rational approximation of the complex error function for the damping
parameter ``a != 0`` and a polynomial/Lagrange table interpolation for
``a = 0`` (after Hui, Armstrong & Wray 1977, JQSRT 19, 509; modified by
Solanki 1985 and Wittmann 1986).

Definitions
-----------
For a line profile with Doppler width ``dldop`` and damping ``a``:

    H(a, v) = (1/sqrt(pi)) * Re[w(z)],   z = a + i v        (Voigt)
    F(a, v) = (1/sqrt(pi)) * Im[w(z)]                        (Faraday)

The normalization is such that ``integral H dv = 1`` and the Faraday
profile satisfies ``F(a,v) = -F(a,-v)``.

All operations are vectorized with torch broadcasting; input tensors
must share the same shape.
"""

import torch

__all__ = ["voigt_faraday_profile"]

# Coefficients of the rational approximation of w(z) (Weideman style).
_A = [122.607931777104326, 214.382388694706425, 181.928533092181549,
      93.155580458138441, 30.180142196210589, 5.912626209773153,
      0.564189583562615]
_B = [122.60793177387535, 352.730625110963558, 457.334478783897737,
      348.703917719495792, 170.354001821091472, 53.992906912940207,
      10.479857114260399]

# Tabulated (1/sqrt(pi)) * D(v) for a = 0 (Faraday profile at a=0)
_XDWS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4,
         1.6, 1.8, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0,
         14.0, 16.0, 18.0, 20.0]
_YDWS = [9.9335991e-02, 1.9475104e-01, 2.8263167e-01, 3.5994348e-01,
         4.2443639e-01, 4.7476321e-01, 5.1050407e-01, 5.3210169e-01,
         5.4072434e-01, 5.3807950e-01, 5.0727350e-01, 4.5650724e-01,
         3.9993989e-01, 3.4677279e-01, 3.0134040e-01, 1.7827103e-01,
         1.2934799e-01, 1.0213407e-01, 8.4542692e-02, 7.2180972e-02,
         6.3000202e-02, 5.5905048e-02, 5.0253846e-02, 4.1812878e-02,
         3.5806101e-02, 3.1311397e-02, 2.7820844e-02, 2.5031367e-02]

_SQRT_PI_INV = 0.564189583562615   # 1/sqrt(pi)

# Cache of the a=0 Faraday table tensors per (device, dtype): the tables
# were re-allocated from Python lists on every Voigt call (once per Zeeman
# component of every line of every column), which dominated the host-side
# overhead of the profile loop.
_VDWS_CACHE = {}


def _vdws_tables(device, dtype):
    key = (device, dtype)
    tab = _VDWS_CACHE.get(key)
    if tab is None:
        tab = (torch.tensor(_XDWS, device=device, dtype=dtype),
               torch.tensor(_YDWS, device=device, dtype=dtype))
        if len(_VDWS_CACHE) > 8:      # keep the cache bounded
            _VDWS_CACHE.clear()
        _VDWS_CACHE[key] = tab
    return tab


def voigt_faraday_profile(v, a):
    """
    Evaluate the Voigt H(a,v) and Faraday F(a,v) profiles.

    Parameters
    ----------
    v : torch.Tensor
        Frequency shift in Doppler units (dimensionless).
    a : torch.Tensor
        Damping parameter (dimensionless).  Same shape as ``v`` (or
        broadcastable).

    Returns
    -------
    H, F : torch.Tensor
        Voigt and Faraday profiles, same shape as ``v``.
    """
    sign = torch.where(v < 0, -1.0, 1.0)
    vv = v * sign  # |v|

    # General case a != 0: rational approximation of w(z), z = a - i|v|.
    # Computed for every element (also a == 0, where it is accurate) so
    # the routine stays free of data-dependent control flow; a == 0 is
    # then corrected with the exact table below.
    a0 = a == 0
    z = torch.complex(a, -vv)
    z2 = z
    num = _A[6]
    for k in range(5, -1, -1):
        num = num * z2 + _A[k]
    den = z2 + _B[6]
    for k in range(5, -1, -1):
        den = den * z2 + _B[k]
    w = num / den

    H = w.real
    F = 0.5 * w.imag * sign  # * _SQRT_PI_INV handled below

    H0, F0 = _voigt_a0(vv, sign), _faraday_a0(vv, sign)
    H = torch.where(a0, H0, H)
    F = torch.where(a0, F0, F)
    return H, F


def _voigt_a0(vv, sign):
    """Voigt profile for a = 0: H = exp(-v^2)."""
    return torch.exp(-vv * vv)


def _faraday_a0(vv, sign):
    """
    Faraday profile for a = 0, using the tabulated interpolation.

    F = sign * (1/sqrt(pi)) * D(|v|), with D tabulated in _XDWS/_YDWS
    and a polynomial approximation near 0 / asymptotic tail for large v.
    """
    v = vv
    # v <= 0.1 : D(v) = v * (1 - 2/3 v^2)
    d = torch.where(v <= _XDWS[0], v * (1.0 - 0.66666667 * v * v),
                    torch.zeros_like(v))
    # v >= 20 : D(v) = (1/(2v)) * (1 + 1/(2 v^2))
    d = torch.where(v >= _XDWS[-1], 0.5 / v * (1.0 + 0.5 / (v * v)), d)

    # table lookup + Lagrange 3-point interpolation for 0.1 < v < 20
    # (computed for every element; masked out-of-place, no data-dependent
    # control flow so the routine is vmap-compatible)
    mid = (v > _XDWS[0]) & (v < _XDWS[-1])
    x1, y1 = _vdws_tables(v.device, v.dtype)
    idx = torch.searchsorted(x1, v).clamp(1, len(_XDWS) - 2)
    k = idx - 1
    xa, xb, xc = x1[k - 1], x1[k], x1[k + 1]
    ya, yb, yc = y1[k - 1], y1[k], y1[k + 1]
    d1 = v - xa
    d2 = v - xb
    d3 = v - xc
    d12 = xa - xb
    d13 = xa - xc
    d23 = xb - xc
    dmid = (ya * d2 * d3 / (d12 * d13)
            - yb * d1 * d3 / (d12 * d23)
            + yc * d1 * d2 / (d13 * d23))
    d = torch.where(mid, dmid, d)
    return sign * _SQRT_PI_INV * d
