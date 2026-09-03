# -*- coding: utf-8 -*-
"""
spot.physics.opacity — continuous absorption opacity recipes
============================================================
Continuum absorption coefficient per hydrogen nucleon [cm^2/H].  The
primary (``'mihalas'``) recipe evaluates Mihalas-type formula opacities
for the bound-free / free-free continua of H-, He-, H I, H2-, H2+, He I,
Thomson and Rayleigh scattering, and the C, Na and Mg metals.  Every
contribution is returned per hydrogen nucleon (i.e. divided by p(H')),
so ratios like ``kappa(lambda) / kappa(5000 A)`` are directly usable in
the radiative-transfer equation, which needs only the opacity relative
to a reference wavelength.

Additional continuum-opacity recipes (ported from pyPRT-dsh
``pyprt/synth/synthesis.py``, which calls them "ATLAS" and
"Opacity Project") are available through ``continuum_opacity_recipe``:

* ``'mihalas'``           — the Mihalas-type formula recipe above
  (default);
* ``'atlas'``             — Rosseland mass-absorption table from the
  ATLAS solar ODF set: log10(kappa) on a (logT, logPg, vmic) grid,
  converted to per-H-nucleon [cm^2/H] with ``kappa * rg / N(H')``;
* ``'opacity_project'``   — Opacity Project tables on a (logT, logR)
  grid (per-unit-mass absorption, same conversion).

NOTE on units/normalization: the mihalas recipe returns the absorption
coefficient *per hydrogen nucleon* [cm^2/H].  The ATLAS / Opacity
Project tables store *mass* absorption coefficients [kappa_ross,
cm^2/g]; pyPRT converts with ``rg/NH`` (mass density divided by the
hydrogen-nucleon number density) and the same conversion is reproduced
here, so all recipes return the same per-H-nucleon [cm^2/H] units and
comparable values.
"""

import os

import numpy as np
import torch

from .constants import SIGMA_THOMSON

__all__ = ["continuum_opacity", "continuum_opacity_recipe",
           "continuum_opacity_derivatives"]

# ---------------------------------------------------------------------------
# He I levels: statistical weights and excitation energies [eV]
# ---------------------------------------------------------------------------
_G_HE = [1., 3., 1., 9., 9., 3., 3., 3., 1., 9., 20., 3.]
_CHI_HE = [0., 19.819, 20.615, 20.964, 20.964, 21.217, 21.217,
           22.718, 22.920, 23.007, 23.073, 23.087]

# C/Na/Mg bound-free opacity tables (log10 kappa grids vs T)
_TGRID_C = [3.9999e3, 5e3, 6e3, 7e3, 8e3, 9e3, 1e4, 1.1e4, 1.2e4, 1.3e4,
            1.4e4, 1.5e4, 1.6e4, 1.7e4, 1.8e4, 1.9e4, 2e4, 2.1e4, 2.2e4,
            2.3e4, 2.4e4, 2.5e4, 2.6e4, 2.7e4, 2.8e4, 2.9e4, 3.00001e4]
_FGRID_C = [1.728, 6.660, 9.964, 12.340, 14.134, 15.540, 16.676, 17.614,
            18.402, 19.076, 19.662, 20.172, 20.626, 21.030, 21.394,
            21.722, 22.022, 22.296, 22.548, 22.78, 22.996, 23.196,
            23.384, 23.56, 23.724, 23.878, 24.024]
_FGRID_NA = [1.638, 3.914, 5.461, 6.586, 7.448, 8.130, 8.686, 9.152, 9.546,
             9.887, 10.184, 10.448, 10.682, 10.894, 11.084, 11.258,
             11.418, 11.566, 11.704, 11.830, 11.950, 12.062, 12.166,
             12.264, 12.358, 12.448, 12.532]
_FGRID_MG = [1.784, 5.268, 7.618, 9.318, 10.61, 11.628, 12.452, 13.134,
             13.71, 14.204, 14.632, 15.008, 15.342, 15.64, 15.906, 16.15,
             16.37, 16.574, 16.76, 16.934, 17.094, 17.244, 17.384,
             17.514, 17.638, 17.754, 17.866]

_WGRID_C = [0., 1.1005e-5, 1.1005e-5, 1.2395e-5, 1.2395e-5, 1.4445e-5,
            1.4445e-5, 2.178e-5, 2.912e-5, 3.4625e-5, 3.4625e-5,
            4.7295e-5, 4.7295e-5, 5.4785e-5, 7.1e-5]
_WGRID_NA = [0., 1.2e-5, 1.5e-5, 1.6e-5, 1.7e-5, 1.8e-5, 1.95e-5, 2.1e-5,
             2.2e-5, 2.3e-5, 2.4125e-5, 2.4125e-5, 2.75e-5, 4.0845e-5,
             4.0845e-5, 6.366e-5, 8.1455e-5, 8.1455e-5, 8.9455e-5,
             8.9455e-5, 9.85e-5]
_WGRID_MG = [1.3999e-5, 1.5e-5, 1.6215e-5, 1.6215e-5, 2.5135e-5, 2.5135e-5,
             2.75e-5, 3.7565e-5, 3.7565e-5, 4.8845e-5, 4.8845e-5,
             6.5495e-5, 6.5495e-5, 7.2345e-5, 7.2345e-5, 7.2915e-5,
             7.2915e-5, 8.1135e-5, 8.1135e-5, 9.e-5]

# -(log10 kappa + 20) tables: [nwave, nT]
_CMESH_C = [
    [-2.75, -3.02, -1.16, -1.17, 1.33, 1.32, 8.05, 7.79, 7.66, 7.68, 8.12, 8.07, 8.28, 8.44, 8.14],
    [-2.75, -3.02, -1.48, -1.49, 0.66, 0.65, 5.99, 5.73, 5.61, 5.63, 5.86, 5.79, 5.95, 5.97, 5.67],
    [-2.75, -3.02, -1.68, -1.69, 0.22, 0.20, 4.58, 4.32, 4.21, 4.20, 4.36, 4.25, 4.36, 4.35, 4.02],
    [-2.75, -3.02, -1.83, -1.84, -0.10, -0.11, 3.56, 3.30, 3.20, 3.18, 3.27, 3.10, 3.19, 3.08, 2.83],
    [-2.75, -3.02, -1.94, -1.95, -0.33, -0.34, 2.78, 2.52, 2.38, 2.36, 2.43, 2.22, 2.34, 2.25, 1.93],
    [-2.75, -3.02, -2.08, -2.09, -0.65, -0.67, 1.67, 1.41, 1.27, 1.20, 1.24, 1.03, 1.08, 0.97, 0.66],
    [-2.75, -3.02, -2.27, -2.29, -1.10, -1.12, 0.11, -0.14, -0.21, -0.42, -0.40, -0.64, -0.64, -0.78, -1.09],
    [-2.75, -3.02, -2.37, -2.38, -1.38, -1.41, -0.75, -0.98, -1.16, -1.29, -1.28, -1.57, -1.55, -1.70, -2.01],
    [-2.75, -3.02, -2.51, -2.53, -1.88, -1.91, -1.70, -1.91, -2.10, -2.24, -2.24, -2.55, -2.54, -2.70, -3.01],
]
_CMESH_NA = [
    [8.70, 8.70, 8.70, 8.70, 8.65, 8.60, 8.50, 8.40],
    [8.78, 8.78, 8.78, 8.75, 8.72, 8.61, 8.48, 8.26],
    [8.99, 8.99, 8.99, 8.94, 8.86, 8.62, 8.42, 8.10],
    [9.13, 9.13, 9.12, 9.01, 8.91, 8.63, 8.38, 8.04],
    [9.41, 9.37, 9.31, 9.15, 9.00, 8.62, 8.34, 7.98],
    [10.11, 9.93, 9.72, 9.33, 9.06, 8.59, 8.29, 7.89],
    [10.62, 10.10, 9.72, 9.25, 8.95, 8.47, 8.17, 7.78],
    [9.95, 9.72, 9.47, 9.06, 8.78, 8.33, 8.06, 7.67],
    [9.31, 9.23, 9.11, 8.85, 8.63, 8.23, 7.96, 7.59],
    [9.00, 8.96, 8.88, 8.68, 8.49, 8.13, 7.87, 7.52],
    [8.87, 8.82, 8.75, 8.57, 8.39, 8.02, 7.79, 7.44],
    [10.22, 9.67, 9.31, 8.84, 8.55, 8.10, 7.83, 7.45],
    [9.96, 9.43, 9.07, 8.60, 8.31, 7.87, 7.61, 7.24],
    [9.26, 8.73, 8.34, 7.93, 7.63, 7.19, 6.98, 6.62],
    [11.45, 10.52, 9.86, 9.03, 8.52, 7.79, 7.35, 6.88],
    [10.76, 9.83, 9.18, 8.37, 7.86, 7.12, 6.70, 6.23],
    [10.43, 9.48, 8.84, 8.02, 7.52, 6.87, 6.36, 5.88],
    [10.95, 9.92, 9.24, 8.33, 7.77, 6.94, 6.49, 5.96],
    [10.82, 9.80, 9.12, 8.20, 7.64, 6.82, 6.36, 5.83],
    [11.32, 10.18, 9.41, 8.41, 7.79, 6.93, 6.43, 5.87],
    [11.18, 10.04, 9.27, 8.28, 7.66, 6.78, 6.29, 5.73],
]
_CMESH_MG = [
    [-0.57, -0.87, -1.20, -1.72, -2.05, -2.53, -2.82, -3.18],
    [-1.55, -1.61, -1.71, -1.99, -2.24, -2.67, -2.92, -3.26],
    [-2.08, -2.10, -2.14, -2.30, -2.48, -2.83, -3.06, -3.36],
    [-0.20, -0.89, -1.34, -1.92, -2.26, -2.74, -3.01, -3.33],
    [-0.75, -1.44, -1.89, -2.47, -2.82, -3.30, -3.55, -3.85],
    [2.14, 1.01, 0.25, -0.74, -1.36, -2.24, -2.73, -3.26],
    [2.01, 0.88, 0.14, -0.85, -1.47, -2.35, -2.82, -3.36],
    [2.02, 0.88, 0.07, -0.98, -1.65, -2.59, -3.11, -3.67],
    [3.03, 1.56, 0.57, -0.69, -1.45, -2.48, -3.03, -3.63],
    [2.74, 1.26, 0.27, -0.97, -1.68, -2.73, -3.34, -3.94],
    [2.79, 1.30, 0.32, -0.95, -1.68, -2.73, -3.33, -3.93],
    [2.49, 1.00, 0.00, -1.22, -2.04, -3.10, -3.66, -4.28],
    [2.71, 1.19, 0.17, -1.12, -1.90, -3.00, -3.58, -4.22],
    [2.59, 1.08, 0.07, -1.23, -2.02, -3.12, -3.69, -4.34],
    [2.75, 1.23, 0.20, -1.10, -1.91, -3.02, -3.63, -4.29],
    [2.74, 1.22, 0.18, -1.12, -1.92, -3.04, -3.64, -4.30],
    [3.75, 2.05, 0.88, -0.59, -1.49, -2.77, -3.43, -4.17],
    [3.67, 1.94, 0.78, -0.70, -1.60, -2.88, -3.56, -4.30],
    [3.78, 1.98, 0.82, -0.68, -1.60, -2.88, -3.56, -4.30],
    [3.61, 1.87, 0.69, -0.81, -1.73, -3.00, -3.69, -4.43],
]


def continuum_opacity(lambdas, temperature, pe, pp, refractive_index=1.0):
    """
    Continuum absorption coefficient per hydrogen nucleon.

    Parameters
    ----------
    lambdas : torch.Tensor (Nw,)
        Wavelengths [Angstrom].
    temperature : torch.Tensor (Nb, Nt, 1)
        Temperature [K].
    pe : torch.Tensor (Nb, Nt, 1)
        Electron pressure [dyn/cm^2].
    pp : PartialPressure
        Partial pressures from :func:`ionization_equilibrium`.
    refractive_index : float
        Air refractive index.

    Returns
    -------
    torch.Tensor (Nb, Nt, Nw)
        Continuum absorption coefficient per H nucleon [cm^2/H].
    """
    dev, dtp = lambdas.device, lambdas.dtype
    lambdas = lambdas.to(dev).to(dtp).reshape(-1)      # (Nw,)
    T = temperature.to(dev).to(dtp)                    # (Nb,Nt,1)
    Pe = pe.to(dev).to(dtp)
    shape = T.shape[:-1]                                # (Nb,Nt)
    Nw = lambdas.shape[0]

    # --- wavelength-dependent factors --------------------------------------
    wav_cm = lambdas * 1e-8
    wav_mu = lambdas * 1e-4
    wav_si = wav_cm * 1e2
    wav_km = wav_cm * 1e5
    wav3 = wav_cm ** 3
    freq = 2.997925e10 / (wav_cm * refractive_index)
    lnfreq = torch.log(freq)
    x10002 = 1e8 * wav_cm                          # wavelength [Angstrom]
    deltak = 911.3 / (x10002 * refractive_index)
    ephot = deltak
    ey = torch.pow(ephot, 0.43 + 0.6 * torch.log10(ephot + 10.0))
    divi1 = 1.0 + (5.9856e-2 - 3.4916e-4 * ey / (1.0 + 1e-2 * ey)) * ephot ** 0.83333333
    m0 = torch.sqrt(1.0 / ephot).to(torch.int64) + 1
    l0 = torch.clamp(m0, min=4)
    t1 = 1e-8 / wav_cm
    t2 = t1 ** 2
    t3 = t2 ** 2
    scat1 = t3 * (5.799e-13 + 1.422e-6 * t2 + 2.784 * t3)
    scat2 = t3 * (8.14e-13 + 1.28e-6 * t2 + 1.61 * t3)
    scat3 = 5.484e-14 * t3 * (1.0 + 2.44e5 * t2 + 5.94e-10 * t2 / (x10002 ** 2 - 2.9e5)) ** 2
    x10006 = 1e-34 * x10002 ** 2

    theta = 5040.0 / T
    f1 = 1.4388 / (wav_cm[None, None, :] * T)
    z1 = torch.where(f1 < 1e-3, f1, 1.0 - torch.exp(-f1))

    # --- H- ----------------------------------------------------------------
    # H- bound-free absorption cuts off at lambda <= 1.64189e-4 cm
    # (the H- photodetachment threshold); the total opacity is
    #   hminus = (cbfree * p(H-)/p(H) * z1 / Pe + 1e-26 * b4) * Pe * p(H)/p(H')
    # where cbfree is the bound-free cross-section, z1 the stimulated-
    # emission factor and b4 the free-free Gaunt factor.  The free-free
    # term 1e-26*b4 is added at ALL wavelengths, so the H- free-free
    # continuum continues beyond the bound-free threshold.
    flag_69 = (wav_cm <= 1.64189e-4) & (wav_cm > 1.42e-4)
    x = wav_km
    cbfree = 1e-17 * (6.80133e-3 + x * (1.78708e-1 + x * (1.6479e-1 - x * (2.04842e-2 - 5.95244e-4 * x))))
    x = 16.419 - wav_km
    cbfree = torch.where(flag_69, 1e-17 * x * (2.69818e-1 + x * (2.2019e-1 - x * (4.11288e-2 - 2.73236e-3 * x))), cbfree)
    cbfree = torch.where(wav_cm > 1.64189e-4, torch.zeros_like(cbfree), cbfree)
    b1 = 11.924 - 5.939 * theta
    c1 = 7.0355 - theta * 3.4592e-1
    c2 = wav_km * (-4.0192e-1 + theta * c1)
    b2 = wav_si * (-3.2062 + theta * b1 + c2)
    b3 = 2.7039e-2 * theta - 1.1493e-2 + b2
    b4 = 5.3666e-3 + theta * b3
    hminus = (cbfree * pp["h-_h"] * z1 / Pe + 1e-26 * b4) * Pe * pp["h"]

    # --- He- ---------------------------------------------------------------
    flag_73 = (deltak > 0.3) | (T < 1.5e3) | (T > 1.68e4)
    flag_71 = T < 9.2e3
    a1 = 2.46e-4 + wav_cm * (-1.26e1 + 5.67e6 * wav_cm)
    b1 = theta * (-5.92e-4 + wav_cm * (5.12e1 + 1.3e7 * wav_cm))
    c1 = theta ** 2 * (1.45e-2 - wav_cm * (8.3e1 + 1.8e6 * wav_cm))
    helmin = (a1 + b1 + c1) * 1e-26
    a1 = 9.5114e-9 - T * 2.3544e-13
    a2 = -1.0754e-4 + T * a1
    a3 = 0.49245 + T * a2
    helmin = torch.where(flag_71, x10006[None, None, :] * a3, helmin)
    helmin = helmin * Pe * pp["he"]
    helmin = torch.where(flag_73, torch.zeros_like(helmin), helmin)

    # --- H I ----------------------------------------------------------------
    flag_10 = m0 > 12
    x2 = _um(12.0, T)
    x3 = _um(1.0, T)
    suma = torch.zeros((*shape, Nw), device=dev, dtype=dtp)
    gsum = torch.zeros_like(suma)
    for i in range(1, 13):
        uuu = _um(float(i), T)
        z3 = torch.exp(uuu - x3) / float(i ** 3)
        divi2 = 1.0 + (1.72826e-1 - 3.45652e-1 / (ephot * i ** 2)) * ephot ** 0.33333333
        g_i = divi2 / divi1
        cond = m0[None, None, :] <= i
        suma = suma + torch.where(cond, g_i * z3, torch.zeros_like(g_i))
        gsum = gsum + torch.where(cond, g_i, torch.zeros_like(g_i))
    z2 = torch.full_like(T, 2.0)
    z2 = torch.where(T > 1.3e4, 1.51 + 3.8e-5 * T, z2)
    z2 = torch.where(T > 1.62e4, 11.41 + T * (-1.1428e-3 + T * 3.52e-8), z2)
    a1 = z2
    z2 = 2.08966e-2 * wav3[None, None, :] * z1 / a1
    gg = _gff(T, x10002)
    z4 = 0.5 / x3 * (torch.exp(x2 - x3) + torch.exp(-x3) * (gg - 1.0))
    hneutr = z2 * (suma + z4) * pp["h"]
    hneutr = torch.where(flag_10[None, None, :], torch.zeros_like(hneutr), hneutr)

    # --- H2- ----------------------------------------------------------------
    flag_20 = (deltak > 0.3) | (T < 1.5e3) | (T > 1.68e4)
    h2min = x10006[None, None, :] * (0.88967 + T * (-1.4274e-4 + T * (1.0879e-8 - T * 2.5658e-13)))
    h2min = h2min * Pe * pp["h2"]
    h2min = torch.where(flag_20, torch.zeros_like(h2min), h2min)

    # --- H2+ ----------------------------------------------------------------
    flag_70 = (wav_cm < 3.8e-5) | (wav_cm > 3e-4)
    exp_part1 = 2.30258509 * theta * (7.342e-3 - (-2.409e-15 + (1.028e-30 + (-4.23e-46 + (1.224e-61 - 1.351e-77 * freq) * freq) * freq) * freq) * freq)
    exp_part2 = -3.0233e3 + (3.7797e2 + (-1.82496e1 + (3.9207e-1 - 3.1672e-3 * lnfreq) * lnfreq) * lnfreq) * lnfreq
    h2plus = torch.exp(exp_part1 + exp_part2) * 1e16 * z1 * (pp["h"] * Pe) * (pp["h+"] / pp["e-"]) / (1.38054 * T)
    h2plus = torch.where(flag_70[None, None, :], torch.zeros_like(h2plus), h2plus)

    # --- He I ---------------------------------------------------------------
    heneut = torch.zeros((*shape, Nw), device=dev, dtype=dtp)
    flag_30 = wav_cm > 8.2610e-5
    i0 = torch.ones_like(wav_cm, dtype=torch.int64)
    i0 = torch.where(wav_cm > 5.0420e-6, torch.full_like(i0, 2), i0)
    i0 = torch.where(wav_cm > 2.6003e-5, torch.full_like(i0, 3), i0)
    i0 = torch.where(wav_cm > 3.1210e-5, torch.full_like(i0, 4), i0)
    i0 = torch.where(wav_cm > 3.4210e-5, torch.full_like(i0, 6), i0)
    i0 = torch.where(wav_cm > 3.6788e-5, torch.full_like(i0, 8), i0)
    i0 = torch.where(wav_cm > 6.6322e-5, torch.full_like(i0, 9), i0)
    i0 = torch.where(wav_cm > 7.4351e-5, torch.full_like(i0, 10), i0)
    i0 = torch.where(wav_cm > 7.8438e-5, torch.full_like(i0, 11), i0)
    i0 = torch.where(wav_cm > 8.1910e-5, torch.full_like(i0, 12), i0)
    for i in range(1, 13):
        mask = (i0 == i)[None, None, :]
        pepa = _g_he_contrib(i, lnfreq, theta)
        heneut = heneut + torch.where(mask, pepa, torch.zeros_like(pepa))
    flag_31 = m0 > 12
    suma2 = torch.zeros_like(suma)
    for i in range(1, 13):
        cond = l0[None, None, :] <= i
        uuu = _um(float(i), T)
        z3 = torch.exp(uuu - x3) / float(i ** 3)
        divi2 = 1.0 + (1.72826e-1 - 3.45652e-1 / (ephot * i ** 2)) * ephot ** 0.33333333
        g_i = divi2 / divi1
        suma2 = suma2 + torch.where(cond, g_i * z3, torch.zeros_like(g_i))
    flag_31 = flag_31 | (theta > 3.0) | (torch.abs(z2) < 1e-25) | (torch.abs(suma2 + z4) < 1e-25)
    pepo = 4.0 * torch.pow(10.0, -10.992 * theta) * z2 * (suma2 + z4)
    heneut_temp = torch.where(flag_30[None, None, :], pepo, heneut + pepo)
    heneut = torch.where(flag_31, heneut, heneut_temp)
    heneut = heneut * pp["he"]

    # --- scattering ----------------------------------------------------------
    flag_35 = wav_cm < 1.2e-5
    scatt1 = torch.where(flag_35, torch.zeros_like(wav_cm), scat1 * pp["h"])
    scatt2 = torch.where(flag_35, torch.zeros_like(wav_cm), scat2 * pp["h2"])
    scatt3 = torch.where(flag_35, torch.zeros_like(wav_cm), scat3 * pp["he"])
    escatt = SIGMA_THOMSON * pp["e-"]

    # --- metals: C, Na, Mg --------------------------------------------------
    carbon = _metal_opacity(wav_cm, T, z1, pp.get("c", torch.zeros(shape, device=dev, dtype=dtp)),
                            _TGRID_C, _FGRID_C, _WGRID_C, _CMESH_C, offset=20.0,
                            transpose_cmesh=True, spec=_METAL_C_SPEC)
    sodium = _metal_opacity(wav_cm, T, z1, pp.get("na", torch.zeros(shape, device=dev, dtype=dtp)),
                            _TGRID_C, _FGRID_NA, _WGRID_NA, _CMESH_NA, offset=10.0,
                            spec=_METAL_NA_SPEC)
    magnesium = _metal_opacity(wav_cm, T, z1, pp.get("mg", torch.zeros(shape, device=dev, dtype=dtp)),
                               _TGRID_C, _FGRID_MG, _WGRID_MG, _CMESH_MG, offset=20.0,
                               spec=_METAL_MG_SPEC)

    kappa = (hminus + helmin + hneutr + h2min + h2plus + heneut
             + scatt1 + scatt2 + scatt3
             + escatt + carbon + sodium + magnesium)
    return kappa


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _um(m, T):
    return 1.568399e5 / (T * m ** 2)


def _gff(T, x):
    return (1.0828 + 3.865e-6 * T + x * (7.564e-7 + (4.92e-10 - 2.482e-15 * T) * T
            + x * (5.326e-12 + (-3.904e-15 + 1.879e-20 * T) * T)))


def _g_he_contrib(i, lnfreq, theta):
    f1 = [14.47, -169.385, 11.65, 26.57, 31.059, 35.31, 35.487, 5.51,
          10.36, 21.41, 37.0, 25.54]
    f2 = [-2.0, 21.035, -1.91, -2.9, -3.3, -3.5, -3.6, -1.54, -1.86,
          -2.6, -3.69, -2.89]
    fkny = f1[i - 1] + lnfreq * f2[i - 1]
    if i == 2:
        fkny = fkny - 0.727 * lnfreq ** 2
    return _G_HE[i - 1] * torch.pow(10.0, fkny - theta * _CHI_HE[i - 1])


def _metal_opacity(wav_cm, T, z1, p_el, tg, fg, wg, cmesh, offset,
                   transpose_cmesh=False, spec=None):
    """
    Bound-free opacity of a metal (C/Na/Mg) from precomputed tables.

    Metal bound-free absorption as in the classical Stokes inversion
    reference: the temperature index ``k`` is shared by the three
    elements (they all interpolate the common ``tcatom``/``f*atom``
    grids), but each element has its own mesh-column ("l") mapping and
    its own normalisation constants.  The ``spec`` argument holds, per
    element:

      * C  : k<=5  -> l=k-1, y normalized in the interval;
             6<=k<=7 -> l=5, (y-f(5))/2.542 ;  8<=k<=12 -> l=6, (y-f(7))/3.496
             13<=k<=17 -> l=7, (y-f(12))/1.85 ;  k>17 -> l=8, (y-f(17))/2.002
      * Na : k<=3  -> l=k-1, y normalized in the interval;
             4<=k<=5 -> l=3, (y-f(3))/1.987 ;  6<=k<=7 -> l=4, (y-f(5))/1.238
             8<=k<=12 -> l=5, (y-f(7))/1.762 ;  13<=k<=17 -> l=6, (y-f(12))/0.97
             k>17 -> l=7, (y-f(17))/1.114
      * Mg : k<=3  -> l=k-1, y normalized in the interval;
             4<=k<=5 -> l=3, (y-f(3))/2.992 ;  6<=k<=7 -> l=4, (y-f(5))/1.842
             8<=k<=12 -> l=5, (y-f(7))/2.556 ;  13<=k<=17 -> l=6, (y-f(12))/1.362
             k>17 -> l=7, (y-f(17))/1.496

    ``spec`` is a tuple of (threshold_tidx, l_low, l_high, fg_index, norm)
    entries with ``threshold_tidx`` in 0-based index units (``tidx = k-1``)
    and the original reference ``l`` values converted to 0-based mesh
    columns.

    Returns (Nb, Nt, Nw) per-H opacity.
    """
    dev, dtp = T.device, T.dtype
    shape = T.shape[:-1]                                # (Nb,Nt)
    Nw = wav_cm.shape[0]
    out = torch.zeros((*shape, Nw), device=dev, dtype=dtp)

    t_ok = (T > tg[0]) & (T < tg[-1])
    w_ok = (wav_cm >= wg[0]) & (wav_cm < wg[-1])
    # no early return: everything is masked out-of-place below, which keeps
    # the routine free of data-dependent control flow (vmap-compatible)

    tg_t = torch.tensor(tg, device=dev, dtype=dtp)
    fg_t = torch.tensor(fg, device=dev, dtype=dtp)
    wg_t = torch.tensor(wg, device=dev, dtype=dtp)
    cm_t = torch.tensor(cmesh, device=dev, dtype=dtp)
    if transpose_cmesh:
        # the C table is stored (nT, nw) like the Fortran data block
        cm_t = cm_t.t().contiguous()                    # (nw, nT)

    # temperature index: 0-based, == the 1-based grid index k minus 1
    tidx = torch.searchsorted(tg_t, T.contiguous()).clamp(1, len(tg) - 2)
    fh = fg_t[tidx - 1]
    fi = fg_t[tidx]
    fj = fg_t[tidx + 1]
    th = tg_t[tidx - 1]
    ti = tg_t[tidx]
    tj = tg_t[tidx + 1]
    y0 = _lagrange(T, fh, fi, fj, th, ti, tj)

    # mesh column / normalisation: start from the "in-interval" branch
    # (l = k-1, y = (y0-f(k-1))/(f(k)-f(k-1))), then apply the
    # per-element ranges (from the end; the last matching range wins).
    l = tidx - 1                                      # 0-based l = k-2
    y = (y0 - fg_t[tidx - 1]) / torch.clamp(fg_t[tidx] - fg_t[tidx - 1], min=1e-30)
    for thr, l_lo, l_hi, fidx, norm in reversed(spec or ()):
        sel = tidx > thr
        l = torch.where(sel, torch.full_like(l, l_lo), l)
        y = torch.where(sel, (y0 - fg_t[fidx]) / norm, y)

    # wavelength index (1-based, may reach the last grid point); keep the
    # (1,1,Nw) shape so the table lookup broadcasts to (Nb, Nt, Nw)
    wav3 = wav_cm[None, None, :]
    widx = torch.searchsorted(wg_t, wav3).clamp(1, len(wg) - 1)
    y2 = (wav3 - wg_t[widx - 1]) / (wg_t[widx] - wg_t[widx - 1])
    c1 = cm_t[widx - 1, l] + y2 * (cm_t[widx, l] - cm_t[widx - 1, l])
    c2 = cm_t[widx - 1, l + 1] + y2 * (cm_t[widx, l + 1] - cm_t[widx - 1, l + 1])
    # y has shape (Nb,Nt,1); c1/c2 broadcast with (Nb,Nt,Nw)
    val = z1 * p_el * torch.pow(10.0, y * (c1 - c2) - c1 - offset)
    val = torch.where(t_ok & w_ok[None, None, :], val, torch.zeros_like(val))
    return val


# Per-element spec for _metal_opacity (metal bound-free mesh mapping):
#   (threshold 0-based tidx, l_low (0-based mesh col), l_high, fg index, norm)
_METAL_C_SPEC = ((4, 4, 5, 4, 2.542), (6, 5, 6, 6, 3.496), (11, 6, 7, 11, 1.85),
                 (16, 7, 8, 16, 2.002))
_METAL_NA_SPEC = ((2, 2, 3, 2, 1.987), (4, 3, 4, 4, 1.238), (6, 4, 5, 6, 1.762),
                  (11, 5, 6, 11, 0.970), (16, 6, 7, 16, 1.114))
_METAL_MG_SPEC = ((2, 2, 3, 2, 2.992), (4, 3, 4, 4, 1.842), (6, 4, 5, 6, 2.556),
                  (11, 5, 6, 11, 1.362), (16, 6, 7, 16, 1.496))


def _lagrange(x, yh, yi, yj, xh, xi, xj):
    """3-point Lagrange interpolation of the metal bound-free tables:
    FINT = Y1*D2*D3/(D12*D13) - Y2*D1*D3/(D12*D23) + Y3*D1*D2/(D13*D23).

    NOTE: the middle term carries a MINUS and the (D12*D23) denominator
    (equivalent to the standard second Lagrange weight (x-x1)(x-x3)/
    ((x2-x1)(x2-x3))).  The previous implementation used the wrong
    denominator/sign for the middle term, which scattered the metal
    (C/Na/Mg) bound-free opacity off by orders of magnitude at all
    temperatures.
    """
    d1 = x - xh
    d2 = x - xi
    d3 = x - xj
    d12 = xh - xi
    d13 = xh - xj
    d23 = xi - xj
    return (yh * d2 * d3 / (d12 * d13) - yi * d1 * d3 / (d12 * d23)
            + yj * d1 * d2 / (d13 * d23))


# ---------------------------------------------------------------------------
# continuum opacity derivatives  (d kappa/dT, d kappa/dPe)
# ---------------------------------------------------------------------------
def continuum_opacity_derivatives(lambdas, temperature, pe, pp, dpp, ddpp,
                                   refractive_index=1.0):
    """
    Continuum opacity and its derivatives d kappa/dT, d kappa/dPe.

    The analytic derivatives feed the linearised radiative transfer and
    the inversion's response-matrix step; ``dpp``/``ddpp`` are the gasb
    log-derivatives (d ln p/dT, d ln p/dPe) for the same PartialPressure
    keys.

    Returns (kappa, dkappa, ddkappa), all (Nb, Nt, Nw).
    """
    dev, dtp = lambdas.device, lambdas.dtype
    lambdas = lambdas.to(dev).to(dtp).reshape(-1)      # (Nw,)
    T = temperature.to(dev).to(dtp)                    # (Nb,Nt,1)
    Pe = pe.to(dev).to(dtp)
    shape = T.shape[:-1]
    Nw = lambdas.shape[0]

    wav_cm = lambdas * 1e-8
    wav_mu = lambdas * 1e-4
    wav_si = wav_cm * 1e2
    wav_km = wav_cm * 1e5
    wav3 = wav_cm ** 3
    freq = 2.997925e10 / (wav_cm * refractive_index)
    lnfreq = torch.log(freq)
    x10002 = 1e8 * wav_cm
    deltak = 911.3 / (x10002 * refractive_index)
    ephot = deltak
    ey = torch.pow(ephot, 0.43 + 0.6 * torch.log10(ephot + 10.0))
    divi1 = 1.0 + (5.9856e-2 - 3.4916e-4 * ey / (1.0 + 1e-2 * ey)) * ephot ** 0.83333333
    m0 = torch.sqrt(1.0 / ephot).to(torch.int64) + 1
    l0 = torch.clamp(m0, min=4)
    t1 = 1e-8 / wav_cm
    t2 = t1 ** 2
    t3 = t2 ** 2
    scat1 = t3 * (5.799e-13 + 1.422e-6 * t2 + 2.784 * t3)
    scat2 = t3 * (8.14e-13 + 1.28e-6 * t2 + 1.61 * t3)
    scat3 = 5.484e-14 * t3 * (1.0 + 2.44e5 * t2 + 5.94e-10 * t2 / (x10002 ** 2 - 2.9e5)) ** 2
    x10006 = 1e-34 * x10002 ** 2

    theta = 5040.0 / T
    dtheta = -5040.0 / (T * T)                          # d theta/dT
    f1 = 1.4388 / (wav_cm[None, None, :] * T)
    df1 = -f1 / T
    z1 = torch.where(f1 < 1e-3, f1, 1.0 - torch.exp(-f1))
    dz1 = torch.where(f1 < 1e-3, df1, df1 * torch.exp(-f1))

    # --- H- ----------------------------------------------------------------
    flag_69 = (wav_cm <= 1.64189e-4) & (wav_cm > 1.42e-4)
    x = wav_km
    cbfree = 1e-17 * (6.80133e-3 + x * (1.78708e-1 + x * (1.6479e-1 - x * (2.04842e-2 - 5.95244e-4 * x))))
    x = 16.419 - wav_km
    cbfree = torch.where(flag_69, 1e-17 * x * (2.69818e-1 + x * (2.2019e-1 - x * (4.11288e-2 - 2.73236e-3 * x))), cbfree)
    cbfree = torch.where(wav_cm > 1.64189e-4, torch.zeros_like(cbfree), cbfree)
    b1 = 11.924 - 5.939 * theta
    db1 = -5.939 * dtheta
    c1 = 7.0355 - theta * 3.4592e-1
    dc1 = -3.4592e-1 * dtheta
    c2 = wav_km * (-4.0192e-1 + theta * c1)
    dc2 = wav_km * (dtheta * c1 + theta * dc1)
    b2 = wav_si * (-3.2062 + theta * b1 + c2)
    db2 = wav_si * (dtheta * b1 + theta * db1 + dc2)
    b3 = 2.7039e-2 * theta - 1.1493e-2 + b2
    db3 = 2.7039e-2 * dtheta + db2
    b4 = 5.3666e-3 + theta * b3
    db4 = dtheta * b3 + theta * db3
    hminus = (cbfree * pp["h-_h"] * z1 / Pe + 1e-26 * b4) * Pe * pp["h"]
    dhminus = (cbfree * (pp["h-_h"] * dpp["h-_h"] * z1 + pp["h-_h"] * dz1)
               / Pe + 1e-26 * db4) * Pe * pp["h"] \
        + hminus * dpp["h"]
    ddhminus = (cbfree * pp["h-_h"] * z1 / Pe * (ddpp["h-_h"] - 1.0 / Pe))
    ddhminus = ddhminus * Pe * pp["h"] + hminus * ddpp["h"] + hminus / Pe

    # --- He- ---------------------------------------------------------------
    flag_73 = (deltak > 0.3) | (T < 1.5e3) | (T > 1.68e4)
    flag_71 = T < 9.2e3
    a1 = 2.46e-4 + wav_cm * (-1.26e1 + 5.67e6 * wav_cm)
    b1 = theta * (-5.92e-4 + wav_cm * (5.12e1 + 1.3e7 * wav_cm))
    c1 = theta ** 2 * (1.45e-2 - wav_cm * (8.3e1 + 1.8e6 * wav_cm))
    helmin = (a1 + b1 + c1) * 1e-26
    dhelmin = (db1 + dc1) * 1e-26
    a1 = 9.5114e-9 - T * 2.3544e-13
    a2 = -1.0754e-4 + T * a1
    a3 = 0.49245 + T * a2
    da1 = -2.3544e-13
    da2 = a1 + T * da1
    da3 = a2 + T * da2
    helmin = torch.where(flag_71, x10006[None, None, :] * a3, helmin)
    dhelmin = torch.where(flag_71, x10006[None, None, :] * da3, dhelmin)
    helmin = helmin * Pe * pp["he"]
    dhelmin = dhelmin * Pe * pp["he"] + helmin * dpp["he"]
    dhelmin = torch.where(flag_73, torch.zeros_like(dhelmin), dhelmin)
    dhelmin = torch.where(flag_73, torch.zeros_like(dhelmin), dhelmin)
    dhelmin_p = helmin * ddpp["he"] + helmin / Pe
    dhelmin_p = torch.where(flag_73, torch.zeros_like(dhelmin_p), dhelmin_p)
    helmin = torch.where(flag_73, torch.zeros_like(helmin), helmin)

    # --- H I ----------------------------------------------------------------
    flag_10 = m0 > 12
    x2 = _um(12.0, T)
    x3 = _um(1.0, T)
    dx2 = -x2 / T
    dx3 = -x3 / T
    suma = torch.zeros((*shape, Nw), device=dev, dtype=dtp)
    dsum = torch.zeros_like(suma)
    for i in range(1, 13):
        uuu = _um(float(i), T)
        z3 = torch.exp(uuu - x3) / float(i ** 3)
        dz3 = z3 * (-uuu / T + x3 / T)            # d ln z3/dT = du/dT - dx3/dT
        divi2 = 1.0 + (1.72826e-1 - 3.45652e-1 / (ephot * i ** 2)) * ephot ** 0.33333333
        g_i = divi2 / divi1
        cond = m0[None, None, :] <= i
        suma = suma + torch.where(cond, g_i * z3, torch.zeros_like(g_i))
        dsum = dsum + torch.where(cond, g_i * dz3, torch.zeros_like(g_i))
    z2 = torch.full_like(T, 2.0)
    dz2 = torch.zeros_like(T)
    z2 = torch.where(T > 1.3e4, 1.51 + 3.8e-5 * T, z2)
    dz2 = torch.where(T > 1.3e4, 3.8e-5, dz2)
    z2 = torch.where(T > 1.62e4, 11.41 + T * (-1.1428e-3 + T * 3.52e-8), z2)
    dz2 = torch.where(T > 1.62e4, -1.1428e-3 + 2.0 * T * 3.52e-8, dz2)
    a1 = z2
    da1 = dz2
    z2 = 2.08966e-2 * wav3[None, None, :] * z1 / a1
    dzz2 = dz1 / z1 - da1 / a1
    dz2 = z2 * dzz2
    gg = _gff(T, x10002)
    dgg = _dgff(T, x10002)
    z4 = 0.5 / x3 * (torch.exp(x2 - x3) + torch.exp(-x3) * (gg - 1.0))
    dz4 = -z4 * dx3 / x3 + 0.5 / x3 * (
        torch.exp(x2 - x3) * (dx2 - dx3)
        + torch.exp(-x3) * (-dx3 * (gg - 1.0) + dgg))
    hneutr = z2 * (suma + z4) * pp["h"]
    dhneutr = hneutr * (dzz2 + dpp["h"]
                        + (dsum + dz4) / (suma + z4))
    dhneutr = torch.where(torch.abs(suma + z4) <= 1e-30,
                          hneutr * (dzz2 + dpp["h"]), dhneutr)
    ddhneutr = hneutr * ddpp["h"]
    hneutr = torch.where(flag_10[None, None, :], torch.zeros_like(hneutr), hneutr)
    dhneutr = torch.where(flag_10[None, None, :], torch.zeros_like(dhneutr), dhneutr)
    ddhneutr = torch.where(flag_10[None, None, :], torch.zeros_like(ddhneutr), ddhneutr)

    # --- H2- ----------------------------------------------------------------
    flag_20 = (deltak > 0.3) | (T < 1.5e3) | (T > 1.68e4)
    h2min = x10006[None, None, :] * (0.88967 + T * (-1.4274e-4 + T * (1.0879e-8 - T * 2.5658e-13)))
    dh2min = x10006[None, None, :] * (-1.4274e-4 + 2.0 * T * (1.0879e-8 - 3.0 * T * 2.5658e-13))
    h2min = h2min * Pe * pp["h2"]
    dh2min = dh2min * Pe * pp["h2"] + h2min * dpp["h2"]
    ddh2min = h2min * (1.0 / Pe + ddpp["h2"])
    h2min = torch.where(flag_20, torch.zeros_like(h2min), h2min)
    dh2min = torch.where(flag_20, torch.zeros_like(dh2min), dh2min)
    ddh2min = torch.where(flag_20, torch.zeros_like(ddh2min), ddh2min)

    # --- H2+ ----------------------------------------------------------------
    flag_70 = (wav_cm < 3.8e-5) | (wav_cm > 3e-4)
    exp_part1 = 2.30258509 * theta * (7.342e-3 - (-2.409e-15 + (1.028e-30 + (-4.23e-46 + (1.224e-61 - 1.351e-77 * freq) * freq) * freq) * freq) * freq)
    exp_part2 = -3.0233e3 + (3.7797e2 + (-1.82496e1 + (3.9207e-1 - 3.1672e-3 * lnfreq) * lnfreq) * lnfreq) * lnfreq
    h2plus = torch.exp(exp_part1 + exp_part2) * 1e16 * z1 * (pp["h"] * Pe) * (pp["h+"] / pp["e-"]) / (1.38054 * T)
    d1 = 2.30258509 * dtheta * (7.342e-3 - (-2.409e-15 + (1.028e-30 + (-4.23e-46 + (1.224e-61 - 1.351e-77 * freq) * freq) * freq) * freq) * freq)
    dh2plus = h2plus * (d1 + dz1 / z1 + dpp["h"] + (dpp["h+"] - dpp["e-"]) - 1.0 / T)
    ddh2plus = h2plus * (ddpp["h"] + (ddpp["h+"] - ddpp["e-"]) + 1.0 / Pe)
    h2plus = torch.where(flag_70[None, None, :], torch.zeros_like(h2plus), h2plus)
    dh2plus = torch.where(flag_70[None, None, :], torch.zeros_like(dh2plus), dh2plus)
    ddh2plus = torch.where(flag_70[None, None, :], torch.zeros_like(ddh2plus), ddh2plus)

    # --- He I ---------------------------------------------------------------
    heneut = torch.zeros((*shape, Nw), device=dev, dtype=dtp)
    dheneut = torch.zeros_like(heneut)
    flag_30 = wav_cm > 8.2610e-5
    i0 = torch.ones_like(wav_cm, dtype=torch.int64)
    i0 = torch.where(wav_cm > 5.0420e-6, torch.full_like(i0, 2), i0)
    i0 = torch.where(wav_cm > 2.6003e-5, torch.full_like(i0, 3), i0)
    i0 = torch.where(wav_cm > 3.1210e-5, torch.full_like(i0, 4), i0)
    i0 = torch.where(wav_cm > 3.4210e-5, torch.full_like(i0, 6), i0)
    i0 = torch.where(wav_cm > 3.6788e-5, torch.full_like(i0, 8), i0)
    i0 = torch.where(wav_cm > 6.6322e-5, torch.full_like(i0, 9), i0)
    i0 = torch.where(wav_cm > 7.4351e-5, torch.full_like(i0, 10), i0)
    i0 = torch.where(wav_cm > 7.8438e-5, torch.full_like(i0, 11), i0)
    i0 = torch.where(wav_cm > 8.1910e-5, torch.full_like(i0, 12), i0)
    for i in range(1, 13):
        mask = (i0 == i)[None, None, :]
        pepa = _g_he_contrib(i, lnfreq, theta)
        dpepa = pepa * (-dtheta * _CHI_HE[i - 1]) * 2.3025851
        heneut = heneut + torch.where(mask, pepa, torch.zeros_like(pepa))
        dheneut = dheneut + torch.where(mask, dpepa, torch.zeros_like(dpepa))
    flag_31 = m0 > 12
    suma2 = torch.zeros_like(suma)
    dsum2 = torch.zeros_like(suma)
    for i in range(1, 13):
        cond = l0[None, None, :] <= i
        uuu = _um(float(i), T)
        z3 = torch.exp(uuu - x3) / float(i ** 3)
        dz3 = z3 * (-uuu / T + x3 / T)
        divi2 = 1.0 + (1.72826e-1 - 3.45652e-1 / (ephot * i ** 2)) * ephot ** 0.33333333
        g_i = divi2 / divi1
        suma2 = suma2 + torch.where(cond, g_i * z3, torch.zeros_like(g_i))
        dsum2 = dsum2 + torch.where(cond, g_i * dz3, torch.zeros_like(g_i))
    flag_31 = flag_31 | (theta > 3.0) | (torch.abs(z2) < 1e-25) | (torch.abs(suma2 + z4) < 1e-25)
    pepo = 4.0 * torch.pow(10.0, -10.992 * theta) * z2 * (suma2 + z4)
    dpepo = pepo * (-2.3025851 * 10.992 * dtheta + dz2 / z2
                    + (dsum2 + dz4) / (suma2 + z4))
    heneut_temp = torch.where(flag_30[None, None, :], pepo, heneut + pepo)
    dheneut_temp = torch.where(flag_30[None, None, :], dpepo,
                               dheneut + dpepo)
    heneut = torch.where(flag_31, heneut, heneut_temp)
    dheneut = torch.where(flag_31, dheneut, dheneut_temp)
    heneut = heneut * pp["he"]
    dheneut = dheneut * pp["he"] + heneut * dpp["he"]
    ddhneut_p = heneut * ddpp["he"]

    # --- scattering ----------------------------------------------------------
    flag_35 = wav_cm < 1.2e-5
    scatt1 = torch.where(flag_35, torch.zeros_like(wav_cm), scat1 * pp["h"])
    scatt2 = torch.where(flag_35, torch.zeros_like(wav_cm), scat2 * pp["h2"])
    scatt3 = torch.where(flag_35, torch.zeros_like(wav_cm), scat3 * pp["he"])
    dscatt1 = torch.where(flag_35, torch.zeros_like(wav_cm), scatt1 * dpp["h"])
    dscatt2 = torch.where(flag_35, torch.zeros_like(wav_cm), scatt2 * dpp["h2"])
    dscatt3 = torch.where(flag_35, torch.zeros_like(wav_cm), scatt3 * dpp["he"])
    ddscatt1 = torch.where(flag_35, torch.zeros_like(wav_cm), scatt1 * ddpp["h"])
    ddscatt2 = torch.where(flag_35, torch.zeros_like(wav_cm), scatt2 * ddpp["h2"])
    ddscatt3 = torch.where(flag_35, torch.zeros_like(wav_cm), scatt3 * ddpp["he"])
    escatt = 6.653e-25 * pp["e-"]
    descatt = escatt * dpp["e-"]
    ddescatt = escatt * ddpp["e-"]

    # --- metals: C, Na, Mg ------------------------------------------------
    carbon, dcarbon, ddcarbon = _metal_opacity_derivatives(
        wav_cm, T, theta, z1, dz1, pp.get("c", torch.zeros(shape, device=dev, dtype=dtp)),
        dpp.get("c", torch.zeros(shape, device=dev, dtype=dtp)),
        ddpp.get("c", torch.zeros(shape, device=dev, dtype=dtp)),
        _TGRID_C, _FGRID_C, _WGRID_C, _CMESH_C, offset=20.0,
        transpose_cmesh=True)
    sodium, dsodium, ddsodium = _metal_opacity_derivatives(
        wav_cm, T, theta, z1, dz1, pp.get("na", torch.zeros(shape, device=dev, dtype=dtp)),
        dpp.get("na", torch.zeros(shape, device=dev, dtype=dtp)),
        ddpp.get("na", torch.zeros(shape, device=dev, dtype=dtp)),
        _TGRID_C, _FGRID_NA, _WGRID_NA, _CMESH_NA, offset=10.0)
    magnesium, dmgatom, ddmgatom = _metal_opacity_derivatives(
        wav_cm, T, theta, z1, dz1, pp.get("mg", torch.zeros(shape, device=dev, dtype=dtp)),
        dpp.get("mg", torch.zeros(shape, device=dev, dtype=dtp)),
        ddpp.get("mg", torch.zeros(shape, device=dev, dtype=dtp)),
        _TGRID_C, _FGRID_MG, _WGRID_MG, _CMESH_MG, offset=20.0)

    kappa = (hminus + helmin + hneutr + h2min + h2plus + heneut
             + scatt1 + scatt2 + scatt3
             + escatt + carbon + sodium + magnesium)
    dkappa = (dhminus + dhelmin + dhneutr + dh2min + dh2plus + dheneut
              + dscatt1 + dscatt2 + dscatt3
              + descatt + dcarbon + dsodium + dmgatom)
    ddkappa = (ddhminus + dhelmin_p + ddhneutr + ddh2min + ddh2plus
               + ddhneut_p
               + ddscatt1 + ddscatt2 + ddscatt3
               + ddescatt + ddcarbon + ddsodium + ddmgatom)
    return kappa, dkappa, ddkappa


def _dgff(T, x):
    """d gff/dT: temperature derivative of the free-free Gaunt factor."""
    return (3.865e-6 + x * (4.92e-10 - 2.0 * 2.482e-15 * T
            + x * (-3.904e-15 + 2.0 * T * 1.879e-20)))


def _metal_opacity_derivatives(wav_cm, T, theta, z1, dz1, p_el, dp_el,
                               ddp_el, tg, fg, wg, cmesh, offset,
                               transpose_cmesh=False):
    """
    Metal bound-free opacity and its temperature/electron-pressure
    derivatives (carbon/sodium/magnesium).

    Returns (val, dval_dT, dval_dPe).
    """
    dev, dtp = T.device, T.dtype
    shape = T.shape[:-1]
    Nw = wav_cm.shape[0]

    t_ok = (T > tg[0]) & (T < tg[-1])
    w_ok = (wav_cm >= wg[0]) & (wav_cm < wg[-1])

    tg_t = torch.tensor(tg, device=dev, dtype=dtp)
    fg_t = torch.tensor(fg, device=dev, dtype=dtp)
    wg_t = torch.tensor(wg, device=dev, dtype=dtp)
    cm_t = torch.tensor(cmesh, device=dev, dtype=dtp)
    if transpose_cmesh:
        cm_t = cm_t.t().contiguous()

    tidx = torch.searchsorted(tg_t, T.contiguous()).clamp(1, len(tg) - 2)
    fh = fg_t[tidx - 1]
    fi = fg_t[tidx]
    fj = fg_t[tidx + 1]
    th = tg_t[tidx - 1]
    ti = tg_t[tidx]
    tj = tg_t[tidx + 1]
    y0 = _lagrange(T, fh, fi, fj, th, ti, tj)
    # d y0/dT via Lagrange coefficients (linear-in-T on each segment)
    d12 = th - ti
    d13 = th - tj
    d23 = ti - tj
    dy0 = (fh * ((T - ti) + (T - tj)) / (d12 * d13)
           + fi * ((T - th) + (T - tj)) / (d23 * d13)
           + fj * ((T - th) + (T - ti)) / (d12 * d23))
    l = torch.clamp(tidx - 1, max=1)
    l = torch.where(tidx > 2, torch.full_like(l, 2), l)
    l = torch.where(tidx > 4, torch.full_like(l, 3), l)
    l = torch.where(tidx > 6, torch.full_like(l, 4), l)
    l = torch.where(tidx > 11, torch.full_like(l, 5), l)
    l = torch.where(tidx > 16, torch.full_like(l, 6), l)
    y = torch.where(tidx <= 2, (y0 - fg_t[tidx - 1]) / (fg_t[tidx] - fg_t[tidx - 1]), y0)
    y = torch.where(tidx > 2, (y0 - fg_t[2]) / 1.987, y)
    y = torch.where(tidx > 4, (y0 - fg_t[4]) / 1.238, y)
    y = torch.where(tidx > 6, (y0 - fg_t[6]) / 1.762, y)
    y = torch.where(tidx > 11, (y0 - fg_t[11]) / 0.970, y)
    y = torch.where(tidx > 16, (y0 - fg_t[16]) / 1.114, y)
    denom_y = torch.where(tidx <= 2, fg_t[tidx] - fg_t[tidx - 1],
                          torch.ones_like(fg_t[0]) * 0.0)
    dy = dy0 / denom_y
    dy = torch.where(tidx > 2, dy0 / 1.987, dy)
    dy = torch.where(tidx > 4, dy0 / 1.238, dy)
    dy = torch.where(tidx > 6, dy0 / 1.762, dy)
    dy = torch.where(tidx > 11, dy0 / 0.970, dy)
    dy = torch.where(tidx > 16, dy0 / 1.114, dy)

    wav3 = wav_cm[None, None, :]
    widx = torch.searchsorted(wg_t, wav3).clamp(1, len(wg) - 1)
    y2 = (wav3 - wg_t[widx - 1]) / (wg_t[widx] - wg_t[widx - 1])
    c1 = cm_t[widx - 1, l] + y2 * (cm_t[widx, l] - cm_t[widx - 1, l])
    c2 = cm_t[widx - 1, l + 1] + y2 * (cm_t[widx, l + 1] - cm_t[widx - 1, l + 1])
    val = z1 * p_el * torch.pow(10.0, y * (c1 - c2) - c1 - offset)
    # NOTE: the metal bound-free opacity derivatives retain ONLY the
    # dz1/z1 + dp_el (and dd p_el) terms; the temperature dependence of
    # the table interpolation y(T) is deliberately NOT differentiated
    # (dval = val*(dz1/z1+dp_el), ddval = val*ddp_el).  Keep identical.
    dval = val * (dz1 / z1 + dp_el)
    ddval = val * (ddp_el)
    val = torch.where(t_ok & w_ok[None, None, :], val, torch.zeros_like(val))
    dval = torch.where(t_ok & w_ok[None, None, :], dval, torch.zeros_like(dval))
    ddval = torch.where(t_ok & w_ok[None, None, :], ddval, torch.zeros_like(ddval))
    return val, dval, ddval


# ---------------------------------------------------------------------------
# ATLAS / Opacity Project recipes (ported from pyPRT-dsh synthesis.py)
# ---------------------------------------------------------------------------
# Both recipes return the absorption coefficient PER HYDROGEN NUCLEON
# [cm^2/H], exactly like :func:`continuum_opacity` (mihalas).  pyPRT's
# conversion of the table value is
#     kappaC = 10**interp(logT, logPg|logR) * rg / NH,
#     NH     = p(H') / (kB T)                       [cm^-3]
#     amw    = Sum(abu*wgt) / Sum(abu)              [mean molecular weight]
#     rg     = Pg / (T * Rg_gas / amw)              [g/cm^3] (gas density)
# with Rg_gas = const.R * 1e7 (erg/mol/K, cgs) and p(H') the partial
# pressure of hydrogen nuclei.  The same conversion is implemented
# below; spot's PartialPressure exposes both ``pg`` and ``h_prime``.

_RG_CGS = 8.314462618e7           # R [erg/(mol K)]  (scipy.constants.R*1e7)
_KB_CGS = 1.380649e-16            # kB [erg/K]      (scipy.constants.k*1e7)

# Sun-like abundance fractions used to pick the Opacity Project table
# (default; recomputed from the spot THEVENIN table when the recipe
# is first used — see _op_xyz).
_OP_DEFAULT_XYZ = (0.738, 0.249, 0.0129)

_OP_XYZ_CACHE = {}


def _op_xyz():
    """(X, Y, Z) mass fractions of the spot default abundance mix.

    pyPRT: X = 1/amw, Y = X*abu_He*wgt_He, Z = 1-X-Y (its combined
    amw includes He).  Here the same definition is evaluated with the
    spot THEVENIN abundances.
    """
    if "xyz" in _OP_XYZ_CACHE:
        return _OP_XYZ_CACHE["xyz"]
    from .atoms import ATOM_WEIGHTS, AbundanceTable
    abu = AbundanceTable.from_default().relative       # N(X)/N(H)
    amw = float(np.sum(abu * ATOM_WEIGHTS))
    x = 1.0 / amw
    y = x * abu[1] * ATOM_WEIGHTS[1]                   # He (z=2 -> index 1)
    z = 1.0 - x - y
    xyz = (float(x), float(y), float(z))
    _OP_XYZ_CACHE["xyz"] = xyz
    return xyz


def _interp3(x, y, z, xg, yg, zg, f):
    """Trilinear interpolation of a regular 3-D grid ``f`` (nx, ny, nz).

    ``x``/``y``/``z`` are same-shaped tensors, e.g. (Nb, Nt, 1); grid
    coordinates must be increasing and out-of-range points are clamped
    to the boundary.  Returns the same shape as the coordinates.
    """
    def _idx(coord, grid):
        n = grid.shape[0]
        i = torch.searchsorted(grid, coord.contiguous()).clamp(1, n - 1)
        return i - 1, i

    ix0, ix1 = _idx(x, xg)
    iy0, iy1 = _idx(y, yg)
    iz0, iz1 = _idx(z, zg)

    def _w(coord, c0, c1, grid):
        lo, hi = grid[c0], grid[c1]
        denom = torch.clamp(hi - lo, min=1e-30)
        return (coord - lo) / denom

    wx = _w(x, ix0, ix1, xg)      # same shape as x
    wy = _w(y, iy0, iy1, yg)
    wz = _w(z, iz0, iz1, zg)
    # gather the 8 corners (advanced indexing keeps the coord shape)
    out = (1 - wx) * (1 - wy) * (1 - wz) * f[ix0, iy0, iz0] \
        + wx * (1 - wy) * (1 - wz) * f[ix1, iy0, iz0] \
        + (1 - wx) * wy * (1 - wz) * f[ix0, iy1, iz0] \
        + wx * wy * (1 - wz) * f[ix1, iy1, iz0] \
        + (1 - wx) * (1 - wy) * wz * f[ix0, iy0, iz1] \
        + wx * (1 - wy) * wz * f[ix1, iy0, iz1] \
        + (1 - wx) * wy * wz * f[ix0, iy1, iz1] \
        + wx * wy * wz * f[ix1, iy1, iz1]
    return out


def _interp2(x, y, xg, yg, f):
    """Bilinear interpolation of a regular 2-D grid ``f`` (nx, ny)."""
    def _idx(coord, grid):
        n = grid.shape[0]
        i = torch.searchsorted(grid, coord.contiguous()).clamp(1, n - 1)
        return i - 1, i

    ix0, ix1 = _idx(x, xg)
    iy0, iy1 = _idx(y, yg)

    def _w(coord, c0, c1, grid):
        lo, hi = grid[c0], grid[c1]
        denom = torch.clamp(hi - lo, min=1e-30)
        return (coord - lo) / denom

    wx = _w(x, ix0, ix1, xg)
    wy = _w(y, iy0, iy1, yg)
    out = (1 - wx) * (1 - wy) * f[ix0, iy0] \
        + wx * (1 - wy) * f[ix1, iy0] \
        + (1 - wx) * wy * f[ix0, iy1] \
        + wx * wy * f[ix1, iy1]
    return out


# ---------------------------------------------------------------------------
# ATLAS solar ODF table (lazy-loaded once)
# ---------------------------------------------------------------------------
_ATLAS_CACHE = {}


def _load_atlas_tables():
    """Load and cache the ATLAS Rosseland table (logT, logPg, vmic)."""
    if "grids" in _ATLAS_CACHE:
        return _ATLAS_CACHE["grids"], _ATLAS_CACHE["values"]
    from ..utils.data_io import resource_path
    path = os.path.join(resource_path("opacity_tables"),
                        "ATLAS_solar_ODF.txt")
    data = np.loadtxt(path, skiprows=2)
    logt = np.unique(data[:, 0])
    logp = np.unique(data[:, 1])
    vmic = np.array([0.0, 1.0, 2.0, 4.0, 8.0])
    nt, np_, nv = len(logt), len(logp), len(vmic)
    kapp = data[:, 2:7].reshape(nt, np_, nv)          # (nT, nP, nV)
    _ATLAS_CACHE["grids"] = (torch.as_tensor(logt, dtype=torch.float32),
                             torch.as_tensor(logp, dtype=torch.float32),
                             torch.as_tensor(vmic, dtype=torch.float32))
    _ATLAS_CACHE["values"] = torch.as_tensor(kapp, dtype=torch.float32)
    return _ATLAS_CACHE["grids"], _ATLAS_CACHE["values"]


# ---------------------------------------------------------------------------
# Opacity Project tables (lazy-loaded once, sun-like X/Y/Z default)
# ---------------------------------------------------------------------------
_OP_CACHE = {}


def _load_op_tables(xyz=None):
    """Load and cache the Opacity Project table nearest to ``xyz``.

    The table selection follows pyPRT: |XYZ - (X,Y,Z)| .sum() is
    minimised over the ``OPtab_summaries.csv`` catalogue.  ``xyz``
    defaults to the spot THEVENIN composition.
    """
    if xyz is None:
        xyz = _op_xyz()
    key = tuple(round(v, 4) for v in xyz)
    if key in _OP_CACHE:
        return _OP_CACHE[key]
    from ..utils.data_io import resource_path
    base = resource_path("opacity_tables")
    summ = np.loadtxt(os.path.join(base, "OPtab_summaries.csv"),
                      delimiter=",", skiprows=1, usecols=(0, 1, 2, 3))
    tabs = summ[:, 0].astype(int)
    xyzs = summ[:, 1:4]
    dist = np.abs(xyzs - np.asarray(xyz)[None, :]).sum(axis=1)
    idx = int(np.argmin(dist))
    fname = os.path.join(base, "OPtabs", f"opacity{tabs[idx]:03d}.csv")
    raw = np.loadtxt(fname, delimiter=",", skiprows=1)
    logt = raw[:, 0]
    vals = raw[:, 1:]
    n_r = vals.shape[1]
    logr = -8.0 + 0.5 * np.arange(n_r)                # -8.0 ... 1.0
    g = (torch.as_tensor(logt, dtype=torch.float32),
         torch.as_tensor(logr, dtype=torch.float32))
    v = torch.as_tensor(vals, dtype=torch.float32)
    _OP_CACHE[key] = (g, v)
    return g, v


_AMW_CACHE = {}


def _mean_molecular_weight():
    """(Mean) molecular weight of the spot default (THEVENIN) mix.

    Follows pyPRT's ``amw = Sum(abu * wgt) / Sum(abu)`` with spot's
    own abundance table and atomic weights, so the gas-density
    conversion is consistent with the package's abundances.  This
    constant is a pure proportionality factor shared by both table
    recipes (it cancels in the ``kappa(lambda)/kappa(5000)`` ratio the
    RTE consumes, but matters for the absolute per-H-units).
    """
    if "amw" in _AMW_CACHE:
        return _AMW_CACHE["amw"]
    from .atoms import ATOM_WEIGHTS, AbundanceTable
    abu = AbundanceTable.from_default().relative       # N(X)/N(H), len 92
    wsum = float(np.sum(abu * ATOM_WEIGHTS))
    asum = float(np.sum(abu))
    amw = wsum / asum
    _AMW_CACHE["amw"] = amw
    return amw


def _mass_density_to_unit(pg, t, hprime, refidx=1.0):
    """(rg, NH) with rg [g/cm^3] and NH [cm^-3]; follows pyPRT exactly.

    pyPRT uses amw = Sum(abu*wgt)/Sum(abu) normalized by Sum(abu).
    Here the same definition is evaluated with spot's own THEVENIN
    abundance table and atomic weights (see :func:`_mean_molecular_weight`),
    so the conversion is self-consistent with the package abundances.
    """
    # p(H') / (kB T): number density of hydrogen nuclei
    nh = hprime / (_KB_CGS * t)
    # gas density: rg = Pg / (T * Rg/amw) with amw = mean molecular weight
    amw = _mean_molecular_weight()
    rg = pg / (t * (_RG_CGS / amw))
    return rg, nh


def _atlas_kappa(lambdas, temperature, pe, pp, vmic=None, refidx=1.0):
    """ATLAS Rosseland opacity per hydrogen nucleon (see module doc)."""
    dev, dtp = lambdas.device, lambdas.dtype
    g, f = _load_atlas_tables()
    g0 = tuple(x.to(device=dev, dtype=dtp) for x in g)
    f = f.to(device=dev, dtype=dtp)
    logt = torch.log10(temperature.to(dev).to(dtp))
    logp = torch.log10(pp["pg"].to(dev).to(dtp))
    if vmic is None:
        v = torch.zeros_like(logt)
    else:
        v = vmic.to(dev).to(dtp)
    val10 = _interp3(logt, logp, v.clamp(0.0, 8.0), *g0, f)
    rg, nh = _mass_density_to_unit(pp["pg"].to(dev).to(dtp),
                                   temperature.to(dev).to(dtp),
                                   pp["h_prime"].to(dev).to(dtp))
    kappa = torch.pow(10.0, val10) * rg / nh
    return kappa


def _op_kappa(lambdas, temperature, pe, pp, vmic=None, refidx=1.0):
    """Opacity Project table opacity per hydrogen nucleon."""
    dev, dtp = lambdas.device, lambdas.dtype
    g, f = _load_op_tables()
    g0 = tuple(x.to(device=dev, dtype=dtp) for x in g)
    f = f.to(device=dev, dtype=dtp)
    logt = torch.log10(temperature.to(dev).to(dtp))
    logr_grid, _ = g[0], g[1]
    rg, nh = _mass_density_to_unit(pp["pg"].to(dev).to(dtp),
                                   temperature.to(dev).to(dtp),
                                   pp["h_prime"].to(dev).to(dtp))
    logr = torch.log10(rg / torch.pow(temperature.to(dev).to(dtp) * 1e-6, 3.0))
    val10 = _interp2(logt, logr, g0[0], g0[1], f)
    kappa = torch.pow(10.0, val10) * rg / nh
    return kappa


def continuum_opacity_recipe(name, lambdas, temperature, pe, pp,
                             vmic=None, refractive_index=1.0):
    """
    Continuum opacity per hydrogen nucleon chosen by recipe name.

    Parameters
    ----------
    name : str
        ``'mihalas'`` (default, Mihalas-type formula recipe) | ``'atlas'`` |
        ``'opacity_project'`` | ``'none'`` (case-insensitive).
    lambdas : torch.Tensor (Nw,)
        Wavelengths [Angstrom] (ignored by the table recipes — the
        Rosseland tables are wavelength-averaged).
    temperature : torch.Tensor (Nb, Nt, 1)  [K]
    pe : torch.Tensor (Nb, Nt, 1)  [dyn/cm^2]
    pp : PartialPressure from :func:`ionization_equilibrium`.
    vmic : torch.Tensor (Nb, Nt, 1) or None
        Microturbulence [km/s] (ATLAS table's third grid axis; the
        Opacity Project table does not use it).
    refractive_index : float
        Unused by the table recipes (kept for API compatibility).

    Returns
    -------
    torch.Tensor (Nb, Nt, Nw)
        Continuum absorption per H nucleon [cm^2/H].
    """
    key = str(name).strip().lower()
    if key == "mihalas":
        return continuum_opacity(lambdas, temperature, pe, pp,
                                 refractive_index=refractive_index)
    if key == "atlas":
        return _atlas_kappa(lambdas, temperature, pe, pp, vmic=vmic)
    if key in ("opacity_project", "opacity project"):
        return _op_kappa(lambdas, temperature, pe, pp, vmic=vmic)
    raise ValueError(
        f"unknown continuum_opacity recipe {name!r}; available: "
        f"['mihalas', 'atlas', 'opacity_project']")
