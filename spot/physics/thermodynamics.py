# -*- coding: utf-8 -*-
"""
spot.physics.thermodynamics — Planck function, HSRA continuum and
air refractive index
=================================================================
The LTE Planck function, the HSRA reference continuum intensity used to
normalize emergent profiles, and the air refractive index used to
convert between vacuum and air wavelengths.
"""

import math

import torch

from .constants import HC_K, TWO_H_C2

__all__ = ["planck_intensity", "hsra_continuum", "air_refractive_index"]


def planck_intensity(temperature, wavelength_cm, bratio=1.0):
    """
    Planck function in LTE (per unit wavelength, CGS).

    Parameters
    ----------
    temperature : torch.Tensor
        Temperature [K], arbitrary shape.
    wavelength_cm : float
        Wavelength in cm (or broadcastable tensor).
    bratio : float or torch.Tensor
        Departure-coefficient ratio b_low/b_up (1.0 in LTE).

    Returns
    -------
    torch.Tensor
        B_lambda = 2hc^2/lambda^5 / (exp(hc/(lambda kT) / bratio) - 1)
        [erg/(s cm^3 sr)].
    """
    a = HC_K / wavelength_cm                      # hc/(lambda k)
    c = TWO_H_C2 / wavelength_cm ** 5
    alpha = a / temperature / bratio
    # Rayleigh-Jeans limit for very small alpha: c/alpha avoids the
    # catastrophic cancellation in exp(alpha) - 1
    small = alpha < 2e-4
    planck = torch.where(small, c / alpha, c / (torch.exp(alpha) - 1.0))
    return planck


# ---------------------------------------------------------------------------
# HSRA continuum intensity at a wavelength (piecewise polynomial fits)
# ---------------------------------------------------------------------------
_C1 = [-4.906054765549e13, 1.684734544039e11, 1.507254517567e7, -7561.242976546, 0.]
_C2 = [-4.4650822755e14, 6.1319780351059e11, -9.350928003805e7, 0., 0.]
_C3 = [-1.025961e15, 1.3172859e12, -3.873465e8, 46486.541, -2.049]
_C4 = [4.861821e15, -2.2589885e12, 4.3764376e8, -39279.61444, 1.34388]
_C5 = [1.758394e15, -3.293986e11, 1.6782617e7, 0., 0.]
_C6 = [1.61455557e16, -6.544209e12, 1.0159316e9, -70695.58136, 1.852022]
_C7 = [7.97805136e14, -1.16906597e11, 5.315222e6, -4.57327954, -3.473452e-3]


def hsra_continuum(x):
    """
    Continuum intensity of the HSRA model at wavelength ``x`` [Angstrom].

    Emergent profiles are normalized by this reference continuum value.

    Parameters
    ----------
    x : float
        Wavelength in Angstrom.

    Returns
    -------
    float
        HSRA continuum intensity (CGS units of the Planck function).
    """
    if isinstance(x, torch.Tensor):
        return _hsra_torch(x)
    c = None
    if x < 3644.15:
        c = _C1
    elif x < 3750.0:
        c = _C2
    elif x < 6250.0:
        c = _C3
    elif x < 8300.0:
        c = _C4
    elif x < 8850.0:
        c = _C5
    elif x < 10000.0:
        c = _C6
    elif x < 11000.0:
        c = _C7
    else:
        return _conhsra3(x)
    val = c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * c[4])))
    return float(val)


def _hsra_torch(x):
    """Vectorized torch version of hsra_continuum."""
    c1 = torch.tensor(_C1, device=x.device, dtype=x.dtype)
    c2 = torch.tensor(_C2, device=x.device, dtype=x.dtype)
    c3 = torch.tensor(_C3, device=x.device, dtype=x.dtype)
    c4 = torch.tensor(_C4, device=x.device, dtype=x.dtype)
    c5 = torch.tensor(_C5, device=x.device, dtype=x.dtype)
    c6 = torch.tensor(_C6, device=x.device, dtype=x.dtype)
    c7 = torch.tensor(_C7, device=x.device, dtype=x.dtype)

    def poly(c, v):
        return c[0] + v * (c[1] + v * (c[2] + v * (c[3] + v * c[4])))

    out = torch.where(x < 3644.15, poly(c1, x), torch.zeros_like(x))
    out = torch.where((x >= 3644.15) & (x < 3750.0), poly(c2, x), out)
    out = torch.where((x >= 3750.0) & (x < 6250.0), poly(c3, x), out)
    out = torch.where((x >= 6250.0) & (x < 8300.0), poly(c4, x), out)
    out = torch.where((x >= 8300.0) & (x < 8850.0), poly(c5, x), out)
    out = torch.where((x >= 8850.0) & (x < 10000.0), poly(c6, x), out)
    out = torch.where((x >= 10000.0) & (x < 11000.0), poly(c7, x), out)
    out = torch.where(x >= 11000.0,
                      torch.tensor([_conhsra3(float(v)) for v in x.cpu().tolist()],
                                   device=x.device, dtype=x.dtype), out)
    return out


def air_refractive_index(wavelength_um, temperature=15.0, pressure=760.0,
                         humidity=0.0):
    """
    Refractive index of air (Edlén-type formula).

    Parameters
    ----------
    wavelength_um : float
        Wavelength in microns.
    temperature : float
        Temperature [C] (15 C default).
    pressure : float
        Pressure [mmHg] (760 default).
    humidity : float
        Relative humidity [%] (0 default).

    Returns
    -------
    float
        n_air - 1 + 1 (the refractive index).
    """
    s = 1.0 / wavelength_um ** 2
    r = (6.4328e-5 + 2.94981e-2 / (146.0 - s)
         + 2.554e-4 / (41.0 - s))
    y = 1.0 / (273.15 + temperature)
    alpha = 3.67e-3 + 3.3e-4 * math.exp((0.19 - wavelength_um) / 0.12)

    if temperature <= 0.0:
        wvsp = 10.0 ** (77.4021323 + y * (-9.6982e4 + y * (5.50733046e7 + y * (
            -1.70804596e10 + y * (2.96751446e12 - y * (2.73866936e14
            - 1.04883576e16 * y))))))
    else:
        wvsp = 10.0 ** (-55.1132754 + y * (1.19551822e5 + y * (-9.70108858e7 + y * (
            4.12856913e10 + y * (-9.87931835e12 + y * (1.25868645e15
            - 6.66885272e16 * y))))))

    wvp = 1e-2 * humidity * wvsp
    n = (1.0 + 1.31579e-3 * r * pressure / (1.0 + alpha * (temperature - 15.0)
         / (1.0 + 15.0 * alpha)) - 5.49e-8 * wvp / (1.0 + alpha * temperature))
    return n


def air_refractive_index_torch(wavelength_um, temperature=15.0, pressure=760.0,
                               humidity=0.0):
    """
    Vectorized torch version of :func:`air_refractive_index`.

    Pure tensor arithmetic (no Python loop / ``.numpy()``), so it can be
    called inside torch.func vmap/jacfwd transforms (the synthesis builds
    the K matrix under jacfwd for the analytic response functions).
    """
    s = 1.0 / wavelength_um ** 2
    r = (6.4328e-5 + 2.94981e-2 / (146.0 - s)
         + 2.554e-4 / (41.0 - s))
    y = 1.0 / (273.15 + temperature)
    alpha = 3.67e-3 + 3.3e-4 * torch.exp((0.19 - wavelength_um) / 0.12)

    if temperature <= 0.0:
        wvsp = 10.0 ** (77.4021323 + y * (-9.6982e4 + y * (
            5.50733046e7 + y * (-1.70804596e10 + y * (
                2.96751446e12 - y * (2.73866936e14 - 1.04883576e16 * y))))))
    else:
        wvsp = 10.0 ** (-55.1132754 + y * (1.19551822e5 + y * (
            -9.70108858e7 + y * (4.12856913e10 + y * (
                -9.87931835e12 + y * (1.25868645e15
                - 6.66885272e16 * y))))))

    wvp = 1e-2 * humidity * wvsp
    n = (1.0 + 1.31579e-3 * r * pressure / (1.0 + alpha * (temperature - 15.0)
         / (1.0 + 15.0 * alpha)) - 5.49e-8 * wvp / (1.0 + alpha * temperature))
    return n


def _conhsra3(x):
    """HSRA continuum for lambda > 11000 Angstrom (piecewise polynomial table)."""
    par = _CONHSRA3_PAR
    i = 0
    while x > par[i][0] and i < len(par) - 1:
        i += 1
    if x <= par[i][0] and i > 0:
        i -= 1
    xx = x / par[i][1]
    yy = par[i][3] + xx * (par[i][4] + xx * (par[i][5] + xx * (par[i][6]
                                                               + xx * (par[i][7] + xx * par[i][8]))))
    yyoff = yy + par[i][2]
    yyoff = min(yyoff, 88.7)
    yyoff = max(yyoff, -87.3)
    return float(math.exp(yyoff))


# (x0, xnorm, yoff, p0..p5) — 38 entries, the piecewise polynomial table
_CONHSRA3_PAR = [
    (11000.0, 1000.0, 30.0, 3.4828959, 0.29332542, -0.079358369, 0.0050384849, -0.00010911067, 0.0),
    (14202.0, 14000.0, 31.633, 2.2519136, -2.21467, 0.0, 0.0, 0.0, 0.0),
    (14250.0, 1000.0, 30.0, 4.0522751, -0.18186949, 0.00083747599, 0.0, 0.0, 0.0),
    (14500.0, 14000.0, 31.580, 2.2966776, -2.2066212, 0.0, 0.0, 0.0, 0.0),
    (14577.0, 14000.0, 31.580, 2.2530627, -2.155536, 0.0, 0.0, 0.0, 0.0),
    (14600.0, 1000.0, 30.0, 16.038509, -2.6510218, 0.17030757, -0.0038723181, 0.0, 0.0),
    (16350.0, 16000.0, 31.280, -2.2817106, 7.4979477, -5.1365681, 0.0, 0.0, 0.0),
    (16419.0, 16000.0, 31.280, 5.0576959, -6.3530731, 1.3913925, 0.0, 0.0, 0.0),
    (16450.0, 1000.0, 30.0, 7.2813407, -0.58288516, 0.018436802, -0.00036871446, 3.22391e-6, 0.0),
    (22750.0, 22800.0, 30.080, 2.0074463, 2.2886963, -6.8588867, 2.5595589, 0.0, 0.0),
    (22776.3, 22800.0, 30.080, 3.7565713, -3.7561476, 0.0, 0.0, 0.0, 0.0),
    (22800.0, 1000.0, 30.0, 6.8152949, -0.49897589, 0.012840221, -0.00020295009, 1.37788e-6, 0.0),
    (30000.1, 30000.0, 29.035, 18.251404, -32.633972, 14.387756, 0.0, 0.0, 0.0),
    (30049.999, 10000.0, 30.0, 4.5429795, -2.3911509, 0.18562412, 0.0, 0.0, 0.0),
    (32750.1, 30000.0, 28.700, 1.9952565, 1.9831088, -5.4161187, 1.7658206, 0.0, 0.0),
    (32797.9, 30000.0, 28.700, 3.8456789, -3.5194291, 0.0, 0.0, 0.0, 0.0),
    (32799.999, 10000.0, 30.0, 5.8927794, -3.8081024, 0.71566516, -0.080796182, 0.0038640825, 0.0),
    (44600.102, 44600.0, 27.503, 3.9016303, -3.8992909, 0.0, 0.0, 0.0, 0.0),
    (44641.602, 44600.0, 27.503, 3.9005669, -3.8980503, 0.0, 0.0, 0.0, 0.0),
    (44650.002, 10000.0, 30.0, 4.9412975, -2.9472656, 0.42248154, -0.036235511, 0.0013139183, 0.0),
    (58300.0, 58300.0, 26.454, 3.9359391, -3.9338882, 0.0, 0.0, 0.0, 0.0),
    (58307.398, 58300.0, 26.454, 3.9334708, -3.9313729, 0.0, 0.0, 0.0, 0.0),
    (58349.998, 10000.0, 30.0, 2.217764, -1.3022186, 0.053848838, 0.0, 0.0, 0.0),
    (60950.0, 100000.0, 25.0, 8.9635353, -22.786591, 25.027283, -16.435974, 4.5663452, 0.0),
    (73700.0, 73800.0, 25.525, 3.959552, -3.9584524, 0.0, 0.0, 0.0, 0.0),
    (73795.398, 73800.0, 25.525, 3.9558706, -3.9547552, 0.0, 0.0, 0.0, 0.0),
    (73850.0, 100000.0, 25.0, 8.2165692, -18.74957, 16.827433, -9.0172343, 2.0436585, 0.0),
    (91100.0, 91100.0, 24.690, 3.9799142, -3.9787151, 0.0, 0.0, 0.0, 0.0),
    (91105.203, 91100.0, 24.690, 3.972252, -3.9710477, 0.0, 0.0, 0.0, 0.0),
    (91150.0, 100000.0, 25.0, 6.5153636, -11.660272, 5.7148422, -1.2492957, 0.0, 0.0),
    (110900.0, 100000.0, 20.0, 10.828452, -9.7908309, 4.0163633, -0.73414296, 0.0, 0.0),
    (131150.0, 131150.0, 23.239, 3.990582, -3.9896151, 0.0, 0.0, 0.0, 0.0),
    (131191.5, 131150.0, 23.239, 3.983481, -3.9824753, 0.0, 0.0, 0.0, 0.0),
    (131200.0, 100000.0, 20.0, 10.113794, -8.1683628, 2.7866455, -0.42295997, 0.0, 0.0),
    (160850.0, 100000.0, 20.0, 9.1879743, -6.4657758, 1.7401452, -0.20796399, 0.0, 0.0),
    (210800.0, 100000.0, 20.0, 8.2386927, -5.0976305, 1.0811192, -0.10187501, 0.0, 0.0),
    (260750.0, 100000.0, 20.0, 7.4709058, -4.206995, 0.73613949, -0.057257409, 0.0, 0.0),
    (310700.0, 100000.0, 20.0, 6.9713039, -3.9056023, 0.74610918, -0.093224197, 0.0064254836, -0.00018543588),
]
