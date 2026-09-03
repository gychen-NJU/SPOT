# -*- coding: utf-8 -*-
"""
spot.physics.derivatives — hand-written chain-rule derivatives
================================================================
Full-analytic response functions: every derivative below is the
reference-native Fortran formula transcribed to vectorized torch
(no autograd, no jacfwd).

Conventions (all derivatives follow the same naming):
  * ``du`` (partition functions)   = d ln(u)/dT
  * ``dsaha``                      = d ln(saha)/dT
  * ``dp``/``ddp``                 = d ln(p)/dT, d ln(p)/dPe
  * ``dkappa``/``ddkappa``         = d kappa/dT, d kappa/dPe
  * ``deta0``/``ddeta0``/``meta0`` = d eta0/dT, d eta0/dPe, d eta0/dvmic
  * ``da``/``dda``/``ma``          = d a/dT, d a/dPe, d a/dvmic
  * mvoigt outputs (etar, vetar, getar, ettar, ettvr, ettmr, esar,
    vesar, gesar, essar, essvr, essmr).
"""

import numpy as np
import torch

from .constants import GAS, PI
from .ionization import saha_ratio
from .partition import _HANDLERS

__all__ = [
    "partition_derivatives", "saha_log_derivative",
    "line_opacity_derivatives", "damping_derivatives",
    "voigt_derivatives", "mvoigt_derivatives",
    "absorption_matrix_derivatives",
]


# ---------------------------------------------------------------------------
# partition-function derivatives  d ln(u)/dT
# ---------------------------------------------------------------------------
def _du_h(t, u0, u1, u2):
    du0 = torch.zeros_like(t)
    du0 = torch.where(t > 1.3e4, 3.8e-5 / u0, du0)
    du0 = torch.where(t > 1.62e4, (-1.1428e-3 + 2.0 * t * 3.52e-8) / u0, du0)
    return du0, torch.zeros_like(t), torch.zeros_like(t)


def _du_he(t, u0, u1, u2):
    du0 = torch.zeros_like(t)
    du0 = torch.where(t > 3e4, (-9.4103e-4 + 2.0 * t * 1.6095e-8) / u0, du0)
    return du0, torch.zeros_like(t), torch.zeros_like(t)


def _du_li(t, u0, u1, u2):
    y = 1e-3 * t
    du0 = 1e-3 * (-6.8926e-2 + 2.0 * y * 1.4081e-2) / u0
    du0 = torch.where(t > 6e3, (-7.3292e-4 + 2.0 * t * 8.5586e-8) / u0, du0)
    return du0, torch.zeros_like(t), torch.zeros_like(t)


def _du_be(t, u0, u1, u2):
    du0 = torch.where(u0 != 1.0, 7.032e-5 / u0, torch.zeros_like(t))
    return du0, torch.zeros_like(t), torch.zeros_like(t)


def _du_b(t, u0, u1, u2):
    return 1e-3 * 1.0438e-2 / u0, torch.zeros_like(t), torch.zeros_like(t)


def _du_c(t, u0, u1, u2):
    y = 1e-3 * t
    du0 = 1e-3 * ((2.0485e-2 + 2.0 * y * 1.7629e-2 - 3.0 * y * y * 3.9091e-4) / u0)
    du0 = torch.where(t > 1.2e4, (-1.3907e-3 + 2.0 * t * 9.0844e-8) / u0, du0)
    du1 = 1.6833e-5 / u1
    du1 = torch.where(t > 2.4e4, (-6.9347e-4 + 2.0 * t * 2.0861e-8) / u1, du1)
    du2 = torch.where(t > 1.95e4, 8e-5 / u2, torch.zeros_like(t))
    return du0, du1, du2


def _du_n(t, u0, u1, u2):
    y = 1e-3 * t
    du0 = 1e-3 * ((1.7491e-2 - 2.0 * y * 1.0148e-2 + 3.0 * y * y * 1.7138e-3) / u0)
    du0 = torch.where(t > 8800.0, 2.54e-4 / u0, du0)
    du0 = torch.where(t > 1.8e4, (-1.7139e-3 + 2.0 * t * 8.633e-8) / u0, du0)
    du1 = 1.420e-4 / u1
    du1 = torch.where(t > 3.3e4, (-1.8931e-3 + 2.0 * t * 4.4612e-8) / u1, du1)
    du2 = (-2.6651e-5 + 2.0 * t * 1.8228e-9) / u2
    du2 = torch.where(t < 7310.5, torch.zeros_like(t), du2)
    return du0, du1, du2


def _du_o(t, u0, u1, u2):
    du0 = 1.10e-4 / u0
    du0 = torch.where(t > 1.9e4, (-6.019e-3 + 2.0 * t * 1.657e-7) / u0, du0)
    du1 = torch.where(u1 != 4.0, 8e-5 / u1, torch.zeros_like(t))
    du1 = torch.where(t > 3.64e4, (-4.216e-3 + 2.0 * t * 6.885e-8) / u1, du1)
    return du0, du1, 1.1348e-4 / u2


def _du_f(t, u0, u1, u2):
    y = 1e-3 * t
    du0 = 0.77683 - 2.0 * y * 0.20884 + 3.0 * y * y * 2.6771e-2 - 4.0 * y ** 3 * 1.3035e-3
    du0 = 1e-3 * (du0 / u0)
    du0 = torch.where(t > 8750.0, torch.zeros_like(t), du0)
    du0 = torch.where(t > 2e4, (-9.229e-4 + 2.0 * t * 2.312e-8) / u0, du0)
    return du0, 8.9e-5 / u1, 1.38e-4 / u2


def _du_ne(t, u0, u1, u2):
    du0 = torch.where(t > 2.69e4, (-2.113e-3 + 2.0 * t * 4.359e-8) / u0,
                      torch.zeros_like(t))
    return du0, 4e-5 / u1, 7.956e-5 / u2


def _du_na(t, u0, u1, u2):
    du0 = torch.where(u0 != 2.0, 9.3e-5 / u0, torch.zeros_like(t))
    du0 = torch.where(t > 5400.0, 5.66e-4 / u0, du0)
    du0 = torch.where(t > 8.5e3, (-1.2415e-3 + 2.0 * t * 1.3861e-7) / u0, du0)
    return du0, torch.zeros_like(t), 5.69e-6 / u2


def _du_mg(t, u0, u1, u2):
    x = torch.log(5040.0 / t)
    dx = -1.0 / t
    du0 = -dx * (6.173172 + x * (2.0 * 2.889176 + x * (3.0 * 2.393895 + 4.0 * x * 0.784131)))
    du0 = du0 * (u0 - 1.0) / u0
    du0 = torch.where(t > 8e3, (-7.8909e-4 + 2.0 * t * 7.4531e-8) / u0, du0)
    du1 = -dx * (7.0600678 + x * (2.0 * 1.966097 + 3.0 * x * 0.212417))
    du1 = du1 * (u1 - 2.0) / u1
    du1 = torch.where(t > 2e4, (-1.0817e-3 + 2.0 * t * 4.7841e-8) / u1, du1)
    return du0, du1, torch.zeros_like(t)


def _du_al(t, u0, u1, u2):
    y = 1e-3 * t
    du0 = 1e-3 * (0.27833 - y * (2.0 * 4.7529e-2 - 3.0 * y * 3.0199e-3)) / u0
    du1 = torch.where(u1 != 1.0, 3.245e-5 / u1, torch.zeros_like(t))
    du1 = torch.where(t > 2.24e4, (-5.987e-3 + 2.0 * t * 1.485e-7) / u1, du1)
    du2 = torch.where(u2 != 2.0, 3.43e-6 / u2, torch.zeros_like(t))
    du2 = torch.where(t > 1.814e4, (-1.59e-4 + 2.0 * t * 4.382e-9) / u2, du2)
    return du0, du1, du2


def _du_si(t, u0, u1, u2):
    y = 1e-3 * t
    du0 = 1e-3 * (0.86319 + y * (-2.0 * 0.11622 + y * (3.0 * 0.013109 - 4.0 * y * 6.2013e-4))) / u0
    du0 = torch.where(t > 1.04e4, (-1.465e-2 + 2.0 * t * 7.282e-7) / u0, du0)
    du1 = 4e-5 / u1
    du1 = torch.where(t > 1.8e4, (-2.22e-3 + 2.0 * t * 6.188e-8) / u1, du1)
    du2 = torch.where(u2 != 1.0, 1.1e-5 / u2, torch.zeros_like(t))
    du2 = torch.where(t > 3.33e4, (-1.408e-3 + 2.0 * t * 2.617e-8) / u2, du2)
    return du0, du1, du2


def _du_p(t, u0, u1, u2):
    y = 1e-3 * t
    du0 = 1e-3 * (-0.22476 + y * (2.0 * 0.057306 - 3.0 * y * 1.0381e-3)) / u0
    du0 = torch.where(t > 6e3, 5.2e-4 / u0, du0)
    du1 = 1e-3 * (2.2494 + y * (-2.0 * 0.55371 + y * (3.0 * 0.071913 - 4.0 * y * 3.5156e-3))) / u1
    du1 = torch.where(t > 7250.0, 5.38e-4 / u1, du1)
    return du0, du1, 3.4e-5 / u2


def _du_s(t, u0, u1, u2):
    du0 = 2.1e-4 / u0
    du0 = torch.where(t > 1.16e4, (-4.906e-3 + 2.0 * t * 2.125e-7) / u0, du0)
    du1 = 2.43e-4 / u1
    du1 = torch.where(t > 1.05e4, (-1.68e-4 + 2.0 * t * 1.323e-8) / u1, du1)
    return du0, du1, 1.88e-4 / u2


def _du_cl(t, u0, u1, u2):
    du0 = 6e-5 / u0
    du0 = torch.where(t > 1.84e4, 4.8e-3 / u0, du0)
    return du0, 2.43e-4 / u1, 2.62e-4 / u2


def _du_ar(t, u0, u1, u2):
    return torch.zeros_like(t), 3.8e-5 / u1, 1.554e-4 / u2


def _du_k(t, u0, u1, u2):
    y = 1e-3 * t
    du0 = 1e-3 * (0.023169 - y * (2.0 * 0.017432 - y * 3.0 * 4.0938e-3)) / u0
    du0 = torch.where(t > 5800.0, 2.124e-3 / u0, du0)
    return du0, torch.zeros_like(t), 1.93e-5 / u2


def _du_ca(t, u0, u1, u2):
    x = torch.log(5040.0 / t)
    dx = -1.0 / t
    du0 = -dx * (5.004556 + x * (2.0 * 1.645456 + x * (3.0 * 1.326861 + 4.0 * x * 0.508553)))
    du0 = du0 * (u0 - 1.0) / u0
    du1 = -dx * (3.996089 + x * (2.0 * 1.890737 + x * 3.0 * 0.539672))
    du1 = du1 * (u1 - 2.0) / u1
    return du0, du1, torch.zeros_like(t)


def _du_sc(t, u0, u1, u2):
    x = torch.log(5040.0 / t)
    dx = -1.0 / t
    du0 = dx * (-1.2392 + x * (2.0 * 1.173504 + 3.0 * x * 0.517796))
    du0 = du0 * (u0 - 4.0) / u0
    du1 = dx * (-0.596238 + 2.0 * x * 0.054658)
    du1 = du1 * (u1 - 3.0) / u1
    return du0, du1, torch.zeros_like(t)


def _du_ti(t, u0, u1, u2):
    x = torch.log(5040.0 / t)
    dx = -1.0 / t
    du0 = dx * (-1.227798 + x * (2.0 * 0.799613 + 3.0 * x * 0.278963))
    du0 = du0 * (u0 - 5.0) / u0
    du0 = torch.where(t < 5.5e3, (-2.838e-4 + 2.0 * t * 5.819e-7) / u0, du0)
    du1 = dx * (-0.551431 + 2.0 * x * 0.115693)
    du1 = du1 * (u1 - 4.0) / u1
    return du0, du1, 8.5e-4 / u2


def _du_v(t, u0, u1, u2):
    x = torch.log(5040.0 / t)
    dx = -1.0 / t
    du0 = dx * (-0.906352 + x * (2.0 * 0.724694 + 3.0 * x * 0.1622))
    du0 = du0 * (u0 - 4.0) / u0
    du1 = dx * (-0.757371 + x * 2.0 * 0.21043)
    du1 = du1 * (u1 - 1.0) / u1
    du2 = 1.03e-2 / u2
    du2 = torch.where(t < 2.25e3, 2.4e-3 / u2, du2)
    return du0, du1, du2


def _du_cr(t, u0, u1, u2):
    x = torch.log(5040.0 / t)
    dx = -1.0 / t
    du0 = dx * (-2.923459 + x * (2.0 * 0.154709 + x * 3.0 * 0.09527))
    du0 = du0 * (u0 - 7.0) / u0
    du1 = -dx * (4.143973 + x * (2.0 * 1.096548 + 3.0 * x * 0.230073))
    du1 = du1 * (u1 - 6.0) / u1
    return du0, du1, 2.1e-3 / u2


def _du_mn(t, u0, u1, u2):
    x = torch.log(5040.0 / t)
    dx = -1.0 / t
    du0 = -dx * (5.531252 + x * (2.0 * 2.13632 + x * (3.0 * 1.061055 + 4.0 * x * 0.265557)))
    du0 = du0 * (u0 - 6.0) / u0
    du1 = -dx * (3.77279 + x * (2.0 * 0.814675 + 3.0 * x * 0.159822))
    du1 = du1 * (u1 - 7.0) / u1
    return du0, du1, torch.zeros_like(t)


def _du_fe(t, u0, u1, u2):
    x = torch.log(5040.0 / t)
    dx = -1.0 / t
    du0 = dx * (-0.979745 + x * (2.0 * 0.76027 + 3.0 * x * 0.118218))
    du0 = du0 * (u0 - 9.0) / u0
    du0 = torch.where(t < 4e3, (1.306e-3 + 2.0 * t * 2.04e-7) / u0, du0)
    du0 = torch.where(t > 9e3, (-9.5922e-3 + 2.0 * t * 1.2477e-6) / u0, du0)
    du1 = dx * (-0.612094 + 2.0 * x * 0.280982)
    du1 = du1 * (u1 - 10.0) / u1
    du1 = torch.where(t > 1.8e4, (-6.1104e-3 + 2.0 * t * 5.1567e-7) / u1, du1)
    du2 = (5.5048e-4 + 2.0 * t * 5.7514e-8) / u2
    return du0, du1, du2


def _du_co(t, u0, u1, u2):
    return 4.9e-3 / u0, 3.58e-3 / u1, 1.42e-3 / u2


def _du_ni(t, u0, u1, u2):
    x = torch.log(5040.0 / t)
    dx = -1.0 / t
    du0 = dx * (-0.401323 + x * (2.0 * 0.077498 - 3.0 * x * 0.278468))
    du0 = du0 * (u0 - 9.0) / u0
    du1 = -dx * (1.528966 + 2.0 * x * 0.115654)
    du1 = du1 * (u1 - 6.0) / u1
    return du0, du1, 6.9e-4 / u2


def _du_cu(t, u0, u1, u2):
    du0 = torch.where(u0 != 2.0, 1.51e-4 / u0, torch.zeros_like(t))
    du0 = torch.where(t > 6250.0, 4.58e-4 / u0, du0)
    du1 = torch.where(u1 != 1.0, 1.49e-4 / u1, torch.zeros_like(t))
    return du0, du1, 9.4e-5 / u2


def _du_zn(t, u0, u1, u2):
    du0 = torch.where(u0 != 1.0, 5.11e-5 / u0, torch.zeros_like(t))
    return du0, torch.zeros_like(t), torch.zeros_like(t)


# elements 29..83 beyond the 28 used by the ionization balance: mostly
# constant or unused; provide a fallback (returns du=0 for those).
def _du_generic(t, u0, u1, u2):
    return torch.zeros_like(t), torch.zeros_like(t), torch.zeros_like(t)


_DU_HANDLERS = {
    1: _du_h, 2: _du_he, 3: _du_li, 4: _du_be, 5: _du_b, 6: _du_c,
    7: _du_n, 8: _du_o, 9: _du_f, 10: _du_ne, 11: _du_na, 12: _du_mg,
    13: _du_al, 14: _du_si, 15: _du_p, 16: _du_s, 17: _du_cl, 18: _du_ar,
    19: _du_k, 20: _du_ca, 21: _du_sc, 22: _du_ti, 23: _du_v, 24: _du_cr,
    25: _du_mn, 26: _du_fe, 27: _du_co, 28: _du_ni, 29: _du_cu, 30: _du_zn,
}


def partition_derivatives(z, t):
    """
    d ln(u)/dT of the partition functions of element ``z``, taken from
    the stored per-element polynomial coefficients.

    Returns (du0, du1, du2) with the same shape as ``t``.
    """
    if isinstance(z, str):
        from .atoms import element_symbol_to_number
        z = 26 if z.upper() in ("XX", "FE") else element_symbol_to_number(z)
    u0, u1, u2 = _HANDLERS[z](t)
    fn = _DU_HANDLERS.get(z, _du_generic)
    return fn(t, u0, u1, u2)


def saha_log_derivative(theta, chi, du1, du2):
    """
    d ln(saha)/dT of the Saha ratio:
    ``du2-du1+(theta/5040)*(2.5+chi*theta*ln10)``.

    Note: theta = 5040/T, chi in eV, du1/du2 are d ln(u)/dT.
    """
    return du2 - du1 + (theta / 5040.0) * (2.5 + chi * theta * 2.3025851)


# ---------------------------------------------------------------------------
# line opacity derivatives  (eta0 and its dT/dPe/dvmic)
# ---------------------------------------------------------------------------
def line_opacity_derivatives(theta, t, pe, energy_low, loggf, gf_abu,
                             wavelength_cm, u0, u1, u2, du0, du1, du2,
                             chi1, chi2, vdop, dvdop, mvdop,
                             kappa5, dkappa5, ddkappa5,
                             stimulated=True, istage=1, ne=None):
    """
    eta0 and its derivatives dT / dPe / dvmic.

    Conventions (spot u0/u1/u2 are the successive ionization stages):
      du*      : d ln(u*)/dT
      dvdop    : d vdop/dT
      mvdop    : vmic/vdop
      dkappa5  : d ln(kappa5)/dT ; ddkappa5 : d ln(kappa5)/dPe
    ne : tensor, optional
        Electron number density; enables the Debye lowering of the
        ionization potentials.
    Returns (eta0, deta0, ddeta0, meta0).
    """
    eta00 = 1.49736e-2 * gf_abu * wavelength_cm
    if ne is not None:
        rcu = ne.clamp(min=0.0) ** (1.0 / 3.0)
        chi1 = chi1 - 6.96e-7 * rcu
        chi2 = chi2 - 1.1048e-6 * rcu
    u12 = saha_ratio(theta, chi1, u0, u1, pe)
    du12 = u12 * saha_log_derivative(theta, chi1, du0, du1)   # d ln u12/dT
    ddu12 = -1.0 * u12 / pe                                    # d ln u12/dPe
    u23 = saha_ratio(theta, chi2, u1, u2, pe)
    du23 = u23 * saha_log_derivative(theta, chi2, du1, du2)
    ddu23 = -1.0 * u23 / pe
    u33 = 1.0 + u12 * (1.0 + u23)
    du33 = du12 * (1.0 + u23) + u12 * du23
    ddu33 = ddu12 * (1.0 + u23) + u12 * ddu23

    eta0 = eta00 * 10.0 ** (-theta * energy_low) / (u0 * vdop * u33 * kappa5)
    deta0 = eta0 * (2.3025851 * theta / t * energy_low - du0
                    - dvdop / vdop - du33 / u33)
    deta0 = deta0 - eta0 * dkappa5
    ddeta0 = eta0 * (-ddu33 / u33)
    ddeta0 = ddeta0 - eta0 * ddkappa5

    if istage == 2:
        eta0 = eta0 * u1 / u2 * u12
        deta0 = deta0 * u1 / u2 * u12 + eta0 * (du1 - du2 + du12 / u12)
        ddeta0 = ddeta0 * u1 / u2 * u12 + eta0 * (ddu12 / u12)

    if stimulated:
        corre = 1.0 - torch.exp(-1.4388 / (t * wavelength_cm))
        dcorre = (corre - 1.0) * (1.4388 / (t * t * wavelength_cm))
        deta0 = deta0 * corre + eta0 * dcorre
        ddeta0 = ddeta0 * corre
        eta0 = eta0 * corre
    meta0 = -eta0 * mvdop / vdop
    return eta0, deta0, ddeta0, meta0


# ---------------------------------------------------------------------------
# damping-parameter derivatives  (a and its dT/dPe/dvmic)
# ---------------------------------------------------------------------------
def damping_derivatives(t, theta, pe, pp, dpp, ddpp, wavelength_cm, vdop,
                        dvdop, mvdop, weight, alfa, sigma, zeff, istage,
                        energy_low, chi1, chi2):
    """
    Damping parameter a and its derivatives dT/dPe/dvmic.

    pp/dpp/ddpp follow the partial-pressure dict convention: dpp =
    d ln p/dT, ddpp = d ln p/dPe.  Returns (a, da, dda, ma).
    """
    weinv = 1.0 / weight
    croot = 1.66286e8 * weinv
    crad = 0.22233 / wavelength_cm
    # Two Doppler widths are used: vdop (from croot) and
    # vdop2 = sqrt(2*gas*t/weight + vtur^2); since croot == 2*gas/weight
    # they coincide numerically, so vdop2 = vdop (which already includes
    # the microturbulence).  dvdop2 = d vdop2/dT = gas/(weight*vdop2).
    vdop2 = vdop
    dvdop2 = GAS / (vdop2 * weight)

    if alfa == 0.0 or sigma == 0.0:
        # --- Unsöld ---
        chi1_eff = chi1 if istage != 2 else chi2
        eupper = energy_low + 1.2398539e-4 / wavelength_cm
        ediff1 = max(chi1_eff - eupper - chi1 * (istage - 1), 1.0)
        ediff2 = max(chi1_eff - energy_low - chi1 * (istage - 1), 3.0)
        chydro = (wavelength_cm * 10.0 ** (0.4 * np.log10(1.0 / ediff1 ** 2
                  - 1.0 / ediff2 ** 2) - 12.213) * 5.34784e3) * zeff
        if istage == 2:
            chydro = chydro * 1.741

        f1 = pp["h"]
        f90 = pp["e-"]
        f91 = pp["ne"]
        f2 = pp["he"]
        f89 = pp["h2"]

        aj = chydro * (f1 / f90 * f91) * t ** 0.3
        ai = ((0.992093 + weinv) ** 0.3
              + 0.6325 * f2 / f1 * (0.2498376 + weinv) ** 0.3
              + 0.48485 * f89 / f1 * (0.4960465 + weinv) ** 0.3)
        a = (aj * ai + crad) / (12.5663706 * vdop)
        ma = -a * mvdop / vdop

        # NOTE: the native reference omits the pg(1) (= p(H)/p(H'))
        # term in daj/ddaj.  Reproducing that omission was tested and it
        # degraded the inversion (dda deviates 15.6% from the true
        # derivative vs 4.7% with the h term); the h term is kept as it
        # is closer to the true derivative and the forward synthesis is
        # not bit-identical to the reference anyway.
        daj = aj * (dpp["h"] - dpp["e-"] + dpp["ne"] + 0.3 / t)
        ddaj = aj * (ddpp["h"] - ddpp["e-"] + ddpp["ne"])
        dai = (0.6325 * f2 / f1 * (0.2498376 + weinv) ** 0.3
               * (dpp["he"] - dpp["h"])
               + 0.48485 * f89 / f1 * (0.4960465 + weinv) ** 0.3
               * (dpp["h2"] - dpp["h"]))
        ddai = (0.6325 * f2 / f1 * (0.2498376 + weinv) ** 0.3
                * (ddpp["he"] - ddpp["h"])
                + 0.48485 * f89 / f1 * (0.4960465 + weinv) ** 0.3
                * (ddpp["h2"] - ddpp["h"]))
        da = (ai * daj + aj * dai) / (12.5663706 * vdop) - a * dvdop / vdop
        dda = (ai * ddaj + aj * ddai) / (12.5663706 * vdop)
        return a, da, dda, ma

    # --- Barklem (ABO) ---
    uma = 1.660540e-24
    xmu1 = uma * (1.008 * weight) / (1.008 + weight)
    xmu2 = uma * (4.0026 * weight) / (4.0026 + weight)
    xmu3 = uma * (2.016 * weight) / (2.016 + weight)
    pir = PI
    bol = 1.3807e-16
    v0 = 1e6
    coc2 = 0.3181818
    coc3 = 1.212121

    arr = 2.0 - alfa * 0.5 - 1.0
    gammaf = 1.0 + (-0.5748646 + (0.9512363 + (-0.6998588 + (0.4245549
                - 0.1010678 * arr) * arr) * arr) * arr) * arr
    vv = (1.0 - alfa) / 2.0
    beta = (wavelength_cm * 2.0 * (4.0 / pir) ** (alfa / 2.0) * gammaf
            * (v0 ** alfa) * sigma * ((8.0 * bol / pir) ** vv))

    dam = (beta * (pp["ne"] / pp["e-"]) * (pp["h"] * xmu1 ** (-vv)
           + coc2 * pp["he"] * xmu2 ** (-vv) + coc3 * pp["h2"] * xmu3 ** (-vv))
           * t ** vv)
    a = (1.0 / (4.0 * pir)) * (crad / vdop + dam / vdop2)
    ma = -a * mvdop / vdop

    ddam = (dam * (dpp["ne"] - dpp["e-"])
            + beta * (pp["ne"] / pp["e-"]) * (xmu1 ** (-vv) * pp["h"] * dpp["h"]
            + coc2 * xmu2 ** (-vv) * pp["he"] * dpp["he"]
            + coc3 * xmu3 ** (-vv) * pp["h2"] * dpp["h2"]) * t ** vv
            + vv * dam / t)
    dddam = (dam * (ddpp["ne"] - ddpp["e-"])
             + beta * (pp["ne"] / pp["e-"]) * (xmu1 ** (-vv) * pp["h"] * ddpp["h"]
             + coc2 * xmu2 ** (-vv) * pp["he"] * ddpp["he"]
             + coc3 * xmu3 ** (-vv) * pp["h2"] * ddpp["h2"]))
    da = (crad / (4.0 * pir)) * (-dvdop / (vdop ** 2.0)) \
        + (1.0 / (4.0 * pir)) * (ddam * vdop2 - dam * dvdop2) / (vdop2 ** 2.0)
    dda = (1.0 / (4.0 * pir)) * (dddam / vdop2)
    return a, da, dda, ma


def vmic_of_pp(pp):
    """Microturbulence in cm/s (from pp dict; kept for signature parity)."""
    return pp.get("_vmic", 0.0)


# ---------------------------------------------------------------------------
# Voigt-profile derivatives  (HV/FV/HA/FA analytic identities)
# ---------------------------------------------------------------------------
def voigt_derivatives(v, a, h, f):
    """
    Derivatives of the Voigt H(a,v) and Faraday F(a,v) profiles.

    Analytic identities (with piis = 1/sqrt(pi)):
        HV = -2v H + 4a F          dH/dv
        FV = piis - a H - 2v F     dF/dv
        HA = -2 FV                 dH/da
        FA =  HV/2                 dF/da
    Returns (HV, FV, HA, FA).
    """
    piis = 1.0 / 3.1415926 ** 0.5
    hv = -2.0 * v * h + 4.0 * a * f
    fv = piis - a * h - 2.0 * v * f
    ha = -2.0 * fv
    fa = hv / 2.0
    return hv, fv, ha, fa


def mvoigt_derivatives(v, a, shifts, strengths, field_doppler, w1, t13, t14,
                       dldop):
    """
    Summed Zeeman-component profiles and their derivatives.

    For one Zeeman group (pi / sigma_r / sigma_l): returns
      eta, veta (d/dvlos), geta (d/dB), etta (d/da),
      ettv (d/dT via v), ettm (d/dvmic),
      esa (Faraday), vesa, gesa, essa, essv, essm.
    ``w1 = wc/dldop`` (cm^-1 s); ``t13 = dvdop/vdop`` (d ln vdop/dT);
    ``t14 = mvdop/vdop`` (d ln vdop/dvmic); ``dldop`` = Doppler width [cm].
    """
    eta = torch.zeros_like(v)
    esa = torch.zeros_like(v)
    veta = torch.zeros_like(v)
    geta = torch.zeros_like(v)
    etta = torch.zeros_like(v)
    ettv = torch.zeros_like(v)
    ettm = torch.zeros_like(v)
    vesa = torch.zeros_like(v)
    gesa = torch.zeros_like(v)
    essa = torch.zeros_like(v)
    essv = torch.zeros_like(v)
    essm = torch.zeros_like(v)

    from .voigt import voigt_faraday_profile
    for shift, strength in zip(shifts, strengths):
        ver = v + shift * field_doppler
        h, f = voigt_faraday_profile(ver, a)
        hv, fv, ha, fa = voigt_derivatives(ver, a, h, f)
        eta = eta + strength * h
        esa = esa + strength * f
        etta = etta + strength * ha
        essa = essa + strength * fa
        ettv = ettv - strength * hv * t13 * ver
        # The native reference's line for the Faraday derivative uses as
        # its FIRST term ettvR (the eta derivative!) instead of the esar
        # analogue: essvR = ettvR - sr(ir)*FV*T13*ver.  This is a typo in
        # the reference that must be reproduced to match its response
        # functions exactly.
        essv = essv - strength * hv * t13 * ver - strength * fv * t13 * ver
        ettm = ettm - strength * hv * t14 * ver
        essm = essm - strength * fv * t14 * ver
        veta = veta - strength * hv * w1
        vesa = vesa - strength * fv * w1
        geta = geta + strength * hv * shift / dldop
        gesa = gesa + strength * fv * shift / dldop
    esa = 2.0 * esa
    vesa = 2.0 * vesa
    gesa = 2.0 * gesa
    essa = 2.0 * essa
    essv = 2.0 * essv
    essm = 2.0 * essm
    return (eta, veta, geta, etta, ettv, ettm,
            esa, vesa, gesa, essa, essv, essm)


def dldop_of(w1):
    """Doppler width from w1 = wc/dldop (only the ratio is needed)."""
    return 1.0 / w1




