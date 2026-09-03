# -*- coding: utf-8 -*-
"""
spot.physics.pressure — ionization equilibrium partial pressures
=================================================================
Given temperature and electron pressure, solve the statistical-
equilibrium (Saha + molecular) equations for the partial pressures of
all species, normalized to the hydrogen nucleon pressure p(H').

The solution is returned as a dict of tensors whose keys follow the
partial-pressure ``pg()`` convention (see the module docstring of
:mod:`spot.physics` for the key list).
"""

import torch

from ..physics.atoms import (IONIZATION_ENERGY_1, IONIZATION_ENERGY_2,
                             AbundanceTable)
from .ionization import (molecular_balance, molecular_balance_derivatives,
                         saha_ratio)
from .partition import partition_functions

__all__ = ["ionization_equilibrium", "ionization_equilibrium_cached",
           "ionization_equilibrium_cache", "PartialPressure"]

# Number of elements considered in the ionization balance (28)
N_ELEMENTS = 28


def ionization_equilibrium_cache(theta, abundance=None):
    """
    Cache the T-dependent pieces of :func:`ionization_equilibrium`.

    Returns ``(u0, u1, u2, cmol)``: partition functions of the first
    ``N_ELEMENTS`` elements at ``t = 5040/theta`` (each ``(28, *shape)``)
    and the molecular coefficients ``cmol`` ``(2, *shape)``.
    The partial pressures still depend on pe only through the Saha
    ratios, so at fixed ``theta`` the equilibrium can be re-evaluated
    for many pe values without recomputing the partition functions
    (the hot loop of :func:`spot.physics.hydrostatic.pe_from_pg`).
    """
    if abundance is None:
        abundance = AbundanceTable.from_default()

    device, dtype = theta.device, theta.dtype
    t = 5040.0 / theta

    cmol = molecular_balance(theta)                     # (2, *shape)
    cmol = torch.clamp(cmol, -30.0, 30.0)

    u_list = [partition_functions(iz, t) for iz in range(1, N_ELEMENTS + 1)]
    u0 = torch.stack([u[0] for u in u_list])   # (N_ELEMENTS, *shape)
    u1 = torch.stack([u[1] for u in u_list])
    u2 = torch.stack([u[2] for u in u_list])
    return u0, u1, u2, cmol


def ionization_equilibrium_cached(theta, pe, u0, u1, u2, cmol,
                                  abundance=None):
    """
    :func:`ionization_equilibrium` with precomputed T-dependent data.

    ``u0/u1/u2/cmol`` come from :func:`ionization_equilibrium_cache`.
    """
    if abundance is None:
        abundance = AbundanceTable.from_default()

    device, dtype = theta.device, theta.dtype
    t = 5040.0 / theta
    shape = theta.shape

    neg_pe = pe <= 0
    pe_safe = torch.where(neg_pe, torch.full_like(pe, 1e-10), pe)

    g4 = pe_safe * torch.pow(10.0, cmol[0])             # H2+ constant
    g5 = pe_safe * torch.pow(10.0, cmol[1])             # H2  constant
    g4 = torch.where(neg_pe, torch.zeros_like(g4), g4)
    g5 = torch.where(neg_pe, torch.zeros_like(g5), g5)

    # atomic data of the first N_ELEMENTS elements
    chi1 = torch.tensor(IONIZATION_ENERGY_1[:N_ELEMENTS],
                        device=device, dtype=dtype)
    chi2 = torch.tensor(IONIZATION_ENERGY_2[:N_ELEMENTS],
                        device=device, dtype=dtype)
    rel_abu = torch.tensor([abundance.relative[iz - 1] for iz in range(1, N_ELEMENTS + 1)],
                           device=device, dtype=dtype)

    # H ionization
    g2 = saha_ratio(theta, chi1[0], u0[0], u1[0], pe_safe)   # p(H+)/p(H)
    g3 = saha_ratio(theta, 0.754, torch.ones_like(u0[0]), u0[0], pe_safe)
    g3 = torch.clamp(g3, 1e-30, 1e30)
    g3 = 1.0 / g3                                            # p(H-)/p(H)

    # metals contribution to the electron budget: g1 = sum_i p_i a_i (1+2 b_i)
    # vectorized over all elements at once (single kernel per operation
    # instead of a per-element Python loop).  Extra leading dims: element
    # (N_ELEMENTS-1) x depth-batch (..., 1).
    theta4 = theta.unsqueeze(0)                        # (1, ..., 1)
    pe4 = pe_safe.unsqueeze(0)
    chi1m = chi1[1:].reshape(-1, 1, 1, 1)              # (N_ELEMENTS-1, 1, 1, 1)
    chi2m = chi2[1:].reshape(-1, 1, 1, 1)
    abum = rel_abu[1:].reshape(-1, 1, 1, 1)
    u0m = u0[1:]                                       # (N_ELEMENTS-1, *shape)
    u1m = u1[1:]
    u2m = u2[1:]
    a = saha_ratio(theta4, chi1m, u0m, u1m, pe4)
    b = saha_ratio(theta4, chi2m, u1m, u2m, pe4)
    c = torch.clamp(1.0 + a * (1.0 + b), 1e-20, 1e20)
    p_i = abum / c                                     # p(el)/p(H')
    ss1 = torch.clamp(1.0 + 2.0 * b, 1e-20, 1e20)
    ss = p_i * a * ss1
    g1 = ss.sum(dim=0)                                 # (..., 1)
    izm = torch.arange(1, N_ELEMENTS, device=device)   # 1..N_ELEMENTS-1
    pp = {_element_key(int(iz)): p_i[i] for i, iz in enumerate(izm + 1)}

    a = 1.0 + g2 + g3
    a = torch.clamp(a, 1e-15, 1e15)
    b = 2.0 * (1.0 + g2 / g5 * g4)
    b = torch.clamp(b, 1e-15, 1e15)
    c = torch.clamp(g5, 1e-15, 1e15)
    d = torch.clamp(g2 - g3, 1e-15, 1e15)
    e = torch.clamp(g2 / g5 * g4, 1e-15, 1e15)

    # quadratic for f1 = p(H)/p(H')
    c1 = c * b ** 2 + a * d * b - e * a ** 2
    c2 = 2.0 * a * e - d * b + a * b * g1
    c3 = -(e + b * g1)
    c1 = torch.clamp(c1, 1e-15, 1e15)

    f1 = 0.5 * c2 / c1
    f1 = -f1 + torch.sign(c1) * torch.sqrt(f1 ** 2 - c3 / c1)
    f1 = torch.clamp(f1, 1e-30, 1e30)

    f5 = (1.0 - a * f1) / b                          # p(H2)/p(H')
    f4 = e * f5                                      # p(H2+)/p(H')
    f3 = g3 * f1                                     # p(H-)/p(H')
    f2 = g2 * f1                                     # p(H+)/p(H')

    fe = torch.abs(f2 - f3 + f4 + g1)                # p(e-)/p(H')
    fe = torch.clamp(fe, 1e-15, 1e15)
    phtot = pe_safe / fe                             # p(H') = p(H)+p(H+)+...

    # iterate the H2 self-consistency (unconditional: 5 fixed iterations,
    # vmap-safe; the correction is negligible when H2 is not abundant)
    const6 = g5 / pe_safe * f1 ** 2
    const7 = torch.clamp(f2 - f3 + g1, 1e-15, 1e15)
    for _ in range(5):
        f5 = phtot * const6
        f4 = e * f5
        fe = torch.clamp(torch.abs(const7 + f4), 1e-15, 1e15)
        phtot = pe_safe / fe

    pg = pe_safe * (1.0 + (f1 + f2 + f3 + f4 + f5 + 0.1014) / fe)
    pg = torch.clamp(pg, 1e-20, 1e20)

    out = PartialPressure()
    out["h"] = f1                       # pg(1)  p(H)/p(H')
    out["he"] = pp.get("he", torch.zeros(shape, device=device, dtype=dtype))  # pg(2)
    out.update(pp)                      # pg(3..28): p(element)/p(H')
    out["pg"] = pg                      # pg(84) gas pressure [dyn/cm^2]
    out["h_prime"] = phtot              # pg(85) p(H')
    out["h+"] = f2                      # pg(86) p(H+)/p(H')
    out["h-"] = f3                      # pg(87) p(H-)/p(H')
    out["h2+"] = f4                     # pg(88) p(H2+)/p(H')
    out["h2"] = f5                      # pg(89) p(H2)/p(H')
    out["e-"] = fe                      # pg(90) p(e-)/p(H')
    out["ne"] = pe_safe / (1.38054e-16 * t)   # pg(91) n(e) [cm^-3]
    out["h+_h"] = g2                    # pg(92) p(H+)/p(H)
    out["h-_h"] = g3                    # pg(93) p(H-)/p(H)
    return out


class PartialPressure(dict):
    """
    Partial pressures per hydrogen nucleon, p(X)/p(H').

    Keys (partial-pressure convention):
        'h', 'h+', 'h-', 'h2+', 'h2', 'e-', 'h_prime', 'pg', 'ne',
        'he', 'li', 'be', 'b', 'c', 'n', 'o', 'f', 'ne2', 'na', 'mg',
        'al', 'si', 'p', 's', 'cl', 'ar', 'k', 'ca', 'sc', 'ti', 'v',
        'cr', 'mn', 'fe', 'co', 'ni'

    where 'h' = p(H)/p(H'), 'h+' = p(H+)/p(H'), 'h2' = p(H2)/p(H'),
    'e-' = p(e-)/p(H'), 'h_prime' = p(H'), 'pg' = gas pressure
    [dyn/cm^2], 'ne' = electron density [cm^-3] and the element names
    are p(element)/p(H').
    """


def ionization_equilibrium(theta, pe, abundance=None):
    """
    Solve the ionization + molecular equilibrium at given theta and Pe.

    Parameters
    ----------
    theta : torch.Tensor (..., 1)
        5040 / T.
    pe : torch.Tensor (..., 1)
        Electron pressure [dyn/cm^2].
    abundance : AbundanceTable, optional
        Element abundances; defaults to the packaged THEVENIN table.

    Returns
    -------
    PartialPressure
        All species partial pressures (same shape as ``theta``).
    """
    if abundance is None:
        abundance = AbundanceTable.from_default()

    u0, u1, u2, cmol = ionization_equilibrium_cache(theta, abundance)
    return ionization_equilibrium_cached(theta, pe, u0, u1, u2, cmol,
                                         abundance)


def _element_key(z):
    """Dict key for element Z (2..28): the lowercase symbol."""
    from ..physics.atoms import element_number_to_symbol
    return element_number_to_symbol(z).lower()


# ---------------------------------------------------------------------------
# partial-pressure derivatives: d ln(p)/dT and d ln(p)/dPe for every entry
# ---------------------------------------------------------------------------
def ionization_equilibrium_derivatives(theta, pe, abundance=None):
    """
    Logarithmic derivatives of all partial pressures.

    Returns ``(pp, dpp, ddpp)`` where ``pp`` is the same dict as
    :func:`ionization_equilibrium` and ``dpp``/``ddpp`` hold
    ``d ln(p)/dT`` and ``d ln(p)/dPe`` for each key (log-derivatives
    consumed by the opacity and line-synthesis derivatives).

    The equations are transcribed to match the reference implementation
    so the derivatives are consistent with :func:`ionization_equilibrium`.
    """
    if abundance is None:
        abundance = AbundanceTable.from_default()

    device, dtype = theta.device, theta.dtype
    t = 5040.0 / theta
    shape = theta.shape

    neg_pe = pe <= 0
    pe_safe = torch.where(neg_pe, torch.full_like(pe, 1e-10), pe)

    cmol = molecular_balance(theta)
    dcmol = molecular_balance_derivatives(theta)      # d ln(cm)/dT (molecb)
    cmol = torch.clamp(cmol, -30.0, 30.0)
    g4 = pe_safe * torch.pow(10.0, cmol[0])
    g5 = pe_safe * torch.pow(10.0, cmol[1])
    g4 = torch.where(neg_pe, torch.zeros_like(g4), g4)
    g5 = torch.where(neg_pe, torch.zeros_like(g5), g5)
    dg4 = dcmol[0] * 2.3025851                         # d ln g4/dT
    ddg4 = 1.0 / pe_safe                               # d ln g4/dPe
    dg5 = dcmol[1] * 2.3025851
    ddg5 = 1.0 / pe_safe

    chi1 = torch.tensor(IONIZATION_ENERGY_1[:N_ELEMENTS],
                        device=device, dtype=dtype)
    chi2 = torch.tensor(IONIZATION_ENERGY_2[:N_ELEMENTS],
                        device=device, dtype=dtype)
    rel_abu = torch.tensor(
        [abundance.relative[iz - 1] for iz in range(1, N_ELEMENTS + 1)],
        device=device, dtype=dtype)

    # partition functions + their d ln/dT for all elements
    from .derivatives import partition_derivatives
    u_list = [partition_functions(iz, t) for iz in range(1, N_ELEMENTS + 1)]
    du_list = [partition_derivatives(iz, t) for iz in range(1, N_ELEMENTS + 1)]
    u0 = torch.stack([u[0] for u in u_list])
    u1 = torch.stack([u[1] for u in u_list])
    u2 = torch.stack([u[2] for u in u_list])
    du0 = torch.stack([d[0] for d in du_list])
    du1 = torch.stack([d[1] for d in du_list])
    du2 = torch.stack([d[2] for d in du_list])

    def dln_saha(chi, dlow, dup):
        # d ln(saha)/dT
        return dup - dlow + (theta / 5040.0) * (2.5 + chi * theta * 2.3025851)

    # H ionization
    g2 = saha_ratio(theta, chi1[0], u0[0], u1[0], pe_safe)
    dg2 = dln_saha(chi1[0], du0[0], du1[0])            # d ln g2/dT
    ddg2 = -1.0 / pe_safe                              # d ln g2/dPe
    g3 = saha_ratio(theta, 0.754, torch.ones_like(u0[0]), u0[0], pe_safe)
    g3 = torch.clamp(g3, 1e-30, 1e30)
    g3 = 1.0 / g3                                      # p(H-)/p(H)
    dg3 = -1.0 * dln_saha(0.754, torch.zeros_like(du0[0]), du0[0])
    ddg3 = 1.0 / pe_safe

    # metals contribution to the electron budget (elements 2..N_ELEMENTS)
    theta4 = theta.unsqueeze(0)
    pe4 = pe_safe.unsqueeze(0)
    chi1m = chi1[1:].reshape(-1, 1, 1, 1)
    chi2m = chi2[1:].reshape(-1, 1, 1, 1)
    abum = rel_abu[1:].reshape(-1, 1, 1, 1)
    u0m, u1m, u2m = u0[1:], u1[1:], u2[1:]
    du0m, du1m, du2m = du0[1:], du1[1:], du2[1:]

    a = saha_ratio(theta4, chi1m, u0m, u1m, pe4)
    da = dln_saha(chi1m, du0m, du1m)                   # d ln a/dT
    dda = -1.0 / pe4                                   # d ln a/dPe
    b = saha_ratio(theta4, chi2m, u1m, u2m, pe4)
    dlb = b * dln_saha(chi2m, du1m, du2m)              # d b/dT (absolute)
    ddlb = -b / pe4                                    # d b/dPe
    c = 1.0 + a * (1.0 + b)
    c = torch.clamp(c, 1e-20, 1e20)
    p_i = abum / c
    dp_i = -(a * da * (1.0 + b) + a * dlb) / c
    ddp_i = -(a * dda * (1.0 + b) + a * ddlb) / c
    ss1 = torch.clamp(1.0 + 2.0 * b, 1e-20, 1e20)
    ss = p_i * a * ss1
    dss = dp_i + da + 2.0 * dlb / ss1
    ddss = ddp_i + dda + 2.0 * ddlb / ss1
    g1 = ss.sum(dim=0)
    dlg1 = (ss * dss).sum(dim=0)
    ddlg1 = (ss * ddss).sum(dim=0)

    # H2 self-consistency (a,b,c,d,e, f1, f5, f4, f3, f2, fe, phtot)
    aq = 1.0 + g2 + g3
    dla = g2 * dg2 + g3 * dg3
    ddla = g2 * ddg2 + g3 * ddg3

    bq = 2.0 * (1.0 + g2 / g5 * g4)
    dlbq = (bq - 2.0) * (dg2 - dg5 + dg4)
    ddlbq = (bq - 2.0) * (ddg2 - ddg5 + ddg4)
    cq = g5
    dlcq = dg5 * g5
    ddlcq = ddg5 * g5
    dq = g2 - g3
    dldq = g2 * dg2 - g3 * dg3
    ddldq = g2 * ddg2 - g3 * ddg3
    eq = g2 / g5 * g4
    de_log = dg2 - dg5 + dg4                          # d ln e/dT
    dde_log = ddg2 - ddg5 + ddg4                      # d ln e/dPe
    dleq = eq * de_log                                # d e/dT (absolute)
    ddleq = eq * dde_log                              # d e/dPe (absolute)

    def clamp_t(x):
        return torch.clamp(x, 1e-15, 1e15)

    aq, bq, cq, dq, eq = (clamp_t(x) for x in (aq, bq, cq, dq, eq))

    c1 = cq * bq ** 2 + aq * dq * bq - eq * aq ** 2
    dlc1 = (dlcq * bq * bq + (cq * 2.0 * bq + aq * dq) * dlbq
            + dla * (dq * bq - 2.0 * eq * aq) + dldq * aq * bq
            - dleq * aq * aq)
    ddlc1 = (ddlcq * bq * bq + (cq * 2.0 * bq + aq * dq) * ddlbq
             + ddla * (dq * bq - 2.0 * eq * aq) + ddldq * aq * bq
             - ddleq * aq * aq)
    c2 = 2.0 * aq * eq - dq * bq + aq * bq * g1
    dlc2 = (dla * (2.0 * eq + bq * g1) + dlbq * (aq * g1 - dq)
            - dldq * bq + dleq * 2.0 * aq + aq * bq * dlg1)
    ddlc2 = (ddla * (2.0 * eq + bq * g1) + ddlbq * (aq * g1 - dq)
             - ddldq * bq + ddleq * 2.0 * aq + aq * bq * ddlg1)
    c3 = -(eq + bq * g1)
    dlc3 = -dleq - dlbq * g1 - bq * dlg1
    ddlc3 = -ddleq - ddlbq * g1 - bq * ddlg1

    c1 = clamp_t(c1)
    f1 = 0.5 * c2 / c1
    dc1 = dlc1 / c1
    dc2 = dlc2 / c2
    dlf1 = f1 * (dc2 - dc1)
    ddc1 = ddlc1 / c1
    ddc2 = ddlc2 / c2
    ddlf1 = f1 * (ddc2 - ddc1)
    sgn = torch.sign(c1)
    dlf1 = -dlf1 + sgn * (2.0 * f1 * dlf1 - dlc3 / c1
                          + dlc1 * c3 / (c1 * c1)) \
        / (2.0 * torch.sqrt(f1 ** 2 - c3 / c1))
    ddlf1 = -ddlf1 + sgn * (2.0 * f1 * ddlf1 - ddlc3 / c1
                            + ddlc1 * c3 / (c1 * c1)) \
        / (2.0 * torch.sqrt(f1 ** 2 - c3 / c1))
    f1 = -f1 + sgn * torch.sqrt(f1 ** 2 - c3 / c1)

    f5 = (1.0 - aq * f1) / bq
    small5 = torch.abs(f5) < 1e-30
    dlf5 = (-dla * f1 - aq * dlf1) / bq \
        - ((1.0 - aq * f1) * dlbq) / (bq * bq)
    ddlf5 = (-ddla * f1 - aq * ddlf1) / bq \
        - ((1.0 - aq * f1) * ddlbq) / (bq * bq)
    dlf5 = torch.where(small5, torch.zeros_like(dlf5), dlf5)
    ddlf5 = torch.where(small5, torch.zeros_like(ddlf5), ddlf5)
    df5 = dlf5 / f5
    ddf5 = ddlf5 / f5
    df5 = torch.where(small5, torch.zeros_like(df5), df5)
    ddf5 = torch.where(small5, torch.zeros_like(ddf5), ddf5)

    f4 = eq * f5
    small4 = torch.abs(f4) < 1e-30
    dlf4 = f5 * dleq + eq * dlf5
    ddlf4 = f5 * ddleq + eq * ddlf5
    dlf4 = torch.where(small4, torch.zeros_like(dlf4), dlf4)
    ddlf4 = torch.where(small4, torch.zeros_like(ddlf4), ddlf4)
    df4 = dlf4 / f4
    ddf4 = ddlf4 / f4
    df4 = torch.where(small4, torch.zeros_like(df4), df4)
    ddf4 = torch.where(small4, torch.zeros_like(ddf4), ddf4)

    f3 = g3 * f1
    dlf3 = f3 * dg3 + g3 * dlf1
    ddlf3 = f3 * ddg3 + g3 * ddlf1
    f2 = g2 * f1
    dlf2 = f2 * dg2 + g2 * dlf1
    ddlf2 = f2 * ddg2 + g2 * ddlf1

    small1 = torch.abs(f1) < 1e-30
    divf1 = torch.where(small1, torch.zeros_like(f1), dlf1 / f1)
    ddivf1 = torch.where(small1, torch.zeros_like(f1), ddlf1 / f1)
    dlf200 = dg2 + divf1
    ddlf200 = ddg2 + ddivf1

    fe = f2 - f3 + f4 + g1
    smallfe = torch.abs(fe) < 1e-30
    dlfe = dlf2 - dlf3 + dlf4 + dlg1
    ddlfe = ddlf2 - ddlf3 + ddlf4 + ddlg1
    dlfe = torch.where(smallfe, torch.zeros_like(dlfe), dlfe)
    ddlfe = torch.where(smallfe, torch.zeros_like(ddlfe), ddlfe)
    dfe = dlfe / fe
    ddfe = ddlfe / fe
    dfe = torch.where(smallfe, torch.zeros_like(dfe), dfe)
    ddfe = torch.where(smallfe, torch.zeros_like(ddfe), ddfe)

    fe = clamp_t(fe)
    phtot = pe_safe / fe
    dphtot = -dfe
    ddphtot = 1.0 / pe_safe - ddfe

    # H2 self-consistency iteration (up to 5 passes; spot always iterates)
    if True:
        const6 = g5 / pe_safe * f1 ** 2
        dconst6 = dg5 + 2.0 * divf1
        ddconst6 = ddg5 - 1.0 / pe_safe + 2.0 * ddivf1
        const7 = f2 - f3 + g1
        const7 = clamp_t(const7)
        dlconst7 = dlf2 - dlf3 + dlg1
        ddlconst7 = ddlf2 - ddlf3 + ddlg1
        dconst7 = dlconst7 / const7
        ddconst7 = ddlconst7 / const7
        for _ in range(5):
            f5 = phtot * const6
            df5 = dphtot + dconst6
            ddf5 = ddphtot + ddconst6
            f4 = eq * f5
            df4 = de_log + df5
            ddf4 = dde_log + ddf5
            fe = const7 + f4
            fe = clamp_t(fe)
            dfe = (dlconst7 + df4 * f4) / fe
            ddfe = (ddlconst7 + ddf4 * f4) / fe
            phtot = pe_safe / fe
            dphtot = -dfe
            ddphtot = 1.0 / pe_safe - ddfe
        dlf5 = df5 * f5
        ddlf5 = ddf5 * f5
        dlf4 = df4 * f4
        ddlf4 = ddf4 * f4
        dlfe = dfe * fe
        ddlfe = ddfe * fe

    pg = pe_safe * (1.0 + (f1 + f2 + f3 + f4 + f5 + 0.1014) / fe)
    dlpg = pe_safe * (dlf1 + dlf2 + dlf3 + dlf4 + dlf5) / fe \
        - (pg - pe_safe) * (dlfe / fe)
    ddlpg = pe_safe * (ddlf1 + ddlf2 + ddlf3 + ddlf4 + ddlf5) / fe \
        - (pg - pe_safe) * (ddlfe / fe)
    ddlpg = ddlpg + pg / pe_safe
    pg = clamp_t(pg)

    # assemble output dicts (keys mirror ionization_equilibrium)
    pp = {}
    dpp = {}
    ddpp = {}
    pp["h"] = f1
    dpp["h"] = divf1
    ddpp["h"] = ddivf1
    pp["pg"] = pg
    dpp["pg"] = dlpg / pg
    ddpp["pg"] = ddlpg / pg
    pp["h_prime"] = phtot
    dpp["h_prime"] = dphtot
    ddpp["h_prime"] = ddphtot
    pp["h+"] = f2
    dpp["h+"] = dlf200
    ddpp["h+"] = ddlf200
    small3 = torch.abs(f3) < 1e-30
    divf3 = torch.where(small3, torch.zeros_like(f3), dlf3 / f3)
    ddivf3 = torch.where(small3, torch.zeros_like(f3), ddlf3 / f3)
    pp["h-"] = f3
    dpp["h-"] = divf3
    ddpp["h-"] = ddivf3
    pp["h2+"] = f4
    dpp["h2+"] = df4
    ddpp["h2+"] = ddf4
    pp["h2"] = f5
    dpp["h2"] = df5
    ddpp["h2"] = ddf5
    pp["e-"] = fe
    dpp["e-"] = dfe
    ddpp["e-"] = ddfe
    pp["h+_h"] = g2
    dpp["h+_h"] = dg2
    ddpp["h+_h"] = ddg2
    pp["h-_h"] = g3
    dpp["h-_h"] = dg3
    ddpp["h-_h"] = ddg3
    izm = torch.arange(1, N_ELEMENTS, device=device)
    for i, iz in enumerate(izm + 1):
        key = _element_key(int(iz))
        pp[key] = p_i[i]
        dpp[key] = dp_i[i]
        ddpp[key] = ddp_i[i]
    pp["he"] = pp.get("he", torch.zeros(shape, device=device, dtype=dtype))
    dpp["he"] = dpp.get("he", torch.zeros(shape, device=device, dtype=dtype))
    ddpp["he"] = ddpp.get("he", torch.zeros(shape, device=device, dtype=dtype))
    # electron density n(e) = pe/kT  (pg(91)); MUST come after the element
    # loop, whose key "ne" (Ne, Z=10) would otherwise overwrite it
    pp["ne"] = pe_safe / (1.38054e-16 * t)
    dpp["ne"] = -1.0 / t
    ddpp["ne"] = 1.0 / pe_safe
    return pp, dpp, ddpp
