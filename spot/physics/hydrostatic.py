# -*- coding: utf-8 -*-
"""
spot.physics.hydrostatic — hydrostatic-equilibrium (HSE) operators
===================================================================
Hydrostatic-consistency machinery used when ``Pe`` is NOT a free
parameter of the inversion (``mnodos(2) == 0``):

1. ``hydrostatic_pe_coupling`` — the *linearized* coupling operator
   ``M = dPe/dT``.  With a fixed gas-pressure stratification, the
   electron-pressure response is carried into the temperature response
   function through the operator ``grt = rt4 + M^T . grp``.

2. ``pe_from_pg`` + ``hydrostatic_pe_cont`` — the *nonlinear* per-trial
   hydrostatic solve: with a fixed surface gas pressure (e.g. 'Gas
   pressure at surface 1: 500' in the .trol), every trial atmosphere of
   the inversion has its Pe stratification recomputed from T and
   hydrostatic equilibrium (the m(2) == 0 branch).

Both are vectorized with torch (no loops over wav/Stokes; the depth
loop of the solve keeps the same sequential Gauss-Seidel sweep order as
the reference implementation).
"""

import torch

from .pressure import ionization_equilibrium

__all__ = ["hydrostatic_pe_coupling", "pe_from_pg", "hydrostatic_pe_cont"]

_G = 2.7414e4          # solar surface gravity: g = mu * 2.7414e4 cm/s^2, mu = 1
_AVOG = 6.023e23
_LN10 = 2.3025851
_CGAS = 8.31451e7      # ideal-gas constant R [erg/(mol K)]

def _pmu(abundance):
    """Mean molecular weights over the 92 elements: (sum(weight*abu), sum(abu))."""
    from .atoms import ATOM_WEIGHTS, AbundanceTable
    if abundance is None:
        abundance = AbundanceTable.from_default()
    w = ATOM_WEIGHTS
    a = abundance.relative
    return float((w * a).sum()), float(a.sum())


def hydrostatic_pe_coupling(ltau, t, pe, pp, dpp, ddpp, kappa5,
                            dkappa5, ddkappa5, abundance=None):
    """
    Hydrostatic coupling operator dPe/dT (the linearized HSE operator).

    Parameters
    ----------
    ltau : torch.Tensor (Nt,)
        log10(tau5000), decreasing (index 0 = deepest).  The reference
        convention stores the deepest layer at index 1, so the Python
        index is one less than the reference index.
    t, pe : torch.Tensor (Nb, Nt, 1)
        Temperature / electron pressure (state at which the operator
        is linearized; Pe only enters through the gasb derivatives).
    pp, dpp, ddpp : dicts (spot gasb output)
        Partial pressures and their log-derivatives; ``pg`` key is the
        gas pressure ``pg(84)``.
    kappa5, dkappa5, ddkappa5 : torch.Tensor (Nb, Nt, 1)
        Continuum opacity at 5000 A and its absolute d/dT, d/dPe (per H
        nucleon; multiplied by avog/pmusum here, exactly like
        ``kac = ck5*cc``, ``kat = dk5*cc``).
    abundance : AbundanceTable or None.

    Returns
    -------
    torch.Tensor (Nb, Nt, Nt)
        ``M[j, kk]`` = d Pe(j)/d T(kk) (python indices, 0 = deepest;
        reference index = python index + 1).  Only ``j <= kk`` entries
        are non-zero:
        ``M[kk,kk] = px(kk)``, ``M[kk-1,kk] = qx(kk)``,
        ``M[j,kk] = wx(j,kk)`` for ``j <= kk-2``.
    """
    nt = ltau.shape[0]
    dev, dtp = t.device, t.dtype

    pmusum, _ = _pmu(abundance)
    cc = _AVOG / pmusum

    # kappa (per unit mass) and its T/Pe derivatives at 5000 A
    kac = kappa5 * cc                          # (Nb,Nt,1)
    kat = dkappa5 * cc
    kap = ddkappa5 * cc

    # gas-pressure derivatives at the current state (gasb pg(84));
    # all are already (Nb, Nt, 1)
    pgas = pp["pg"]
    dpgas = dpp["pg"]
    ddpgas = ddpp["pg"]

    # first-order gas-pressure and opacity derivatives:
    fx = dpgas * pgas                          # dPg/dT
    cx = 1.0 / (ddpgas * pgas)                 # dPe/dPg
    bx = kap / (ddpgas * pgas)                 # d kappa/dPg
    ax = kat - kap * dpgas / ddpgas            # d kappa/dT | const Pg

    # x(i) = g*(tau(i-1)-tau(i))*ln10 : the hydrostatic depth increment
    # between neighbouring layers (the top value reuses the first one).
    dtau = (ltau[:-1] - ltau[1:]) * _LN10      # (Nt-1,)
    x = torch.zeros(nt, device=dev, dtype=dtp)
    x[1:] = dtau * _G
    x[0] = x[1]

    taue = torch.pow(10.0, ltau)               # (Nt,)
    tauk = taue[None, :, None] / 2.0 / (kac ** 2)     # (Nb,Nt,1)
    d1x = x[None, :, None] * tauk              # (Nb,Nt,1)
    d2x = d1x                                  # d2x(i)=x(i)*tauk as well
    dx = 1.0 + d1x * bx                        # (Nb,Nt,1)

    # rx/sx/tx/px/qx  (all (Nb,Nt); index nt-1 = top)
    rx = torch.zeros((t.shape[0], nt), device=dev, dtype=dtp)
    sx = torch.zeros((t.shape[0], nt), device=dev, dtype=dtp)
    tx = torch.zeros((t.shape[0], nt), device=dev, dtype=dtp)
    px = torch.zeros((t.shape[0], nt), device=dev, dtype=dtp)
    qx = torch.zeros((t.shape[0], nt), device=dev, dtype=dtp)

    for i in range(nt - 1):                        # depth index i+1 .. ntau-1
        dxi = dx[:, i, 0]
        rx[:, i] = (1.0 - d2x[:, i + 1, 0] * bx[:, i + 1, 0]) / dxi
        sx[:, i] = (d2x[:, i + 1, 0] / dxi) * ax[:, i + 1, 0]
        tx[:, i] = (d1x[:, i, 0] / dxi) * ax[:, i, 0]
        px[:, i] = -cx[:, i, 0] * (tx[:, i] + fx[:, i, 0])
    for i in range(1, nt):                         # depth index 2..ntau
        qx[:, i] = -cx[:, i - 1, 0] * (sx[:, i - 1] + tx[:, i] * rx[:, i - 1])

    # wx(i,j): j = i+2 .. ntau-2  (python j = i+2 .. nt-3)
    # NOTE (index-parity of the reference formula, line 280):
    #   wxi = -(sx(j-1) + tx(j)*rx(j-1)) * r1   (1-based reference indices)
    # with python j = j_ref-1 this becomes
    #   wxi = -(sx[j-1] + tx[j]*rx[j-1]) * prod
    # (an earlier port used sx[j]/tx[j+1]/rx[j], i.e. shifted one layer
    #  towards the top; the reference is reproduced only with the shift).
    wx = torch.zeros((t.shape[0], nt, nt), device=dev, dtype=dtp)
    for i in range(nt - 1):
        prod = torch.ones(t.shape[0], device=dev, dtype=dtp)
        for j in range(i + 2, nt - 2):
            # r1 = prod_{k_py=i}^{j-2} rx(k_py)   (reference k=i..j-2)
            prod = prod * rx[:, j - 2]
            wxi = -(sx[:, j - 1] + tx[:, j] * rx[:, j - 1]) * prod
            wx[:, i, j] = cx[:, i, 0] * wxi

    # assemble M (python indexing; py i = reference i-1)
    m = wx.clone()
    for kk in range(nt):
        m[:, kk, kk] = px[:, kk]
        if kk > 0:
            m[:, kk - 1, kk] += qx[:, kk]
    return m


def _inicia_pefrompgt(t, pg):
    """Pe estimate from (T, Pg) assuming only Hydrogen is ionized (H-only Saha)."""
    nu = 0.9091                              # only Hydrogen is ionized
    saha = (-0.4771 + 2.5 * torch.log10(t) - torch.log10(pg)
            - 13.6 * 5040.0 / t)
    saha = torch.pow(10.0, saha)
    aaa = 1.0 + saha
    bbb = -(nu - 1.0) * saha
    ccc = -saha * nu
    ybh = (-bbb + torch.sqrt(bbb * bbb - 4.0 * aaa * ccc)) / (2.0 * aaa)
    return pg * ybh / (1.0 + ybh)


def pe_from_pg(t, pg, pe0, abundance=None, prec=1.0e-5, max_iter=250,
               _u=None):
    """
    Electron pressure from (T, Pg) by a fixed-point iteration.

    One fixed-point step is exactly ``pe = pe0 * pg / pg_gasb(pe0)``:
    the equation ``pe = pg / (1 + (f1+f2+f3+f4+f5+0.1014)/fe)`` and the
    companion ``pg = pe * (1 + (...)/fe)`` with the *same*
    f1..f5/fe at the same (T, pe) are mutually inverse.
    The iteration sequence mirrors the reference fixed point (including
    the (p+p1)/2 averaging and the H-only restart when the first step is
    off by more than a factor of 10).

    Parameters
    ----------
    t, pg, pe0 : torch.Tensor (...,)
        Temperature [K], target gas pressure [dyn/cm^2], initial Pe
        estimate [dyn/cm^2] (same trailing shape).
    _u : optional (u0, u1, u2, cmol) from
        :func:`spot.physics.pressure.ionization_equilibrium_cache` at
        the same theta (internal fast path of the HSE solver).

    Returns
    -------
    torch.Tensor, same shape as ``t``: Pe [dyn/cm^2].
    """
    from .pressure import (ionization_equilibrium_cache,
                           ionization_equilibrium_cached)
    t2 = t.clamp(min=500.0)                # pe: T < 500 -> 500 K
    pg2 = pg.to(t2)
    theta = 5040.0 / t2
    if _u is None:
        u0, u1, u2, cmol = ionization_equilibrium_cache(theta, abundance)
    else:
        u0, u1, u2, cmol = _u

    def step(pcur):
        pp = ionization_equilibrium_cached(theta, pcur, u0, u1, u2, cmol,
                                           abundance)
        pgc = pp["pg"].clamp(min=1.0e-20)
        return pcur * pg2 / pgc

    p = pe0.to(t2).clone()
    p0 = p.clone()
    p = step(p)
    dif = ((p - p0).abs() / p.abs()).clamp(min=0.0)
    # first step off by more than 10x: restart from the H-only estimate
    p = torch.where(dif > 1.0, _inicia_pefrompgt(t2, pg2), p)
    p = p.clamp(min=1.0e-8)
    p1 = p.clone()
    n2 = 0
    while bool((dif > prec).any()) and n2 < max_iter:
        n2 += 1
        p = (p + p1) / 2.0
        p1 = p.clone()
        p = step(p)
        dif = ((p - p1).abs() / p.abs()).clamp(min=0.0)
    return p


def hydrostatic_pe_cont(ltau, t, pe, pg0, abundance=None,
                        inner_max=30, inner_prec=1.0e-5, warn=True):
    """
    HSE-consistent Pe stratification.

    Integrates the hydrostatic equilibrium downward from the surface
    (top of the grid) with a FIXED gas pressure ``pg0`` at the top
    (the boundary condition e.g. 'Gas pressure at surface 1: 500' in
    the .trol) and, at every depth, obtains Pe(T, Pg) through
    :func:`pe_from_pg` using the continuum opacity at 5000 A (at
    5e-5 cm) and the mean molecular weight of the mixture.  This is the
    Pe used by the inversion loop for every trial atmosphere when Pe is
    NOT inverted (the m(2) == 0 branch).

    Parameters
    ----------
    ltau : torch.Tensor (Nt,)
        log10(tau5000), decreasing (0 = deepest).
    t : torch.Tensor (Nb, Nt, 1)
        Temperature [K].
    pe : torch.Tensor (Nb, Nt, 1)
        Initial Pe (used as the first estimate of the top value; the
        result does not depend on it).
    pg0 : float
        Gas pressure at the surface (top) [dyn/cm^2].

    Returns
    -------
    (pe_out, pg)
        pe_out : torch.Tensor (Nb, Nt, 1)  HSE electron pressure.
        pg : torch.Tensor (Nb, Nt)         HSE gas pressure (diagnostic;
        ro/y/z from ro(i)=pesomedio*pg(i)/tsi/cgases are not returned).
    """
    from . import opacity as _opacity
    from .pressure import ionization_equilibrium_cache, \
        ionization_equilibrium_cached

    nt = ltau.shape[0]
    nb = t.shape[0]
    dev, dtp = t.device, t.dtype
    pg0 = float(pg0)

    pmusum, asum = _pmu(abundance)
    cc = _AVOG / pmusum

    # columns / optical depth (cth = 1: tau = tau1)
    taue = torch.pow(10.0, ltau)                       # (Nt,)
    x = torch.zeros(nt, device=dev, dtype=dtp)
    x[1:] = (ltau[:-1] - ltau[1:]) * _LN10 * _G
    x[0] = x[1]

    tS = t[..., 0].contiguous()                        # (Nb, Nt)
    peS = pe[..., 0].clone()                           # (Nb, Nt)
    pg = torch.empty((nb, nt), device=dev, dtype=dtp)
    kappa = torch.empty((nb, nt), device=dev, dtype=dtp)

    lam5 = torch.tensor([5000.0], device=dev, dtype=dtp)

    # partition functions / molecular coefficients depend on T only:
    # compute them once for every depth, then re-use (the pe_from_pg
    # fixed point and the kappa(gasc) updates use per-depth slices).
    u0, u1, u2, cmol = ionization_equilibrium_cache(5040.0 / t, abundance)
    uc_top = (u0[:, :, nt - 1:, :], u1[:, :, nt - 1:, :],
              u2[:, :, nt - 1:, :], cmol[:, :, nt - 1:, :])

    def _kappa(tq, psi, uc):
        """gasc at (tq, psi) -> kappa(5000 A) per unit mass."""
        pp = ionization_equilibrium_cached(5040.0 / tq, psi,
                                           uc[0], uc[1], uc[2], uc[3],
                                           abundance)
        kac = _opacity.continuum_opacity(lam5, tq, psi, pp,
                                         refractive_index=1.0)
        return kac[..., 0] * cc

    # top of the atmosphere (last depth index): fixed boundary state.
    # The first estimate used by pe_from_pg is the T-only Saha estimate
    # (NOT the input Pe): the fixed point is slow to converge at the deep
    # dense layers, so a non-deterministic seed (the caller's Pe column,
    # which changes between inversion steps) would leave a seed-
    # dependent residue and break the reproducibility of the trial
    # atmospheres.  With a deterministic seed, Pe(T, pg0) is unique.
    ttop_q = t[:, nt - 1:, :]
    pg0_q = torch.full_like(ttop_q, pg0)
    psi0 = _inicia_pefrompgt(ttop_q, pg0_q)
    psi_top = pe_from_pg(ttop_q, pg0_q, psi0, abundance, _u=uc_top)
    peS[:, nt - 1] = psi_top[:, 0, 0]
    kappa[:, nt - 1] = _kappa(ttop_q, psi_top, uc_top)[:, 0]
    pg[:, nt - 1] = pg0
    # (The reference also computes ro/y/z here with
    # pesomedio=pmusum/(asum+fe) of the top state; they are diagnostic
    # outputs that do not feed back into Pe/Pg, so they are not returned
    # by this port.)

    # integrate downward: sweep the depths from ntau-1 down to 0, exactly
    # reversing the grid order.
    # (Jacobi-sweep batching was tested and converges to a slightly
    # different discrete fixed point, so the sequential Gauss-Seidel
    # order is kept; the per-depth ionization-equilibrium evaluations are
    # fast because the T-dependent partition functions/molecular
    # constants are cached.)
    for j in range(nt - 2, -1, -1):
        xj = x[j + 1]
        pgpr = pg[:, j + 1] + xj * taue[j + 1] / kappa[:, j + 1]
        tq = t[:, j:j + 1, :]                     # (Nb, 1, 1)
        uc = tuple(u[:, :, j:j + 1, :] for u in (u0, u1, u2, cmol))
        # pass the previous (shallower) depth's Pe as the first estimate
        # of pe_from_pg (fewer fixed-point iterations)
        psi = pe_from_pg(tq, pgpr.unsqueeze(-1).unsqueeze(-1),
                         peS[:, j + 1:j + 2].unsqueeze(-1), abundance,
                         _u=uc)
        peS[:, j] = psi[:, 0, 0]
        km = _kappa(tq, psi, uc)[:, 0]            # kappa(j): (Nb,)
        pgj = pg[:, j + 1] + xj * (taue[j + 1] / kappa[:, j + 1]
                                   + taue[j] / km) / 2.0

        nit = 1
        dif = 2.0 * (pgj - pgpr).abs() / (pgj + pgpr)
        while nit < inner_max and bool((dif > inner_prec).any()):
            nit += 1
            pgpr = pgj
            psi = pe_from_pg(tq, pgpr.unsqueeze(-1).unsqueeze(-1),
                             peS[:, j:j + 1].unsqueeze(-1), abundance,
                             _u=uc)
            peS[:, j] = psi[:, 0, 0]
            km = _kappa(tq, psi, uc)[:, 0]
            pgj = pg[:, j + 1] + xj * (taue[j + 1] / kappa[:, j + 1]
                                       + taue[j] / km) / 2.0
            dif = 2.0 * (pgj - pgpr).abs() / (pgj + pgpr)
        if warn and bool((dif > 0.1).any()):
            print("WARNING (spot hydrostatic_pe_cont): inaccurate "
                  "electronic pressures, dif=%.3e" % float(dif.max()))

        psi = pe_from_pg(tq, pgj.unsqueeze(-1).unsqueeze(-1),
                         peS[:, j:j + 1].unsqueeze(-1), abundance, _u=uc)
        peS[:, j] = psi[:, 0, 0]
        kappa[:, j] = km
        pg[:, j] = pgj


    return peS.unsqueeze(-1), pg
