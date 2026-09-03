# -*- coding: utf-8 -*-
"""
spot.physics.line — line opacity & Zeeman components and damping
================================================================
Spot's line module: it computes the LS-coupling (Russell-Saunders)
Zeeman pattern of a spectral line, the line absorption coefficient per
unit tau5000, and the line damping parameter (Unsöld or Barklem/ABO).
The Zeeman-pattern, line-opacity and damping routines follow the
standard spectral-synthesis conventions.
"""

import numpy as np
import torch

from .constants import ATOMIC_MASS_UNIT, GAS, PI
from .ionization import saha_ratio
from .partition import partition_functions

__all__ = ["zeeman_pattern", "line_opacity", "damping_parameter",
           "component_profiles"]

# Orbital-angular-momentum (L) letter codes: integer values
_LETTER_CODE = {'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6,
                'K': 7, 'L': 8, 'M': 9, 'N': 10, 'O': 11, 'Q': 12}
# Orbital-angular-momentum (L) letter codes: half-integer values
_LETTER_CODE_HALF = {'p': 0.5, '1': 1.5, 'f': 1.5, '2': 2.5, 'h': 2.5,
                     '3': 3.5, 'k': 3.5, '4': 4.5, 'm': 4.5, '5': 5.5,
                     'o': 5.5, '6': 6.5, 'r': 6.5, '7': 7.5, 't': 7.5,
                     '8': 8.5, 'u': 8.5, '9': 9.5, 'v': 9.5, '0': 10.5,
                     'w': 10.5}


def _lande_factor(mult, design, tam):
    """Lande factor of a level from its term (LS-coupling formula)."""
    spin = 0.5 * (mult - 1.0)
    if design in _LETTER_CODE:
        oam = float(_LETTER_CODE[design])
    elif design in _LETTER_CODE_HALF:
        oam = _LETTER_CODE_HALF[design]
    elif design in ("X", "x"):
        # unknown orbital angular momentum ('X'): take L=0
        oam = 0.0
    else:
        raise ValueError(f"unknown orbital angular momentum letter {design!r}")
    if tam == 0.0:
        # a J=0 level is degenerate, so the J=1 expression (TAM(I)=1) is used
        return 1.5 + (spin * (1.0 + spin) - oam * (1.0 + oam)) / 4.0
    return 1.5 + (spin * (1.0 + spin) - oam * (1.0 + oam)) / (2.0 * tam * (1.0 + tam))


def zeeman_pattern(mult_l, design_l, tam_l, mult_u, design_u, tam_u, dl0):
    """
    Zeeman pattern of a line in the LS-coupling (Russell-Saunders) limit.

    Parameters
    ----------
    mult_l, mult_u : int
        Multiplicities 2S+1 of the lower/upper levels.
    design_l, design_u : str
        Orbital angular momentum letters (e.g. 'P', 'D', 'F').
    tam_l, tam_u : float
        Total angular momenta J of the lower/upper levels.
    dl0 : float
        Zeeman splitting per Gauss in wavelength units
        (``4.6686e-5 * lambda_cm**2``), i.e. the unit shift.

    Returns
    -------
    shifts : np.ndarray (Nc,)
        Wavelength shifts of all components [cm/Gauss] (components are
        grouped as pi / sigma_r / sigma_l in that order).
    strengths : np.ndarray (Nc,)
        Normalized intensities of the components (each Zeeman group is
        normalized to unit sum within the group).
    npi, nr : int
        Number of pi and sigma_r components (the sigma_l group has
        Nc - npi - nr components).
    """
    g = [_lande_factor(mult_l, design_l, tam_l),
         _lande_factor(mult_u, design_u, tam_u)]

    # --- special cases ------------------------------------------------------
    # * a level with J = 0  ->  single-component "triplet" with G from the
    #   J=1 Landé formula;
    # * g_lower == g_upper  ->  single component at 0 and sigma at +/-G.
    if tam_l == 0.0 or tam_u == 0.0:
        # For Jl = 0 the pattern is built from the UPPER level's term with
        # TAM=1 (I=2); for Ju = 0 it is built from the LOWER level's term
        # (I=1).  In both cases the lines reduce to the simple triplet
        # DLP=0, DLR=-G, DLL=+G (one component each).
        g_val = g[1] if tam_l == 0.0 else g[0]
        return (np.array([0.0, -g_val, g_val]) * dl0,
                np.array([1.0, 1.0, 1.0]), 1, 1)
    if abs(g[1] - g[0]) < 5e-6:
        return (np.array([0.0, g[0], -g[0]]) * dl0,
                np.array([1.0, 1.0, 1.0]), 1, 1)

    jl, ju = tam_l, tam_u
    level = ju - jl          # +1: Ju>Jl, 0: Ju==Jl, -1: Ju<Jl
    half_int = (2 * jl) % 2 == 1   # half-integer J's

    shifts, strengths = [], []
    if not half_int:
        # ---- integer J's ----
        # pi components: MU = -min(Jl,Ju) .. +min(Jl,Ju)
        mumin = -min(jl, ju)
        mumax = -mumin
        for mu in range(int(mumin), int(mumax) + 1):
            if mu == 0 and level == 0:
                continue
            shift = mu * (g[0] - g[1])
            if level < 0:
                s = 2 * (jl ** 2 - mu ** 2)
            elif level == 0:
                s = 2 * mu ** 2
            else:
                s = 2 * (ju ** 2 - mu ** 2)
            shifts.append(shift)
            strengths.append(s)
        npi = len(shifts)
        # sigma_r components: MU = 1-Jl .. Ju
        for mu in range(int(1 - jl), int(ju) + 1):
            shift = mu * (g[0] - g[1]) - g[0]
            if level < 0:
                s = (jl - mu) * (ju - mu + 2)
            elif level == 0:
                s = (ju + mu) * (ju - mu + 1)
            else:
                s = (ju + mu) * (jl + mu)
            shifts.append(shift)
            strengths.append(s)
        nsigr = len(shifts) - npi
        # sigma_l components: MU = -Ju .. Jl-1
        for mu in range(int(-ju), int(jl)):
            shift = mu * (g[0] - g[1]) + g[0]
            if level < 0:
                s = (jl + mu) * (ju + mu + 2)
            elif level == 0:
                s = (ju - mu) * (ju + mu + 1)
            else:
                s = (ju - mu) * (jl - mu)
            shifts.append(shift)
            strengths.append(s)
    else:
        # ---- half-integer J's ----
        # pi components
        mumin = -ju if jl >= ju else -jl
        mumax = 1 - mumin
        for mu in range(int(mumin), int(mumax) + 1):
            spin = mu - 0.5
            shift = (g[0] - g[1]) * spin
            s2 = spin ** 2
            if level < 0:
                s = 2 * ((jl + 0.5) ** 2 - s2)
            elif level == 0:
                s = 2 * s2
            else:
                s = 2 * ((ju + 0.5) ** 2 - s2)
            shifts.append(shift)
            strengths.append(s)
        npi = len(shifts)
        # sigma_r
        for mu in range(int(-jl), int(ju) + 1):
            shift = (mu + 0.5) * (g[0] - g[1]) - g[0]
            if level < 0:
                s = (jl - mu) * (ju - mu + 2)
            elif level == 0:
                s = (ju + mu + 1) * (ju - mu + 1)
            else:
                s = (ju + mu + 1) * (ju + mu)
            shifts.append(shift)
            strengths.append(s)
        nsigr = len(shifts) - npi
        # sigma_l
        for mu in range(int(-mumax), int(jl) + 1):
            shift = (mu - 0.5) * (g[0] - g[1]) + g[0]
            if level < 0:
                s = (jl + mu) * (ju + mu + 2)
            elif level == 0:
                s = (ju - mu + 1) * (ju + mu + 1)
            else:
                s = (ju - mu + 1) * (jl - mu + 1)
            shifts.append(shift)
            strengths.append(s)

    shifts = np.asarray(shifts, dtype=float)
    strengths = np.asarray(strengths, dtype=float)

    # --- normalize each Zeeman group to unit sum ---------------------------
    if not half_int:
        npi = len(range(int(-min(jl, ju)), int(min(jl, ju)) + 1))
        if level == 0:
            npi -= 1                       # MU=0 skipped for Ju==Jl
        nr = len(range(int(1 - jl), int(ju) + 1))
    else:
        npi = len(range(int(mumin), int(mumax) + 1))
        nr = len(range(int(-jl), int(ju) + 1))
    npi = int(npi)
    nr = int(nr)
    strengths[:npi] /= strengths[:npi].sum()
    strengths[npi:npi + nr] /= strengths[npi:npi + nr].sum()
    strengths[npi + nr:] /= strengths[npi + nr:].sum()

    return shifts * dl0, strengths, npi, nr


def component_profiles(v, a, shifts, strengths, field_doppler):
    """
    Sum the Voigt/Faraday profiles of all Zeeman components.

    Parameters
    ----------
    v : torch.Tensor
        (...,) line-center frequency shift in Doppler units.
    a : torch.Tensor
        Damping parameter (same shape).
    shifts : np.ndarray (Nc,)
        Component shifts [cm] (already multiplied by the Zeeman unit).
    strengths : np.ndarray (Nc,)
        Component strengths.
    field_doppler : torch.Tensor
        B/dldop [Gauss/cm], same shape as v.

    Returns
    -------
    eta, rho : torch.Tensor
        Total absorption (Voigt) and dispersion (Faraday) profiles.
    """
    from .voigt import voigt_faraday_profile

    eta = torch.zeros_like(v)
    rho = torch.zeros_like(v)
    for shift, strength in zip(shifts, strengths):
        vcomp = v + shift * field_doppler
        h, f = voigt_faraday_profile(vcomp, a)
        eta = eta + strength * h
        rho = rho + strength * f * 2.0   # dispersion profile doubled (reference convention)
    return eta, rho


def component_profiles_batched(v, a, shifts_all, strengths_all, npi, nr,
                               field_doppler, max_elements=4.0e7):
    """
    Batched Zeeman component profiles: ONE Voigt/Faraday evaluation over
    an extra component axis (all components evaluated in a single call).

    ``shifts_all`` / ``strengths_all`` are the concatenated
    (pi, sigma_r, sigma_l) component arrays; the three Zeeman groups are
    contiguous with ``npi`` / ``nr`` (and ``Nc - npi - nr``) components.
    The component axis is processed in blocks of at most
    ``max_elements`` profile elements per block so that the (Nb,Nt,Nw,Nc)
    intermediates stay bounded even for large batches (a full
    ``Nc = 12`` block at batch 2048 would allocate ~7 GB per tensor,
    which no longer fits next to the K-matrix temporaries).

    Returns (eta_pi, rho_pi, eta_r, rho_r, eta_l, rho_l) — the same six
    arrays as three sequential :func:`component_profiles` calls with the
    order used by the propagation-matrix assembly.
    """
    from .voigt import voigt_faraday_profile

    nc = int(strengths_all.shape[0])
    nb, nt, nw = v.shape
    # number of components evaluated together (memory bound)
    nblock = max(1, int(max_elements / max(nb * nt * nw, 1)))
    nblock = min(nblock, nc)

    # per-group offsets (pi: 0..npi, r: npi..npi+nr, l: rest)
    idx_pi = slice(0, npi)
    idx_r = slice(npi, npi + nr)
    idx_l = slice(npi + nr, nc)

    etas = [torch.zeros_like(v) for _ in range(3)]
    rhos = [torch.zeros_like(v) for _ in range(3)]
    groups = (idx_pi, idx_r, idx_l)
    for i0 in range(0, nc, nblock):
        i1 = min(i0 + nblock, nc)
        sh = shifts_all[i0:i1].to(v.device).to(v.dtype)
        st = strengths_all[i0:i1].to(v.device).to(v.dtype)
        vcomp = v.unsqueeze(-1) + sh.reshape(1, 1, 1, -1) * field_doppler.unsqueeze(-1)
        h, f = voigt_faraday_profile(vcomp, a.unsqueeze(-1))
        h = h * st.reshape(1, 1, 1, -1)
        f = f * st.reshape(1, 1, 1, -1)
        for g, sl in enumerate(groups):
            lo = max(0, sl.start - i0)          # local index in the block
            hi = min(nblock, sl.stop - i0)
            if hi > lo:
                etas[g] = etas[g] + h[..., lo:hi].sum(-1)
                rhos[g] = rhos[g] + f[..., lo:hi].sum(-1)
    # dispersion profile doubled (matches the reference convention)
    return (etas[0], 2.0 * rhos[0], etas[1], 2.0 * rhos[1],
            etas[2], 2.0 * rhos[2])


def line_opacity(theta, t, pe, energy_low, loggf, gf_abu, wavelength_cm,
                 weight, u0, u1, u2, chi1, chi2, vdop, kappa5,
                 stimulated=True, istage=1, u1_ratio=None, ne=None):
    """
    Line absorption coefficient per unit tau5000 (eta0).

    Parameters
    ----------
    theta : torch.Tensor
        5040/T.
    t : torch.Tensor
        Temperature [K].
    pe : torch.Tensor
        Electron pressure [dyn/cm^2].
    energy_low : float
        Excitation potential of the lower level [eV].
    loggf : float
        log10(gf).
    gf_abu : float
        gf * abundance (relative to H) — precomputed by the caller.
    wavelength_cm : float
        Vacuum wavelength [cm].
    weight : float
        Atomic weight [g/mol].
    u0, u1, u2 : torch.Tensor
        Partition functions of neutral/ion/double-ion.
    chi1, chi2 : float
        First/second ionization potentials [eV].
    vdop : torch.Tensor
        Doppler velocity sqrt(2kT/m + vmic^2) [cm/s].
    kappa5 : torch.Tensor
        Continuum opacity at 5000 A per H nucleon [cm^2/H].
    stimulated : bool
        Include the stimulated-emission correction 1-exp(-hc/lambda kT).
    istage : int
        Ionization stage of the line (1 neutral, 2 singly ionized).
    u1_ratio : torch.Tensor, optional
        Precomputed u1/u2*u12 factor for ionized lines.

    Returns
    -------
    eta0 : torch.Tensor
        Line opacity per unit tau5000 (kappa_L / kappa5).
    """
    eta00 = 1.49736e-2 * gf_abu * wavelength_cm
    # Debye lowering of the ionization potentials by the free electrons:
    #   chi = chi0 - C * ne^(1/3)   (ne = electron number density)
    if ne is not None:
        rcu = ne.clamp(min=0.0) ** (1.0 / 3.0)
        chi1 = chi1 - 6.96e-7 * rcu
        chi2 = chi2 - 1.1048e-6 * rcu
    u12 = saha_ratio(theta, chi1, u0, u1, pe)
    u23 = saha_ratio(theta, chi2, u1, u2, pe)
    u33 = 1.0 + u12 * (1.0 + u23)
    eta0 = eta00 * 10.0 ** (-theta * energy_low) / (u0 * vdop * u33 * kappa5)
    if istage == 2:
        eta0 = eta0 * u1 / u2 * u12
    if stimulated:
        corre = 1.0 - torch.exp(-1.4388 / (t * wavelength_cm))
        eta0 = eta0 * corre
    return eta0


def damping_parameter(t, theta, pe, pp, wavelength_cm, vdop, vmic, weight,
                      alfa, sigma, zeff, istage, energy_low, chi1, chi2):
    """
    Damping parameter a of a line (Unsöld or Barklem/ABO).

    Parameters
    ----------
    t : torch.Tensor        Temperature [K]
    theta : torch.Tensor    5040/T
    pe : torch.Tensor       Electron pressure [dyn/cm^2]
    pp : dict               Partial pressures from ionization_equilibrium
    wavelength_cm : float   Vacuum wavelength [cm]
    vdop : torch.Tensor     Doppler velocity sqrt(2kT/m + vmic^2) [cm/s]
    vmic : torch.Tensor     Microturbulence [cm/s]
    weight : float          Atomic weight [g/mol]
    alfa, sigma : float     ABO damping parameters (0 -> Unsöld)
    zeff : float            multiplicative correction to gamma6 (Unsöld)
    istage : int            Ionization stage
    energy_low : float      Excitation potential [eV]
    chi1, chi2 : float      First/second ionization potentials [eV]

    Returns
    -------
    a : torch.Tensor
        Damping parameter (dimensionless).
    """
    weinv = 1.0 / weight
    croot = 1.66286e8 * weinv           # 2R/m for the Doppler width
    vdop2 = torch.sqrt(2.0 * GAS * t / weight + vmic ** 2)
    crad = 0.22233 / wavelength_cm      # natural (radiation) damping

    if alfa == 0.0 or sigma == 0.0:
        # --- Unsöld broadening ---
        chi1_eff = chi1
        if istage == 2:
            chi1_eff = chi2
        eupper = energy_low + 1.2398539e-4 / wavelength_cm
        ediff1 = max(chi1_eff - eupper - chi1 * (istage - 1), 1.0)
        ediff2 = max(chi1_eff - energy_low - chi1 * (istage - 1), 3.0)
        chydro = (wavelength_cm * 10.0 ** (0.4 * np.log10(1.0 / ediff1 ** 2
                  - 1.0 / ediff2 ** 2) - 12.213) * 5.34784e3)
        chydro = chydro * zeff
        if istage == 2:
            chydro = chydro * 1.741

        f1 = pp["h"]        # p(H)/p(H')
        f90 = pp["e-"]      # p(e-)/p(H')
        f91 = pp["ne"]      # n(e)
        f2 = pp["he"]       # p(He)/p(H')
        f89 = pp["h2"]      # p(H2)/p(H')

        aj = chydro * (f1 / f90 * f91) * t ** 0.3
        ai = ((0.992093 + weinv) ** 0.3
              + 0.6325 * f2 / f1 * (0.2498376 + weinv) ** 0.3
              + 0.48485 * f89 / f1 * (0.4960465 + weinv) ** 0.3)
        a = (aj * ai + crad) / (12.5663706 * vdop)
        return a

    # --- Barklem (ABO) ---
    uma = ATOMIC_MASS_UNIT
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
    return a
