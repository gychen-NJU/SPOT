# -*- coding: utf-8 -*-
"""
spot.physics.absorption — the 4x4 propagation (absorption) matrix
=================================================================
Spot's absorption module: given the line/continuum absorption and
dispersion profiles and the field angles, build the propagation matrix
K used in the polarized radiative-transfer equation (RTE).

Matrix layout (row-major 4x4):

    [ fi    fq    fu    fv  ]
    [ fq    fi   -fv1   fu1 ]
    [ fu    fv1   fi   -fq1 ]
    [ fv   -fu1   fq1   fi  ]

with fi (total absorption), fq/fu/fv (absorption of Q/U/V) and
fq1/fu1/fv1 (magneto-optical dispersion terms).
"""

import torch

__all__ = ["absorption_matrix", "profile_components"]


def profile_components(eta_pi, eta_r, eta_l, rho_pi, rho_r, rho_l,
                       gamma, phi):
    """
    Combine pi/r/l line profiles with the field geometry.

    Parameters
    ----------
    eta_pi, eta_r, eta_l : torch.Tensor
        Absorption profiles of the pi, sigma-r, sigma-l components.
    rho_pi, rho_r, rho_l : torch.Tensor
        Dispersion (Faraday) profiles of the components.
    gamma : torch.Tensor
        Magnetic inclination [rad].
    phi : torch.Tensor
        Magnetic azimuth [rad].

    Returns
    -------
    fi, fq, fu, fv, fq1, fu1, fv1 : torch.Tensor
        The 7 matrix elements (before multiplying by the line opacity).
    """
    tm = 0.5 * (eta_r + eta_l)
    tn = 0.5 * (eta_r - eta_l)
    sm = 0.5 * (rho_r + rho_l)
    sn = 0.5 * (rho_r - rho_l)
    tpm = 0.5 * (eta_pi - tm)
    spm = 0.5 * (rho_pi - sm)

    sg = torch.sin(gamma)
    cg = torch.cos(gamma)
    s2g = sg * sg
    c2g = 1.0 + cg * cg
    sf = torch.sin(2.0 * phi)
    cf = torch.cos(2.0 * phi)

    fi = 0.5 * (eta_pi * s2g + tm * c2g)
    fq = tpm * s2g * cf
    fu = tpm * s2g * sf
    fv = tn * cg
    fq1 = spm * s2g * cf
    fu1 = spm * s2g * sf
    fv1 = sn * cg
    return fi, fq, fu, fv, fq1, fu1, fv1


def absorption_matrix(t0, t3, eta_pi, eta_r, eta_l, rho_pi, rho_r, rho_l,
                      gamma, phi):
    """
    Full propagation matrix K.

    Parameters
    ----------
    t0 : torch.Tensor
        Continuum opacity ratio kappaC(lambda)/kappaC(5000 A).
    t3 : torch.Tensor
        Line opacity ratio kappaL/kappaC(5000 A).
    eta_pi, eta_r, eta_l, rho_pi, rho_r, rho_l : torch.Tensor
        Profiles (same shape).
    gamma, phi : torch.Tensor
        Field angles [rad].

    Returns
    -------
    torch.Tensor of shape (..., 4, 4)
    """
    fi, fq, fu, fv, fq1, fu1, fv1 = profile_components(
        eta_pi, eta_r, eta_l, rho_pi, rho_r, rho_l, gamma, phi)
    fi = t0 + t3 * fi
    fq = t3 * fq
    fu = t3 * fu
    fv = t3 * fv
    # NOTE on signs: the Faraday (dispersion) profiles carry an implicit
    # minus sign relative to the textbook convention, so the
    # off-diagonal rho elements enter the propagation matrix with the
    # opposite sign (validated against the emergent Stokes U).
    fq1 = -t3 * fq1
    fu1 = -t3 * fu1
    fv1 = -t3 * fv1

    zero = torch.zeros_like(fi)
    one = torch.ones_like(fi)
    k = torch.stack([
        torch.stack([fi, fq, fu, fv], dim=-1),
        torch.stack([fq, fi, -fv1, fu1], dim=-1),
        torch.stack([fu, fv1, fi, -fq1], dim=-1),
        torch.stack([fv, -fu1, fq1, fi], dim=-1),
    ], dim=-2)
    return k


def identity_absorption(shape, device, dtype):
    """Absorption matrix of a pure continuum (used by RTE tests)."""
    fi = torch.ones(shape, device=device, dtype=dtype)
    zero = torch.zeros(shape, device=device, dtype=dtype)
    return torch.stack([
        torch.stack([fi, zero, zero, zero], dim=-1),
        torch.stack([zero, fi, zero, zero], dim=-1),
        torch.stack([zero, zero, fi, zero], dim=-1),
        torch.stack([zero, zero, zero, fi], dim=-1),
    ], dim=-2)
