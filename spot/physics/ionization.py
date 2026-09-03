# -*- coding: utf-8 -*-
"""
spot.physics.ionization — Saha equation and molecular balance
==============================================================
Spot's ionization module: the Saha ionization ratio and the molecular
balance (H2 and H2+) used by the thermodynamic and opacity routines.
"""

import torch

__all__ = ["saha_ratio", "molecular_balance"]


def saha_ratio(theta, chi, u_low, u_up, pe):
    """
    Saha ionization ratio n(i+1)/n(i).

    Parameters
    ----------
    theta : torch.Tensor
        ``5040 / T`` (eV/kT in units of log10).
    chi : float or torch.Tensor
        Ionization potential [eV].
    u_low, u_up : torch.Tensor
        Partition functions of the lower and upper ionization stages.
    pe : torch.Tensor
        Electron pressure [dyn/cm^2].

    Returns
    -------
    torch.Tensor
        ``n(i+1)/n(i) = 2 u_up/u_low * (2 pi m_e k T/h^2)^1.5 * 10^(-theta*chi)/Ne``
    """
    return (u_up / u_low) * torch.pow(
        10.0, 9.0805126 - theta * chi) / (pe * theta ** 2.5)


def molecular_balance(theta):
    """
    Molecular balance constants of H2 and H2+.

    Parameters
    ----------
    theta : torch.Tensor
        ``5040 / T``.

    Returns
    -------
    cmol : torch.Tensor of shape (2, *theta.shape)
        cmol[0] -> log10 constant for H2+, cmol[1] -> log10 constant for H2.
    """
    cmol = torch.stack([
        (-11.206998 + theta * (2.7942767 + theta * (7.9196803e-2
         - theta * 2.4790744e-2))),   # H2+
        (-12.533505 + theta * (4.9251644 + theta * (-5.6191273e-2
         + theta * 3.2687661e-3))),   # H2
    ])
    return cmol


def molecular_balance_derivatives(theta):
    """
    d ln(cm)/dT of the molecular balance constants.

    dx = -theta^2/5040 (= d theta/dT), dy(i) = dx * dY/dtheta.
    """
    dx = -theta * theta / 5040.0
    dy = torch.stack([
        dx * (2.7942767 + theta * (2.0 * 7.9196803e-2
              - theta * 3.0 * 2.4790744e-2)),   # H2+
        dx * (4.9251644 + theta * (-2.0 * 5.6191273e-2
              - theta * 3.0 * 3.2687661e-3)),   # H2
    ])
    return dy
