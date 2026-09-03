# -*- coding: utf-8 -*-
"""
spot.inversion.marquardt — Levenberg-Marquardt solver
======================================================
Given the response functions of the model parameters, build the normal
equations

    alpha = J^T W J          (W = 1/sigma^2)
    beta  = J^T W (y_obs - y_mod)

and solve ``(alpha * (1 + lambda)) delta = beta`` with a damped SVD.
The parameter update is multiplicative for T/Pe/micro/B/vlos and
additive for gamma/phi/vmac, matching the node update modes used by the
rest of the inversion package.

This module is device-agnostic: it works on plain torch tensors.
"""

import torch

__all__ = ["marquardt_step", "solve_damped_svd", "chi_square"]


def chi_square(target, model, sigma):
    """
    chi^2 = sum ((target - model)/sigma)^2  over the flattened data.

    Parameters
    ----------
    target : torch.Tensor (Nb, Ndata)
    model : torch.Tensor (Nb, Ndata)
    sigma : torch.Tensor (Nb, Ndata) or (Ndata,) or float
        Noise level per data point.

    Returns
    -------
    torch.Tensor (Nb,)
    """
    sig = sigma if torch.is_tensor(sigma) else torch.as_tensor(sigma)
    if sig.dim() == 1 and sig.shape[0] == target.shape[1]:
        sig = sig[None, :]
    diff = (target - model) / sig
    return (diff * diff).sum(dim=1)


def solve_damped_svd(alpha, beta, lamda, tolerance=1e-4, groups=None):
    """
    Solve ``(alpha * (1 + lamda)) delta = beta`` with damped SVD.

    Parameters
    ----------
    alpha : torch.Tensor (Nb, Np, Np)
        Normal matrix.
    beta : torch.Tensor (Nb, Np)
        Right-hand side.
    lamda : float or torch.Tensor (Nb,)
        Marquardt damping.
    tolerance : float
        Relative singular-value cutoff.
    groups : array-like (Np,), optional
        Physical-quantity id of every parameter.  When given, the
        singular-value truncation is applied *per physical quantity*: a
        singular direction is kept for quantity g if its contribution
        ``w(j)*sum_i v(i,j)^2`` to that group is >= tol * (group
        maximum); directions are accumulated over the groups (two
        refinement passes, w = w1^2/w2).  This prevents the
        large-curvature quantities (T, vlos) from truncating the small
        ones (B, gamma, phi) globally.

    Returns
    -------
    delta : torch.Tensor (Nb, Np)
    """
    if not torch.is_tensor(lamda):
        lamda = torch.full((alpha.shape[0],), float(lamda),
                           device=alpha.device, dtype=alpha.dtype)
    # guard: replace non-finite normal matrices with the diagonal (a pure
    # gradient-descent step) so a single bad batch cannot crash the solve
    alpha = torch.where(torch.isfinite(alpha), alpha,
                        torch.diagonal(alpha, dim1=-2, dim2=-1)
                        .diag_embed().nan_to_num(nan=1.0, posinf=1e30, neginf=-1e30))
    # lambda damps the *diagonal* only
    damp = alpha + lamda[:, None, None] * torch.diagonal(
        alpha, dim1=-2, dim2=-1).diag_embed()
    u, s, vh = torch.linalg.svd(damp, full_matrices=False)   # (Nb,Np,Np)

    if groups is not None:
        groups = torch.as_tensor(groups, device=s.device, dtype=torch.long)
        ng = int(groups.max()) + 1
        v = vh.transpose(-2, -1)                    # (Nb,Np,Np): v[...,i,j]
        s2 = torch.zeros_like(s)
        for _ in range(2):                          # two passes
            ww = torch.zeros_like(s)
            for g in range(ng):
                mask = (groups == g)                # (Np,)
                if not mask.any():
                    continue
                # wt(j) = w(j) * sum_{i in group} v(i,j)^2
                # v[..., mask, :] is (Nb, Nm, Np); sum over the mask dim
                # gives (Nb, Np) — multiply by s (same shape), NOT by
                # s.unsqueeze(-2), which would broadcast (Nb,Np)x(Nb,1,Np)
                # to (Nb,Nb,Np) and break the batch layout (single-column
                # runs are hidden by broadcasting).
                wt = (v[..., mask, :] ** 2).sum(dim=-2) * s
                wtx = wt.abs().amax(dim=-1, keepdim=True).clamp(min=1e-300)
                wtn = wtx * tolerance
                ww = ww + torch.where(wt >= wtn, wt, torch.zeros_like(wt))
            s = ww
            if _ == 0:
                s1 = s.clone()
        # refinement: w = w1^2 / w2, 0 when w2 ~ 0
        s = torch.where(s.abs() > 1e-10, s1 * s1 / s, torch.zeros_like(s))
    else:
        smax = s[:, :1].clamp(min=1e-30)
        s = torch.where(s <= tolerance * smax, torch.zeros_like(s), s)

    s_inv = torch.where(s > 0, 1.0 / s, torch.zeros_like(s))
    # delta = V diag(1/s) U^T beta
    utb = torch.matmul(u.transpose(-2, -1), beta.unsqueeze(-1)).squeeze(-1)
    delta = torch.matmul(vh.transpose(-2, -1), (s_inv * utb).unsqueeze(-1)).squeeze(-1)
    return delta


def marquardt_step(params, delta, update_mode, max_step=None,
                   additive_factor=None):
    """
    Apply the Marquardt correction to the parameters.

    Parameters
    ----------
    params : torch.Tensor (Nb, Np)
        Current parameter values (node values).
    delta : torch.Tensor (Nb, Np)
        Correction from :func:`solve_damped_svd`.
    update_mode : list of str (Np,)
        'multiplicative' (a_new = a*(1+da)) or 'additive' (a_new = a+da).
    max_step : list of float (Np,) or None
        Cap on |da| (multiplicative) or |da|/|a| (additive); None -> no cap.
    additive_factor : torch.Tensor (Np,) or float or None
        Unit conversion applied to the additive update
        (a_new = a + da*factor); the angle channels carry the delta in
        RADIANS while the angle parameters are per-DEGREE, so the
        factor is 180/pi for gamma/phi and 1.0 elsewhere.
    """
    trial = params.clone()
    for j in range(params.shape[1]):
        mode = update_mode[j]
        dj = delta[:, j]
        if max_step is not None and max_step[j] is not None:
            cap = float(max_step[j])
            dj = dj.clamp(-cap, cap)
        if mode == "multiplicative":
            trial[:, j] = params[:, j] * (1.0 + dj)
        else:
            f = 1.0
            if additive_factor is not None:
                f = float(additive_factor[j]) if torch.is_tensor(
                    additive_factor) else float(additive_factor)
            trial[:, j] = params[:, j] + dj * f
    return trial
