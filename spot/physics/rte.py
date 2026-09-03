# -*- coding: utf-8 -*-
"""
spot.physics.rte — formal solutions of the polarized RTE
===========================================================
Formal solvers of

    dI/dtau = K (I - S)

along the line of sight.  All solvers share the spot convention: the
atmosphere is stored with *decreasing* log10(tau) (index 0 = deepest
layer, last index = the top), matching the ordering of the reference
native implementation's internal arrays.  The emergent Stokes vector at
the top (tau -> 0) is returned.

* :func:`hermite_solve` — Hermitian method, 4th order, the default
  (Bellot Rubio, Ruiz Cobo & Collados 1998);
* :func:`hermite_solve_general` — Hermitian solution of the linearized
  (variational) RTE used by the response functions;
* :func:`propagation_operator` — propagation (Green) operator OO of the
  linearized RTE;
* :func:`delo_solve` — DELO (Rees, Durrant & Murphy 1989), 2nd order,
  ported from pyPRT-dsh ``pyprt/phys/rte.py::delo_solve`` and adapted to
  the spot (decreasing-ltau) convention;
* :func:`cn_solve` — A-stable Crank-Nicolson (trapezoidal) step in
  linear tau space, ported from pyPRT-dsh ``pyprt/phys/rte.py::rk4_solve``
  (pyPRT names the method "RK4" and its README advertises
  "Crank-Nicolson + deferred correction", but the reference
  implementation performs the pure Crank-Nicolson base step; this port
  reproduces the reference implementation's numerical behaviour and is
  named ``cn_solve`` for what it actually is; 2nd order).

All solvers use the same deep boundary condition: the diffusion
approximation ``I ~ S + K'^{-1} dS/d(log10 tau)`` below the control
depth ``icontorno``, the shallowest layer whose per-step optical
thickness ``max|K(1,1..4,i)| * deltae(i)`` exceeds the control value
(``boundary_control``).
"""

import torch

__all__ = ["hermite_solve", "hermite_solve_general", "propagation_operator",
           "delo_solve", "cn_solve", "resolve_rte_solver"]

_LOG10 = 2.3025851


def _tau_grid(ltau):
    """Linear tau = 10**ltau on the spot grid (decreasing, index 0 = deepest)."""
    return torch.pow(10.0, ltau)


def _diffusion_reference(ltau, k_matrix, source):
    """Depth-by-depth diffusion-approximation reference solution.

    ``I_ref = S + K'^{-1} dS/d(log10 tau)`` with ``K' = K ln(10) tau``
    (the RTE solution *below* the control depth / at the deep
    boundary).  Shared by every formal solver
    (hermitian, delo, cn) so that the deep boundary condition is
    identical across methods.
    """
    nb, nt, nw = k_matrix.shape[:3]
    device, dtype = k_matrix.device, k_matrix.dtype
    ltau = ltau.to(device=device, dtype=dtype)
    tau = _tau_grid(ltau)
    kp = k_matrix * (_LOG10 * tau[None, :, None, None, None])
    dsource = _quadratic_derivative(source, ltau, dim=1)
    try:
        kinv = torch.linalg.inv(kp)
    except RuntimeError:
        kinv = torch.linalg.pinv(kp)
    return torch.matmul(kinv, dsource.unsqueeze(-1)).squeeze(-1) + source


def _icontorno_index(ltau, k_matrix, boundary_control):
    """(Nb, Nw) index of the shallowest layer whose per-step optical
    thickness exceeds ``boundary_control`` (layers deeper than the
    control depth use the diffusion approximation).

    Exactly the rule inside :func:`hermite_solve`, shared so that all
    solvers use the same deep-boundary convention.
    """
    nb, nt, nw = k_matrix.shape[:3]
    device, dtype = k_matrix.device, k_matrix.dtype
    ltau = ltau.to(device=device, dtype=dtype)
    tau = _tau_grid(ltau)
    deltae = tau[:-1] - tau[1:]               # (Nt-1,), > 0
    taunu = k_matrix[..., 0, :].abs().amax(dim=-1)   # (Nb, Nt, Nw) first row
    over = (taunu[:, :-1] * deltae[None, :, None]) > boundary_control
    has_over = over.any(dim=1)                # (Nb, Nw)
    last_idx = (over.shape[1] - 1) - over.flip(1).to(torch.int64).argmax(dim=1)
    return torch.where(has_over, last_idx, torch.zeros_like(last_idx)
                       ).clamp(max=nt - 2)


def _quadratic_derivative(y, x, dim=-1):
    """Three-point (quadratic) derivative along ``dim``: one-sided
    two-point differences at the boundaries, a three-point formula
    interpolating a quadratic through the interior points.

    ``y`` has shape (..., N, ...) and ``x`` is a 1-D grid of length N
    (broadcast along ``dim``).
    """
    n = y.shape[dim]
    x = x.to(dtype=y.dtype, device=y.device).reshape(
        [1 if i != dim else n for i in range(y.dim())])

    def take(a, idx):
        return torch.index_select(a, dim, torch.as_tensor(idx, device=a.device))

    ds = torch.empty_like(y)
    # index 0: one-sided two-point difference (no neighbor on that side)
    ds = torch.index_copy(
        ds, dim, torch.as_tensor([0], device=y.device),
        (take(y, [1]) - take(y, [0])) / (take(x, [1]) - take(x, [0])))
    ds = torch.index_copy(
        ds, dim, torch.as_tensor([n - 1], device=y.device),
        (take(y, [n - 1]) - take(y, [n - 2])) / (take(x, [n - 1]) - take(x, [n - 2])))
    x1 = take(x, list(range(0, n - 2)))
    x2 = take(x, list(range(1, n - 1)))
    x3 = take(x, list(range(2, n)))
    y1 = take(y, list(range(0, n - 2)))
    y2 = take(y, list(range(1, n - 1)))
    y3 = take(y, list(range(2, n)))
    d1 = 1.0 / (x2 - x3)
    d2 = 1.0 / (x1 - x2)
    d3 = 1.0 / (x1 - x3)
    mid = y3 * (d3 - d1) + y2 * (d1 - d2) + y1 * (d2 - d3)
    ds = torch.index_copy(
        ds, dim, torch.as_tensor(list(range(1, n - 1)), device=y.device), mid)
    return ds


def hermite_solve(ltau, k_matrix, source, boundary_control=5.0,
                  return_layers=False):
    """
    Solve the polarized RTE with the Hermitian method.

    Parameters
    ----------
    ltau : torch.Tensor (Nt,)
        log10(tau5000), *decreasing* from deep to shallow (index 0 is
        the deepest layer) — the spot grid convention.
    k_matrix : torch.Tensor (Nb, Nt, Nw, 4, 4)
        Propagation matrix K per unit tau5000 (already including the
        continuum normalization kappaC/kappa5).
    source : torch.Tensor (Nb, Nt, Nw, 4)
        Source function S (usually (B, 0, 0, 0)).
    boundary_control : float
        Layer optical-depth threshold for the diffusion approximation
        (default 5).
    return_layers : bool, optional
        If True, also return the solution at every layer, shape
        (Nb, Nt, Nw, 4).

    Returns
    -------
    torch.Tensor (Nb, Nw, 4) or (out, layers)
        Emergent Stokes vector at the top of the atmosphere (and the
        per-layer solution if ``return_layers``).
    """
    nb, nt, nw = k_matrix.shape[:3]
    device = k_matrix.device
    dtype = k_matrix.dtype
    ltau = ltau.to(device=device, dtype=dtype)
    tau = torch.pow(10.0, ltau)                 # (Nt,) deep -> shallow
    taue = tau

    # layer steps in linear tau and half-steps in log10 tau
    deltae = taue[:-1] - taue[1:]               # (Nt-1,), > 0
    deltai = (ltau[1:] - ltau[:-1]) / 2.0       # (Nt-1,), < 0 (going up)
    delt2i = deltai ** 2 / 3.0

    # K in units of d/d(log10 tau): K' = K * ln10 * tau
    kp = k_matrix * (_LOG10 * taue[None, :, None, None, None])

    # find the boundary (icontorno): the shallowest layer (largest index
    # in the deep->shallow array) whose per-step optical thickness
    # max(|K(1,1..4,i)|) * deltae(i) exceeds the control value.  Layers
    # deeper than icontorno are optically thick, so they are treated with
    # the diffusion approximation instead of an RTE step.
    taunu = k_matrix[..., 0, :].abs().amax(dim=-1)   # (Nb, Nt, Nw) first row
    over = (taunu[:, :-1] * deltae[None, :, None]) > boundary_control  # (Nb, Nt-1, Nw)
    has_over = over.any(dim=1)                  # (Nb, Nw)
    last_idx = (over.shape[1] - 1) - over.flip(1).to(torch.int64).argmax(dim=1)  # last True
    icontorno = torch.where(has_over, last_idx,
                            torch.zeros_like(last_idx)).clamp(max=nt - 2)

    # derivatives of K' and S with respect to log10 tau
    dkp = _quadratic_derivative(kp, ltau, dim=1)
    dsource = _quadratic_derivative(source, ltau, dim=1)
    kp2 = torch.matmul(kp, kp)                  # K'^2

    eye = torch.eye(4, device=device, dtype=dtype).view(1, 1, 1, 4, 4)

    # diffusion approximation: I = S + K'^-1 dS/d(log10 tau)
    # (only the column of K'^-1 that multiplies dS is needed)
    try:
        kinv = torch.linalg.inv(kp)
    except RuntimeError:
        kinv = torch.linalg.pinv(kp)
    iref = torch.matmul(kinv, dsource.unsqueeze(-1)).squeeze(-1) + source

    # --- precompute every sol-independent term once, vectorized over all
    # layers (the Hermite step itself is sequential, but all matrix
    # combinations can be formed in one shot):
    #   den  = den0 + sol + den1 @ sol
    #   num  = -(kj*delta) + delt2*(dkj + k2j) + eye
    kp_l, kp_r = kp[:, :-1], kp[:, 1:]          # (Nb, Nt-1, Nw, 4, 4)
    dk_l, dk_r = dkp[:, :-1], dkp[:, 1:]
    k2_l, k2_r = kp2[:, :-1], kp2[:, 1:]
    s_l, s_r = source[:, :-1], source[:, 1:]    # (Nb, Nt-1, Nw, 4)
    ds_l, ds_r = dsource[:, :-1], dsource[:, 1:]
    delta = deltai[None, :, None, None]         # (1, Nt-1, 1, 1)
    delt2 = delt2i[None, :, None, None]
    delta5 = deltai[None, :, None, None, None]  # (1, Nt-1, 1, 1, 1)
    delt25 = delt2i[None, :, None, None, None]

    def mm(a, b):
        """(...,4,4) @ (...,4) -> (...,4)"""
        return (a @ b.unsqueeze(-1)).squeeze(-1)

    den0 = -(mm(kp_l, s_l) + mm(kp_r, s_r)) * delta \
        - (mm(dk_l, s_l) - mm(dk_r, s_r) + mm(kp_l, ds_l) - mm(kp_r, ds_r)
           + mm(k2_l, s_l) - mm(k2_r, s_r)) * delt2
    den1 = kp_l * delta5 + delt25 * (dk_l + k2_l)
    num = -(kp_r * delta5) + delt25 * (dk_r + k2_r) + eye

    # initial solution at the deep boundary (index 0)
    sol = iref[:, 0].clone()                    # (Nb, Nw, 4)
    layers = [sol.clone()]

    # integrate upwards from i=1 (deep) to i=nt-1 (top)
    for i in range(1, nt):
        # denominator vector of the Hermite step
        den = den0[:, i - 1] + sol + mm(den1[:, i - 1], sol)

        # numerator matrix of the Hermite step: off-diagonal entries
        # num(ii,jj) = -K'(ii,jj,i)*delta + delt2*(dK'(ii,jj,i)+K'2(ii,jj,i)),
        # with the diagonal incremented by 1 (num(jj,jj) = 1 + num(jj,jj))
        try:
            sol = torch.linalg.solve(num[:, i - 1], den.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            sol = (torch.linalg.pinv(num[:, i - 1]) @ den.unsqueeze(-1)).squeeze(-1)

        # below the control depth, keep the diffusion approximation
        use_diff = (i - 1) < icontorno           # (Nb, Nw)
        sol = torch.where(use_diff[..., None], iref[:, i], sol)
        layers.append(sol.clone())

    if return_layers:
        return sol, torch.stack(layers, dim=1)
    return sol


def propagation_operator(ltau, k_matrix, boundary_control=5.0):
    """
    Propagation (Green) operator OO of the linearized RTE.

    OO[:, i] is the 4x4 matrix that transports a source perturbation at
    depth i to the top of the atmosphere: the response function of the
    quantity x is ``RF(x) = sum_i OO[:, i] @ Q(x, i)`` with the local
    source ``Q(x, i) = (dK/dx)(I-S) - dS/dx`` evaluated at depth i.
    Each layer-step operator OO(i-1) = num^-1 * den2 transports the
    linearized solution one layer upward; the operators are accumulated
    from the top, and the response function follows from OO acting on
    the local source perturbation.

    OO depends only on K, so it is computed once and shared by every
    physical quantity.

    Parameters
    ----------
    ltau : torch.Tensor (Nt,)
        log10(tau5000), decreasing (index 0 = deepest).
    k_matrix : torch.Tensor (Nb, Nt, Nw, 4, 4)
        Propagation matrix K (same as in :func:`hermite_solve`).
    boundary_control : float
        Same as in :func:`hermite_solve`.

    Returns
    -------
    torch.Tensor (Nb, Nt, Nw, 4, 4)
        OO[:, i, :, :, :] transports a perturbation at depth i to the
        top; layers below the control depth have OO = 0 (they do not
        reach the surface).
    """
    nb, nt, nw = k_matrix.shape[:3]
    device, dtype = k_matrix.device, k_matrix.dtype
    ltau = ltau.to(device=device, dtype=dtype)
    tau = torch.pow(10.0, ltau)

    deltae = tau[:-1] - tau[1:]
    deltai = (ltau[1:] - ltau[:-1]) / 2.0
    delt2i = deltai ** 2 / 3.0

    kp = k_matrix * (_LOG10 * tau[None, :, None, None, None])

    taunu = k_matrix[..., 0, :].abs().amax(dim=-1)
    over = (taunu[:, :-1] * deltae[None, :, None]) > boundary_control
    has_over = over.any(dim=1)
    last_idx = (over.shape[1] - 1) - over.flip(1).to(torch.int64).argmax(dim=1)
    icontorno = torch.where(has_over, last_idx,
                            torch.zeros_like(last_idx)).clamp(max=nt - 2)

    dkp = _quadratic_derivative(kp, ltau, dim=1)
    kp2 = torch.matmul(kp, kp)

    eye = torch.eye(4, device=device, dtype=dtype).view(1, 1, 1, 4, 4)
    delta5 = deltai[None, :, None, None, None]
    delt25 = delt2i[None, :, None, None, None]

    # per-layer-step propagation matrices (i-1 -> i)
    kp_l, kp_r = kp[:, :-1], kp[:, 1:]
    dk_l, dk_r = dkp[:, :-1], dkp[:, 1:]
    k2_l, k2_r = kp2[:, :-1], kp2[:, 1:]
    # num = -(K_j*delta) + delt2*(dK_j + K_j^2) + I
    num = -(kp_r * delta5) + delt25 * (dk_r + k2_r) + eye
    # den2 = (K_i*delta + delt2*(dK_i + K_i^2)) + I
    den2 = kp_l * delta5 + delt25 * (dk_l + k2_l) + eye
    step = torch.linalg.solve(num, den2)             # (Nb, Nt-1, Nw, 4, 4)

    # accumulate from the top: OO(i) = OO(i+1) @ step(i).  The matrix at
    # a deeper layer is that one layer above composed with the layer-step
    # matrix, so the accumulated operator carries a perturbation at depth
    # i all the way to the surface.
    oo = torch.zeros((nb, nt, nw, 4, 4), device=device, dtype=dtype)
    oo[:, -1] = torch.eye(4, device=device, dtype=dtype)
    for i in range(nt - 2, -1, -1):
        oo[:, i] = oo[:, i + 1] @ step[:, i]
    # layers below the control depth do not reach the surface
    idx = torch.arange(nt, device=device).view(1, -1, 1)
    mask = (idx < icontorno.unsqueeze(1)).unsqueeze(-1).unsqueeze(-1)  # (Nb,Nt,Nw,1,1)
    oo = torch.where(mask, torch.zeros_like(oo), oo)
    return oo


def hermite_solve_general(ltau, k_matrix, q_source, boundary_control=5.0):
    """
    Hermitian solution of the *linearized* RTE with a general source:

        dY/dtau = K Y + Q

    This is the variational equation satisfied by the response
    functions: with ``Q = (dK/dx)(I - S) - dS/dx``, the solution ``Y``
    is ``dI/dx`` (the variational equation solved for the response
    functions).  The propagation matrix
    K is shared with the forward solve, so the expensive parts
    (``num``, the layer matrices and the icontorno) are computed once
    and reused for every source.

    Parameters
    ----------
    ltau : torch.Tensor (Nt,)
        log10(tau5000), decreasing (index 0 = deepest).
    k_matrix : torch.Tensor (Nb, Nt, Nw, 4, 4)
        Propagation matrix K (same as in :func:`hermite_solve`).
    q_source : torch.Tensor (Nb, Np, Nt, Nw, 4) or (Nb, Nt, Nw, 4)
        Source term Q for each of the Np parameters, in *per-dlogtau*
        units (i.e. multiplied by ln10*tau, matching K' = K ln10 tau).
        The Np dimension is optional (a single source may be given
        without it).
    boundary_control : float
        Same as in :func:`hermite_solve`.

    Returns
    -------
    torch.Tensor (Nb, Np, Nw, 4) or (Nb, Nw, 4)
        Solution at the top of the atmosphere for each source.
    """
    has_p = q_source.dim() == 5
    if has_p:
        nb, np_, nt, nw = q_source.shape[:4]
    else:
        nb, nt, nw = q_source.shape[:3]
    device, dtype = k_matrix.device, k_matrix.dtype
    ltau = ltau.to(device=device, dtype=dtype)
    tau = torch.pow(10.0, ltau)

    deltae = tau[:-1] - tau[1:]
    deltai = (ltau[1:] - ltau[:-1]) / 2.0
    delt2i = deltai ** 2 / 3.0

    kp = k_matrix * (_LOG10 * tau[None, :, None, None, None])

    # icontorno from the first row of K (same rule as hermite_solve)
    taunu = k_matrix[..., 0, :].abs().amax(dim=-1)   # (Nb, Nt, Nw)
    over = (taunu[:, :-1] * deltae[None, :, None]) > boundary_control
    has_over = over.any(dim=1)
    last_idx = (over.shape[1] - 1) - over.flip(1).to(torch.int64).argmax(dim=1)
    icontorno = torch.where(has_over, last_idx,
                            torch.zeros_like(last_idx)).clamp(max=nt - 2)

    dkp = _quadratic_derivative(kp, ltau, dim=1)
    kp2 = torch.matmul(kp, kp)

    if has_p:
        q = q_source.reshape(nb * np_, nt, nw, 4)
        kp_p = kp.unsqueeze(1).expand(nb, np_, nt, nw, 4, 4).reshape(
            nb * np_, nt, nw, 4, 4)
        dkp_p = dkp.unsqueeze(1).expand(nb, np_, nt, nw, 4, 4).reshape(
            nb * np_, nt, nw, 4, 4)
        kp2_p = kp2.unsqueeze(1).expand(nb, np_, nt, nw, 4, 4).reshape(
            nb * np_, nt, nw, 4, 4)
        icontorno_p = icontorno.unsqueeze(1).expand(nb, np_, nw).reshape(
            nb * np_, nw)
        kp_use, dkp_use, kp2_use, icontorno_use = kp_p, dkp_p, kp2_p, icontorno_p
    else:
        q = q_source
        kp_use, dkp_use, kp2_use, icontorno_use = kp, dkp, kp2, icontorno

    eye = torch.eye(4, device=device, dtype=dtype).view(1, 1, 1, 4, 4)

    # precompute layer combinations (shared by every source)
    kp_l, kp_r = kp_use[:, :-1], kp_use[:, 1:]
    dk_l, dk_r = dkp_use[:, :-1], dkp_use[:, 1:]
    k2_l, k2_r = kp2_use[:, :-1], kp2_use[:, 1:]
    q_l, q_r = q[:, :-1], q[:, 1:]
    delta = deltai[None, :, None, None]
    delt2 = delt2i[None, :, None, None]
    delta5 = deltai[None, :, None, None, None]
    delt25 = delt2i[None, :, None, None, None]

    def mm(a, b):
        return (a @ b.unsqueeze(-1)).squeeze(-1)

    # den = (Q_i + Q_j)*delta + sol
    #       + (K_i@Q_i - K_j@Q_j)*delt2
    #       + (K_i*delta + delt2*(dK_i + K_i^2)) @ sol
    # (the dQ/dlogtau term is dropped: the variational equation is solved
    # with source terms concentrated at each layer -- a three-point
    # derivative of a delta-like source would leak into neighbouring
    # layers and break the icontorno boundary, so only OO acting on the
    # local source term is used)
    den0 = (q_l + q_r) * delta + (mm(kp_l, q_l) - mm(kp_r, q_r)) * delt2
    den1 = kp_l * delta5 + delt25 * (dk_l + k2_l)
    num = -(kp_r * delta5) + delt25 * (dk_r + k2_r) + eye

    # boundary condition of the variational equation: perturbations below
    # the control depth (icontorno) do not reach the surface (the layers
    # are optically thick; the propagation operator OO is zero there) so
    # the solution is forced to zero below icontorno and integrated upward
    # from the boundary value 0 at the deepest layer.
    sol = torch.zeros_like(q[:, 0])                 # (Nb', Nw, 4)

    for i in range(1, nt):
        den = den0[:, i - 1] + sol + mm(den1[:, i - 1], sol)
        sol = torch.linalg.solve(num[:, i - 1], den.unsqueeze(-1)).squeeze(-1)
        # below the control depth, keep the zero boundary condition
        use_diff = (i - 1) < icontorno_use
        sol = torch.where(use_diff[..., None], torch.zeros_like(sol), sol)

    if has_p:
        return sol.reshape(nb, np_, nw, 4)
    return sol


def _delo_aux(dtau):
    """DELO auxiliary coefficients E, F, G (Rees et al. 1989).

    ``dtau`` = positive layer optical thickness (deep -> shallow).  The
    G coefficient is evaluated with a series expansion for very thin
    layers (its naive formulation loses precision as dtau -> 0).
    """
    e = torch.exp(-dtau)
    f = 1.0 - e
    # G = (1 - (1+dtau) e^-dtau) / dtau  (~ dtau/2 for small dtau)
    thin = dtau < 1e-6
    g_small = dtau * (0.5 - dtau / 3.0)          # series: dtau/2 - dtau^2/3
    g = torch.where(thin, g_small, (1.0 - (1.0 + dtau) * e) / dtau)
    return e, f, g


def delo_solve(ltau, k_matrix, source, boundary_control=5.0,
               return_layers=False):
    """
    DELO formal solution (Rees, Durrant & Murphy 1989, ApJ 339, 1093).

    Port of pyPRT-dsh ``pyprt/phys/rte.py::delo_solve`` adapted to the
    spot convention.  Within each layer K and S are assumed to vary
    linearly; with

        E_k = exp(-dtau_k),  F_k = 1 - E_k,
        G_k = (1 - (1+dtau_k) E_k) / dtau_k,   K^p = K - I,

    the deep-to-shallow recursion is

        I_k = P_k + L_k I_{k+1}
        P_k = [I + (F_k - G_k) K^p_k]^{-1}
              [(F_k - G_k) K_k S_k + G_k K_{k+1} S_{k+1}]
        L_k = [I + (F_k - G_k) K^p_k]^{-1}[E_k I - G_k K^p_{k+1}]

    with the pair ``k`` (deep) -> ``k+1`` (shallow) — pyPRT indexes the
    pair by the shallower layer and runs its ltau increasing; the
    recursion is algebraically identical, only the grid direction
    differs.

    Parameters
    ----------
    Same as :func:`hermite_solve` (``ltau`` decreasing, index 0 =
    deepest; returned solution at the top).

    Returns
    -------
    torch.Tensor (Nb, Nw, 4) or (out, layers)
    """
    nb, nt, nw = k_matrix.shape[:3]
    device, dtype = k_matrix.device, k_matrix.dtype
    ltau = ltau.to(device=device, dtype=dtype)
    tau = _tau_grid(ltau)                       # decreasing

    dt = tau[:-1] - tau[1:]                     # (Nt-1,), > 0 deep->shallow
    e, f, g = _delo_aux(dt)
    e = e[None, :, None, None, None]            # (1, Nt-1, 1, 1, 1)
    f = f[None, :, None, None, None]
    g = g[None, :, None, None, None]

    eye = torch.eye(4, device=device, dtype=dtype).view(1, 1, 1, 4, 4)
    kp = k_matrix - eye                         # K^p = K - I
    kp_l, kp_r = kp[:, :-1], kp[:, 1:]          # deep, shallow
    sp_l = torch.matmul(k_matrix[:, :-1], source[:, :-1].unsqueeze(-1))
    sp_r = torch.matmul(k_matrix[:, 1:], source[:, 1:].unsqueeze(-1))
    # NOTE: the P source term uses K S (not K^p S) — matches pyPRT.

    try:
        # per-pair matrices shared by all columns/batches
        invt = torch.linalg.inv(eye + (f - g) * kp_r)   # (Nb, Nt-1, Nw, 4, 4)
    except RuntimeError:
        invt = torch.linalg.pinv(eye + (f - g) * kp_r)
    p = torch.matmul(invt, (f - g) * sp_r + g * sp_l)          # (Nb, Nt-1, Nw, 4, 1)
    lft = torch.matmul(invt, e * eye - g * kp_l)               # (Nb, Nt-1, Nw, 4, 4)

    # diffusion-approximation reference (same boundary as hermite_solve)
    iref = _diffusion_reference(ltau, k_matrix, source)
    icontorno = _icontorno_index(ltau, k_matrix, boundary_control)

    sol = iref[:, 0].unsqueeze(-1)              # (Nb, Nw, 4, 1) deepest
    layers = [sol.clone()]
    for i in range(1, nt):
        # pair (i-1 deep, i shallow): coefficients indexed by i-1
        sol = p[:, i - 1] + torch.matmul(lft[:, i - 1], sol)
        use_diff = (i - 1) < icontorno           # (Nb, Nw)
        sol = torch.where(use_diff[..., None, None],
                          iref[:, i].unsqueeze(-1), sol)
        layers.append(sol.clone())

    if return_layers:
        return sol.squeeze(-1), torch.stack(layers, dim=1).squeeze(-1)
    return sol.squeeze(-1)


def cn_solve(ltau, k_matrix, source, boundary_control=5.0,
             return_layers=False):
    """
    A-stable Crank-Nicolson (trapezoidal) formal solution in linear tau
    space, ported from pyPRT-dsh ``pyprt/phys/rte.py::rk4_solve``
    (pyPRT calls this method "RK4"; its README describes the method as
    "Crank-Nicolson + deferred correction", while its reference
    implementation — reproduced here — carries out the pure
    Crank-Nicolson base step; the port is named ``cn_solve``, after the
    algorithm actually implemented):

        I_{k+1} = [I - 0.5 dtau K_{k+1}]^{-1}
                  {[I + 0.5 dtau K_k] I_k
                   - 0.5 dtau (K_k S_k + K_{k+1} S_{k+1})}

    with ``dtau = tau_k - tau_{k+1}`` (shallow minus deep; negative).
    The bottom boundary is the same diffusion-approximation reference as
    :func:`hermite_solve` and the control-depth handling matches the
    other solvers.

    Parameters
    ----------
    Same as :func:`hermite_solve` (``ltau`` decreasing, index 0 =
    deepest; returned solution at the top).

    Returns
    -------
    torch.Tensor (Nb, Nw, 4) or (out, layers)
    """
    nb, nt, nw = k_matrix.shape[:3]
    device, dtype = k_matrix.device, k_matrix.dtype
    ltau = ltau.to(device=device, dtype=dtype)
    tau = _tau_grid(ltau)                       # decreasing

    dtau = tau[1:] - tau[:-1]                   # (Nt-1,), < 0 shallow-deep
    dtau = dtau[None, :, None, None, None]
    eye = torch.eye(4, device=device, dtype=dtype).view(1, 1, 1, 4, 4)

    k_l, k_r = k_matrix[:, :-1], k_matrix[:, 1:]     # deep, shallow
    s_l, s_r = source[:, :-1], source[:, 1:]
    a = eye - 0.5 * dtau * k_r                  # (Nb, Nt-1, Nw, 4, 4)
    b0 = (-0.5 * dtau * (torch.matmul(k_l, s_l.unsqueeze(-1))
                         + torch.matmul(k_r, s_r.unsqueeze(-1)))
          ).squeeze(-1)                         # (Nb, Nt-1, Nw, 4)
    m_l = eye + 0.5 * dtau * k_l                # (Nb, Nt-1, Nw, 4, 4)

    iref = _diffusion_reference(ltau, k_matrix, source)
    icontorno = _icontorno_index(ltau, k_matrix, boundary_control)

    sol = iref[:, 0].clone()                    # (Nb, Nw, 4) deepest
    layers = [sol.clone()]
    for i in range(1, nt):
        b = b0[:, i - 1] + torch.matmul(m_l[:, i - 1],
                                        sol.unsqueeze(-1)).squeeze(-1)
        try:
            sol = torch.linalg.solve(a[:, i - 1],
                                     b.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            sol = (torch.linalg.pinv(a[:, i - 1]) @ b.unsqueeze(-1)).squeeze(-1)
        use_diff = (i - 1) < icontorno           # (Nb, Nw)
        sol = torch.where(use_diff[..., None], iref[:, i], sol)
        layers.append(sol.clone())

    if return_layers:
        return sol, torch.stack(layers, dim=1)
    return sol


def rk4_solve(ltau, k_matrix, source, nsub=8):
    """
    Plain RK4 solution of the RTE on a refined grid (reference solver,
    used by tests to cross-check the Hermitian method).

    Parameters are the same as :func:`hermite_solve`.
    """
    nb, nt, nw = k_matrix.shape[:3]
    device, dtype = k_matrix.device, k_matrix.dtype
    ltau = ltau.to(device=device, dtype=dtype)
    tau = torch.pow(10.0, ltau)

    # refine each layer into nsub sub-layers in linear tau
    taus = []
    for i in range(nt - 1):
        taus.append(torch.linspace(tau[i], tau[i + 1], nsub + 1, device=device, dtype=dtype)[:-1])
    taus.append(tau[-1:])
    tau_r = torch.cat(taus)
    nt_r = tau_r.shape[0]

    def interp(x):
        idx = torch.searchsorted(tau, x).clamp(1, nt - 1)
        w = (x - tau[idx - 1]) / (tau[idx] - tau[idx - 1])
        lo = idx - 1
        hi = idx
        kk = k_matrix[:, lo] * (1 - w)[None, :, None, None, None] + k_matrix[:, hi] * w[None, :, None, None, None]
        ss = source[:, lo] * (1 - w)[None, :, None, None] + source[:, hi] * w[None, :, None, None]
        return kk, ss

    # boundary: diffusion approximation at the deepest point
    kinv = torch.linalg.inv(k_matrix[:, 0].double())
    dtau = (tau[0] - tau[1])
    dS = (source[:, 0].double() - source[:, 1].double()) / dtau
    I = (kinv @ (dS / k_matrix[:, 0].double().amax(dim=(-2, -1), keepdim=True)).unsqueeze(-1)).squeeze(-1) + source[:, 0].double()
    I = I.to(dtype)

    dt = -tau_r[1] + tau_r[0]   # integration step (positive going up)
    for i in range(nt_r - 1):
        x0, x1 = tau_r[i], tau_r[i + 1]
        h = x0 - x1             # > 0
        k0, s0 = interp(x0)
        k1, s1 = interp(x1)
        km, sm = interp(0.5 * (x0 + x1))
        # I' = K(I - S) = K I - K S
        def f(kk, ss, ii):
            return kk @ (ii - ss).unsqueeze(-1)
        k1_ = f(k0, s0, I)
        k2_ = f(km, sm, I + 0.5 * h * k1_)
        k3_ = f(km, sm, I + 0.5 * h * k2_)
        k4_ = f(k1, s1, I + h * k3_)
        I = I + (h / 6.0) * (k1_ + 2 * k2_ + 2 * k3_ + k4_).squeeze(-1)
    return I


# ---------------------------------------------------------------------------
# solver registry
# ---------------------------------------------------------------------------
# RTE solver registry (case-insensitive): config['synthesis']['solver']
# accepts any of these keys; 'hermitian' is the default and remains the
# reference method.
_RTE_SOLVERS = {
    "hermitian": hermite_solve,
    "cn": cn_solve,
    "delo": delo_solve,
}


def resolve_rte_solver(name):
    """Map a ``synthesis.solver`` config value to its solver callable.

    Case-insensitive; raises ValueError on an unknown name.  The
    callables share the signature ``(ltau, k_matrix, source) -> (Nb,
    Nw, 4)``.
    """
    key = str(name).strip().lower()
    try:
        return _RTE_SOLVERS[key]
    except KeyError:
        raise ValueError(
            f"unknown RTE solver {name!r}; available: "
            f"{sorted(_RTE_SOLVERS)}") from None
