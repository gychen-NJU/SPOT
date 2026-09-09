# -*- coding: utf-8 -*-
"""
spot.inversion.nodes — node <-> full-grid parameter conversion
================================================================
The number of free parameters of a physical quantity (the *nodes*) is
much smaller than the number of depth points.  The node values are the
inversion parameters; a cubic-spline interpolation maps them back onto
the full optical-depth grid.

The interpolation is differentiable with respect to the node values so
that response functions can be computed by autograd through the whole
node -> grid -> synthesis chain.
"""

import torch

__all__ = ["node_positions", "node_grid_weights", "nodes_to_grid",
           "grid_to_nodes", "QUANTITY_NAMES", "MULTIPLICATIVE", "ADDITIVE",
           "node_divisors", "snap_node_count",
           "auto_node_count"]

# Quantities that may carry nodes (in layout order: 1..8)
QUANTITY_NAMES = ["T", "Pe", "B", "gamma", "phi", "vlos", "vmic", "vmac"]
# update mode of each quantity:
#   T, Pe, micro, B, vlos -> multiplicative  a_new = a * (1 + da)
#   gamma, phi, vmac      -> additive        a_new = a + da
MULTIPLICATIVE = {"T", "Pe", "B", "vlos", "vmic"}
ADDITIVE = {"gamma", "phi", "vmac"}


def node_positions(ltau, nnodes):
    """
    Optical-depth positions of the nodes for a given node count.

    The nodes are placed at equally spaced *integer* depth steps
    (``mm = (ntau-1)/(mnodos-1)``, nodes at 1, 1+mm, ...), so the node
    grid is strictly uniform and may not reach the surface layer (e.g.
    31 nodes on 55 layers cover depths 0..30).  ``nnodes >= nt`` clamps
    to the full grid.

    Parameters
    ----------
    ltau : torch.Tensor (Nt,)
        log10(tau), decreasing (deepest first).
    nnodes : int
        Number of nodes (>= 1).

    Returns
    -------
    torch.LongTensor (Nn,)
        Indices into the depth grid of the node positions.
    """
    nt = ltau.shape[0]
    if nnodes <= 1:
        return torch.tensor([nt // 2], device=ltau.device, dtype=torch.long)
    if nnodes >= nt:
        return torch.arange(nt, device=ltau.device)
    mm = (nt - 1) // (nnodes - 1)          # integer spacing
    idx = torch.arange(nnodes, device=ltau.device) * mm
    return idx.clamp(0, nt - 1).unique()

def node_grid_weights(ltau, node_idx):
    """
    Cubic-spline interpolation matrix: grid = W @ node_values.

    Natural cubic spline through the node points (normalized depth
    coordinate 0 = deepest, 1 = top), evaluated at every depth point.
    The spline is linear in the node values, hence the whole mapping is
    the matrix W of shape (Nt, Nn); gradients w.r.t. the node values
    are exactly W.

    Parameters
    ----------
    ltau : torch.Tensor (Nt,)
        log10(tau), decreasing.
    node_idx : torch.LongTensor (Nn,)
        Depth indices of the nodes.

    Returns
    -------
    torch.Tensor (Nt, Nn)
    """
    nt = ltau.shape[0]
    nn = node_idx.shape[0]
    device, dtype = ltau.device, ltau.dtype
    if nn == 1:
        return torch.ones(nt, 1, device=device, dtype=dtype)

    x = torch.linspace(0.0, 1.0, nt, device=device, dtype=dtype)
    xn = node_idx.to(device=device).double().to(dtype) / (nt - 1)
    h = torch.diff(xn)                                   # (Nn-1,)

    if nn == 2:
        # linear interpolation between the two nodes (degenerate spline)
        w = torch.zeros(nt, 2, device=device, dtype=dtype)
        t = (x - xn[0]) / (xn[1] - xn[0])
        w[:, 0] = (1.0 - t).clamp(0, 1)
        w[:, 1] = t.clamp(0, 1)
        return w

    if nn == 3:
        # n==3: quadratic interpolation
        w = torch.zeros(nt, 3, device=device, dtype=dtype)
        xx = (x - xn[0]) / (xn[1] - xn[0])
        xx2 = xx * xx
        w[:, 0] = 1.0 - 1.5 * xx + 0.5 * xx2
        w[:, 1] = 2.0 * xx - xx2
        w[:, 2] = -0.5 * xx + 0.5 * xx2
        return w

    # --- not-a-knot cubic spline ----------------------------------------
    # The nodes are strictly equally spaced (node_positions uses the
    # integer step mm), so h is uniform and the normalized second
    # derivatives m' = h^2/6 * m satisfy (SPLINB):
    #   interior i:  m'_{i-1} + 4 m'_i + m'_{i+1} = y_{i-1} - 2 y_i + y_{i+1}
    #   endpoint 1:  m'_1 - 5 m'_2 + m'_3 = -0.5 y_1 + y_2 - 0.5 y_3
    #   endpoint N:  m'_N - 5 m'_{N-1} + m'_{N-2}
    #                                     = -0.5 y_N + y_{N-1} - 0.5 y_{N-2}
    # (not-a-knot boundary; unlike the natural spline used before, its
    #  spline spaces ARE nested when the node set grows, so cycling to a
    #  finer node set does not inflate the chi2 baseline).
    m_mat = torch.zeros(nn, nn, device=device, dtype=dtype)
    d_mat = torch.zeros(nn, nn, device=device, dtype=dtype)
    for i in range(1, nn - 1):
        m_mat[i, i - 1] = 1.0
        m_mat[i, i] = 4.0
        m_mat[i, i + 1] = 1.0
        d_mat[i, i - 1] = 1.0
        d_mat[i, i] = -2.0
        d_mat[i, i + 1] = 1.0
    m_mat[0, 0], m_mat[0, 1], m_mat[0, 2] = 1.0, -5.0, 1.0
    m_mat[nn - 1, nn - 1], m_mat[nn - 1, nn - 2], m_mat[nn - 1, nn - 3] = \
        1.0, -5.0, 1.0
    d_mat[0, 0], d_mat[0, 1], d_mat[0, 2] = -0.5, 1.0, -0.5
    d_mat[nn - 1, nn - 1], d_mat[nn - 1, nn - 2], d_mat[nn - 1, nn - 3] = \
        -0.5, 1.0, -0.5
    # g_mat maps node values y to the normalized second derivatives m'
    g_mat = torch.linalg.solve(m_mat, d_mat)             # (Nn, Nn)

    # --- evaluate at every depth point --------------------------------
    # splines22: y(x) = f1 y_j1 + f2 y_j2 + f3 m'_j1 + f4 m'_j2
    # with f1 = 1-t, f2 = t, f3 = f1*(f1^2-1), f4 = f2*(f2^2-1).
    #
    # NOTE (SIR parity): SIR evaluates the cubic on the LAST segment for
    # every depth point beyond the last node (its j1 = 1+int(...) is not
    # clamped, so t keeps growing and the spline EXTRAPOLATES).  spot
    # clamps the segment index for safety (out-of-range gather) but must
    # NOT clamp t itself: clamping t to [0, 1] collapses the weights to
    # the last node value and silently drops the extrapolation.  This is
    # visible whenever the node set does not reach the surface, e.g. 5
    # nodes on the 55-layer grid cover log tau = 1.4 .. -3.8 while the
    # grid runs to -4.0: the two top layers then disagreed with SIR by up
    # to 0.59 in a spline weight (checked against splines22/SPLINB in
    # audit/step13_spline_parity.py).
    paso = h[0].clamp(min=1e-12)
    j1 = (x / paso).floor().long().clamp(0, nn - 2)      # segment start
    t = (x - xn[j1]) / paso
    f1 = 1.0 - t
    f2 = t
    f3 = f1 * (f1 * f1 - 1.0)
    f4 = f2 * (f2 * f2 - 1.0)

    w = torch.zeros(nt, nn, device=device, dtype=dtype)
    w.scatter_add_(1, j1.unsqueeze(1), f1.unsqueeze(1))
    w.scatter_add_(1, (j1 + 1).unsqueeze(1), f2.unsqueeze(1))
    w = w + f3.unsqueeze(1) * g_mat[j1, :] \
        + f4.unsqueeze(1) * g_mat[(j1 + 1).clamp(max=nn - 1), :]
    return w


def nodes_to_grid(ltau, node_values, node_idx):
    """
    Map node values onto the full depth grid.

    Parameters
    ----------
    ltau : torch.Tensor (Nt,)
    node_values : torch.Tensor (Nb, Nn) or (Nn,)
    node_idx : torch.LongTensor (Nn,)

    Returns
    -------
    torch.Tensor (Nb, Nt) or (Nt,)
    """
    w = node_grid_weights(ltau, node_idx)
    if node_values.dim() == 1:
        return w @ node_values
    return node_values @ w.t()


def grid_to_nodes(ltau, grid_values, node_idx):
    """
    Project a full-grid quantity onto the node values (least squares
    through the spline matrix).

    Parameters
    ----------
    ltau : torch.Tensor (Nt,)
    grid_values : torch.Tensor (Nb, Nt) or (Nt,)
    node_idx : torch.LongTensor (Nn,)

    Returns
    -------
    torch.Tensor (Nb, Nn) or (Nn,)
    """
    w = node_grid_weights(ltau, node_idx)                  # (Nt, Nn)
    wtw = w.t() @ w
    wty = w.t() @ grid_values.t()
    out = torch.linalg.solve(wtw, wty).t()
    return out.squeeze(0) if grid_values.dim() == 1 else out


# ---------------------------------------------------------------------------
# Automatic node selection (node-count auto-tuning)
# ---------------------------------------------------------------------------

def node_divisors(ntau):
    """
    Divisor list ``ndiv`` used to snap node counts.

    ``[-1, 0, 1, 2]`` followed by every ``i`` in 3..ntau with
    ``(ntau-1) % (i-1) == 0`` (i.e. the nodes divide the grid evenly).
    """
    ndiv = [-1, 0, 1, 2]
    for i in range(3, ntau + 1):
        if (ntau - 1) % (i - 1) == 0:
            ndiv.append(i)
    return ndiv


def snap_node_count(m, ntau):
    """
    Snap a node count to the largest divisor <= m.

    Every requested count goes through this snap, so even explicitly
    requested counts are snapped; e.g. 5 nodes on 55 layers become 4.
    """
    ndiv = node_divisors(ntau)
    best = m
    for v in ndiv:
        if v <= m:
            best = v
    return best


def auto_node_count(derivada, mp, ntau):
    """
    Automatic node-count criterion: the effective number of nodes for one
    parameter from the derivative of chi^2 w.r.t. the parameter.

    ``derivada(i) = sum_j difer(j) * rt(i, j)``
    with ``difer(j) = (obs-jmod(j))/sig(j)**2`` and ``rt`` the normalized
    grid response function.  ``mp`` is the maximum allowed node count
    (the snapped cycle cap).  Vectorized over the depth axis; the
    criterion uses candidate counts from the divisor list, trapezoid
    areas of the derivative normalized by a0, sesgo1=0 / sesgo2=1.0.

    Parameters
    ----------
    derivada : torch.Tensor (Nb, Nt)
        Per-depth chi2 derivative of the parameter (any physical units;
        the criterion uses only ratios / signs).
    mp : int
        Maximum number of nodes allowed.
    ntau : int
        Grid size.

    Returns
    -------
    int — the selected node count (``nodosposibles(imax)``).
    """
    d = derivada.mean(dim=0)                     # batch -> single pixel
    d = d.detach().double()
    dev = d.device

    if mp < 2:
        return 1

    # candidate counts: [1, 2] + [j: (ntau-1) % (j-1) == 0]
    nod = [1, 2]
    for j in range(3, ntau + 1):
        if (ntau - 1) % (j - 1) == 0:
            nod.append(j)
    nummax = len(nod)

    a0 = (abs(d[0]) + 2.0 * abs(d[1:-1]).sum() + abs(d[-1])) / 2.0
    a1 = (d[0] + 2.0 * d[1:-1].sum() + d[-1]) / 2.0
    a1 = a1 / a0
    if a1 == 1.0 or not torch.isfinite(a1):
        return 1

    amax = abs(a1)
    # largest candidate index with nodosposibles(i) <= mp (nummax2, 1-based i
    # starts at 2 -> 0-based index 1)
    nummax2 = 1
    for i in range(1, nummax):
        if nod[i] <= mp:
            nummax2 = i
        else:
            break

    dmax = -a0
    dmin = a0
    idmax = idmin = 0
    for i in range(ntau):
        if d[i] > dmax:
            dmax = d[i]
            idmax = i
        if d[i] < dmin:
            dmin = d[i]
            idmin = i
    if dmax * dmin >= 0.0:
        return 1

    i00n = min(idmax, idmin)                     # 0-based
    # first sign-change below the strongest extrema
    i11 = i00n
    while i11 < ntau - 2 and (d[i11] * d[i11 + 1]) > 0.0:
        i11 += 1

    aijmax = -10.0 * a0
    aijmin = 10.0 * a0
    # NOTE (SIR parity): criterio.f initialises ``imax=1`` (1-based), i.e.
    # the 2-node candidate, BEFORE the special split.  With sesgo1 = 0 the
    # special split's area is identically 0, so ``a > amax`` never fires
    # and this initial value is what decides the 2-node case: if no
    # candidate later beats ``amax`` the criterion returns 2 nodes, not 1.
    # In 0-based terms that is ``imax = 1`` -> ``nod[1] = 2``.
    imax = 1
    # special 2-node split at i11
    aij = d[0] / 2.0 + d[1:i11].sum() + d[i11] / 2.0
    aij = aij / a0
    aijmax = max(aijmax, aij)
    aijmin = min(aijmin, aij)
    aij = d[i11] / 2.0 + d[i11 + 1:ntau - 1].sum() + d[ntau - 1] / 2.0
    aij = aij / a0
    aijmax = max(aijmax, aij)
    aijmin = min(aijmin, aij)
    a = abs(aijmax - aijmin) * 0.0              # sesgo1 = 0
    if a == amax:
        return 2
    if a > amax:
        amax = a
        imax = 1                                # candidate '2'

    # candidate loop: equal-area subintervals
    #
    # NOTE (SIR parity): the candidate list is nod[0]=1, nod[1]=2,
    # nod[2]=3, ...  SIR's loop is ``do i=2,nummax2`` in 1-based terms,
    # i.e. it starts at nodosposibles(2) = 2 and never re-tests the
    # 1-node entry (which is handled by the ``a1 == 1`` early return).
    # In 0-based terms that is ``range(2, nummax2 + 1)``.  Starting at 1
    # instead re-evaluates the 2-node candidate with the equal-area
    # rule (a DIFFERENT expression from the special split above) and
    # usually yields the largest area, so the criterion systematically
    # returned 2 nodes; measured against a literal port of criterio.f it
    # agreed only 190/300 times (audit/step15_auto_nodes.py).
    # NOTE (SIR parity, second bug): ``imax`` was reset to 0 here, which
    # threw away the ``imax = 1`` set by the special 2-node branch above.
    # SIR keeps ``imax`` from that branch into the candidate loop, so a
    # derivative whose 2-node split is the best split must return 2; the
    # reset made it return 1 whenever no candidate beat it.
    best = amax
    for i in range(2, nummax2 + 1):
        nodc = nod[i]
        if nodc < 2:
            continue
        paso = int(round((ntau - 1) / float(nodc)))
        aijmax = -10.0 * a0
        aijmin = 10.0 * a0
        for j in range(1, nodc + 1):
            i0 = (j - 1) * paso
            i1 = j * paso
            # middle sum (0-based k = i0 .. i1-2), endpoints halved
            aij = d[i0] / 2.0
            if i1 - 1 > i0:
                aij = aij + d[i0 + 1:i1].sum()
            aij = aij + d[min(i1, ntau - 1)] / 2.0
            aij = aij / a0
            aijmax = max(aijmax, aij)
            aijmin = min(aijmin, aij)
        a = abs(aijmax - aijmin)
        if a > best:
            best = a
            imax = i
    return nod[imax]
