# -*- coding: utf-8 -*-
"""
spot.inversion.inversion — the Inversion class
================================================
Batched, node-based Stokes inversion:

* the atmosphere is described by a small number of *nodes* per
  physical quantity (``config['inversion']['nodes']``);
* a Levenberg-Marquardt scheme with damped SVD iteratively updates the
  node values using the response functions of the full forward model;
* cycles refine the solution with progressively more nodes (config
  nodes may be a list with one entry per cycle; each cycle nests a
  finer node stratification on the previous solution);
* the final parameter errors are estimated from the covariance matrix
  at the solution.

The response functions are computed with autograd through the
differentiable forward model (Synthesis), so the inversion reuses
exactly the same code path as the synthesis.
"""

import time

import numpy as np
import torch

from ..config import merge_config
from ..default import DEFAULT_CONFIG
from ..synthesis.synthesis import Synthesis
from .marquardt import chi_square, marquardt_step, solve_damped_svd
from .nodes import (ADDITIVE, MULTIPLICATIVE, grid_to_nodes, node_positions,
                    nodes_to_grid, node_grid_weights, auto_node_count,
                    snap_node_count)

# Replicate the reference y-factor on the additive angle channels: the
# default forces gamma/phi onto the per-radian scaling matched to the
# reference normal equations; this flag instead uses the per-unit
# node-value scaling (like the multiplicative quantities), configurable
# for back-to-back comparison/testing.
GAMMA_PHI_QUIRK = False

# Normal equations: the reference code carries the angle channels in
# radians while the spot atmosphere/param units are degrees, so the
# angle blocks are scaled by 180/pi to sit on the same footing.
_RAD_TO_DEG = 57.29577951308232

__all__ = ["Inversion", "InversionResult"]


class InversionResult:
    """Container with the results of an inversion."""

    def __init__(self, atmos, chi2, errors=None, history=None, cycles=1,
                 iterations=0, converged=False, runtime=0.0):
        self.atmos = atmos          # (Nb, Nx) final atmosphere
        self.chi2 = chi2            # (Nb,) chi2 of the final fit
        self.errors = errors        # (Nb, Nx) parameter errors or None
        self.history = history      # list of dicts, one per iteration:
                                    #   cycle, iteration, chi2, lambda,
                                    #   accepted, nparams, nodes
        self.cycles = cycles
        self.iterations = iterations
        self.converged = converged
        self.runtime = runtime

    @property
    def chi2_history(self):
        """chi2 of every logged iteration (flat list of floats)."""
        return [h["chi2"] for h in (self.history or [])]

    @property
    def cycles_history(self):
        """[(chi2 list, node config dict), ...] grouped per cycle."""
        groups = []
        for h in (self.history or []):
            if groups and groups[-1][1] == h["nodes"]:
                groups[-1][0].append(h["chi2"])
            else:
                groups.append(([h["chi2"]], h["nodes"]))
        return groups

    def __repr__(self):
        return (f"InversionResult(chi2={self.chi2.tolist()}, "
                f"cycles={self.cycles}, iterations={self.iterations}, "
                f"converged={self.converged}, runtime={self.runtime:.2f}s)")


class Inversion:
    """
    Batched Stokes inversion with nodes + Levenberg-Marquardt.

    Parameters
    ----------
    config : dict, optional
        User configuration merged over the defaults; see
        :mod:`spot.default` for the ``inversion`` section.
    """

    def __init__(self, config=None):
        self.config = merge_config(config or {}, DEFAULT_CONFIG)
        self.synthesis = Synthesis(self.config)
        self.device = self.synthesis.device
        self.dtype = self.synthesis.dtype
        inv = self.config["inversion"]
        self.nodes_cfg = inv["nodes"]
        self.max_cycles = int(inv["max_cycles"])
        self.max_iterations = int(inv["max_iterations"])
        self.chi2_tolerance = float(inv["chi2_tolerance"])
        self.lambda0 = float(inv["lambda0"])
        self.lambda_max = float(inv["lambda_max"])
        self.lambda_factor = float(inv["lambda_factor"])
        self.lambda_decrease = float(inv["lambda_decrease"])
        self.max_step = inv["max_step"]
        self.hse_pg0 = float(inv.get("hse_pg0", 0.0))
        self.stokes_weights = inv["stokes_weights"]
        # Per-physical-quantity singular-value truncation (groups).  A
        # group-wise cutoff keeps the small-curvature quantities (B,
        # gamma, phi) from being drowned out by the large-curvature ones
        # (T, vlos): truncation is applied separately within each
        # quantity rather than globally.  Verified at a near-optimum
        # state: the group version reproduces the reference step (RMS
        # ~6e-2) while global truncation cuts the angle channels and
        # deviates ~20x more.  The earlier failure of the group version
        # at near-optimum states was traced to the missing per-RADIAN
        # angle scaling -- now applied in _run_cycle.
        self.group_truncation = bool(inv.get("group_truncation", True))
        self.svd_tolerance = float(inv["svd_tolerance"])
        self.compute_errors = bool(inv["compute_errors"])
        self.verbose = bool(inv["verbose"])
        self.log_history = bool(inv["log_history"])
        self._wavs_cache = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def invert(self, wavs, ltau, target, initial=None, sigma=None):
        """
        Invert a batch of Stokes profiles.

        Parameters
        ----------
        wavs : torch.Tensor (Nw,)
            Wavelength grid [Angstrom].
        ltau : torch.Tensor (Nt,)
            log10(tau5000), decreasing (deepest first).
        target : torch.Tensor (Nb, Nw, 4)
            Observed Stokes profiles to fit.
        initial : torch.Tensor (Nb, Nx) or str or dict, optional
            Initial-guess atmosphere.  A string is interpreted as a
            preset model name (e.g. 'hot11', 'FALC11'); a dict from
            ``load_atmosphere`` is also accepted.  Defaults to the
            config preset ``config['atmosphere']``.
        sigma : torch.Tensor or float, optional
            Noise level per Stokes sample.  Shape (4, Nw), (Nb, 4, Nw)
            or a scalar.  Default 'auto': 1e-3 * max|I| per batch.

        Returns
        -------
        InversionResult
        """
        wavs = torch.as_tensor(wavs, dtype=self.dtype, device=self.device)
        ltau = torch.as_tensor(ltau, dtype=self.dtype, device=self.device)
        target = torch.as_tensor(target, dtype=self.dtype, device=self.device)
        if target.ndim == 2:
            target = target.unsqueeze(0)
        nb = target.shape[0]
        nt = ltau.shape[0]
        nx = 6 * nt + 2

        if initial is None:
            initial = self.config["atmosphere"]
        atmos = self.synthesis._normalize_atmos(initial, ltau)
        if atmos.shape[1] == 7 * nt + 1:
            # Convention: microturbulence is ONE parameter (a single
            # node; the .mod file stores the same value in every layer),
            # so it is never inverted depth-resolved.  A genuinely
            # depth-varying initial profile is collapsed to its
            # depth-average (the scalar seed of the inversion parameter).
            vmic_mean = atmos[:, 6 * nt:7 * nt].mean(dim=1, keepdim=True)
            atmos = torch.cat([atmos[:, :6 * nt], vmic_mean,
                               atmos[:, 7 * nt:]], dim=1)
            if self.verbose:
                print("inversion: microturbulence is a single parameter "
                      "-- the initial depth profile was averaged into a "
                      "scalar seed", flush=True)
        if atmos.shape[0] == 1 and nb > 1:
            atmos = atmos.repeat(nb, 1)
        if atmos.shape[0] != nb:
            raise ValueError(
                f"initial guess batch {atmos.shape[0]} != target batch {nb}")

        sigma = self._resolve_sigma(sigma, target)
        # the observed data is (Nb, Nw, 4) (wavelength-major) while the
        # noise is per (Stokes, wavelength): transpose to (Nb, Nw, 4)
        # so that the flattened sigma pairs with the flattened data
        # (the stoke-major / wav-major ordering is swapped here).
        if sigma.ndim == 3 and sigma.shape[1] == 4:
            sigma = sigma.permute(0, 2, 1).contiguous()
        self._wavs_cache = wavs

        t0 = time.time()
        nodes_cfg = self._cycle_nodes()
        n_cycles = min(len(nodes_cfg), self.max_cycles)
        history = []
        best_atmos = None
        best_chi2 = None

        for icycle in range(n_cycles):
            n_cfg, auto_set = nodes_cfg[icycle]
            atmos, chi2, hist, iters, converged, final_counts = \
                self._run_cycle(wavs, ltau, target, sigma, atmos, n_cfg,
                                auto_set, icycle)
            history.extend(hist)
            # keep the best result over all cycles (a later cycle with more
            # nodes may occasionally degrade before converging)
            if best_chi2 is None or torch.all(chi2 <= best_chi2):
                best_chi2 = chi2
                best_atmos = atmos.clone()
            if self.verbose:
                print(f"[cycle {icycle + 1}/{n_cycles}] chi2 = "
                      f"{chi2.mean().item():.4e}  ({iters} iters)")
        atmos = best_atmos
        chi2 = best_chi2

        errors = None
        if self.compute_errors:
            errors = self._compute_errors(wavs, ltau, target, sigma, atmos,
                                          final_counts)

        return InversionResult(
            atmos=atmos, chi2=chi2, errors=errors, history=history,
            cycles=n_cycles, iterations=len(history), converged=converged,
            runtime=time.time() - t0)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _cycle_nodes(self):
        """
        Expand the per-quantity node config into a per-cycle list.

        Returns a list of ``(cfg, auto_set)`` per cycle: ``cfg`` maps
        each quantity to its int node count (or -1 when automatic, i.e.
        ``*`` -> 1000) and ``auto_set`` holds the quantity names whose
        count is selected automatically from the chi2 derivative.
        """
        maxlen = 1
        for q in self.nodes_cfg.values():
            if isinstance(q, (list, tuple)):
                maxlen = max(maxlen, len(q))
        cycles = []
        for i in range(maxlen):
            cfg = {}
            auto = set()
            for q, v in self.nodes_cfg.items():
                if isinstance(v, (list, tuple)):
                    val = v[min(i, len(v) - 1)]
                else:
                    val = v
                if (isinstance(val, str)
                        and val.strip().lower() in ("auto", "*")) or \
                        (isinstance(val, int) and val == -1):
                    auto.add(q)
                    cfg[q] = -1
                else:
                    cfg[q] = int(val)
            cycles.append((cfg, auto))
        return cycles

    def _resolve_sigma(self, sigma, target):
        if sigma is None or (isinstance(sigma, str) and sigma == "auto"):
            # photon-noise style per-stokes scaling on max|I| (per column):
            #   sigma_I = 0.118 * sqrt(max|I|)
            #   sigma_Q = sigma_U = sigma_V = 0.204 * sqrt(max|I|)
            scale = target[:, :, 0].abs().amax(dim=1).clamp(min=1e-30).sqrt()
            w = torch.tensor([0.118, 0.204, 0.204, 0.204],
                             device=target.device, dtype=target.dtype)
            nw = target.shape[1]
            return (scale[:, None, None].expand(target.shape[0], nw, 4)
                    * w[None, None, :])                       # (Nb,Nw,4)
        if isinstance(sigma, str) and sigma == "sir":
            # Photon-noise-like weighting:
            #   sig = 1/(sn*sqrt(w)) * sqrt(smax(k,i)/vic(k)), where
            #   vic(k)  = continuum of line k  = max Stokes I within line k
            #   smax(k,1) = max |vic - I| (line depth); smax(k,i>1) = max|stokes|
            # weights enter as sqrt(w) inside sigma (chi2 ~ w/sigma^2).
            # The band is shared by the lines: every wavelength point is
            # assigned to its NEAREST line centre (the band is split per
            # line), so smax/vic are computed per line window and the
            # sigma is uniform per (line, stokes).
            inv = self.config["inversion"]
            sn = float(inv.get("snr", 1000.0))
            w = inv.get("stokes_weights", [1.0, 1.0, 1.0, 1.0])
            w = torch.as_tensor(w, dtype=self.dtype, device=self.device)
            nw = target.shape[1]
            wavs = self._wavs_cache.to(self.device) \
                if self._wavs_cache is not None else None
            if wavs is None:
                # fallback: no line info -> single window over the whole band
                centers = torch.zeros(1, device=self.device, dtype=self.dtype)
                belong = torch.zeros(nw, device=self.device, dtype=torch.long)
            else:
                centers = torch.tensor(
                    [s["wvac"] * 1e8 for s in (self.synthesis._line_static or [])],
                    device=self.device, dtype=self.dtype)
                if centers.numel() == 0:
                    centers = wavs.mean().reshape(1)
                belong = (wavs[None, :] - centers[:, None]).abs().argmin(0)

            sig = torch.zeros((target.shape[0], 4, nw), device=self.device,
                              dtype=self.dtype)
            for k in range(centers.shape[0]):
                sel = belong == k
                if not bool(sel.any()):
                    continue
                tk = target[:, sel, :]                     # (Nb,nsel,4)
                vk = tk[:, :, 0].abs().amax(dim=1).clamp(min=1e-30)
                mk = torch.stack([
                    (vk.unsqueeze(-1) - tk[:, :, 0]).abs().amax(dim=1),
                    tk[:, :, 1].abs().amax(dim=1),
                    tk[:, :, 2].abs().amax(dim=1),
                    tk[:, :, 3].abs().amax(dim=1)], dim=1)  # (Nb,4)
                sk = (1.0 / (sn * torch.sqrt(w))) \
                    * torch.sqrt(mk / vk.unsqueeze(-1))     # (Nb,4)
                sig[:, :, sel] = sk.unsqueeze(-1)
            # cotaminima: points whose Stokes value drops below -0.99 are
            # excluded from the inversion (sigma -> 1e15)
            cotaminima = float(inv.get("cotaminima", -0.99))
            excl = (target < cotaminima).permute(0, 2, 1)   # (Nb,4,Nw)
            big = torch.full_like(sig, 1e15)
            sig = torch.where(excl, big, sig)
            return sig
        if isinstance(sigma, (int, float)):
            return torch.full((target.shape[0], 1, 1), float(sigma),
                              device=self.device, dtype=self.dtype)
        sigma = torch.as_tensor(sigma, dtype=self.dtype, device=self.device)
        if sigma.ndim == 1 and sigma.shape[0] == 4:
            sigma = sigma.unsqueeze(1)      # (4,1) broadcast over Nw
        if sigma.ndim == 2:                 # (4, Nw)
            sigma = sigma[None]
        if sigma.ndim == 3:
            sigma = sigma[:, None] if sigma.shape[1] == 1 else sigma
        return sigma

    def _param_layout(self, nodes_cfg, nt):
        """
        Build the free-parameter layout from the node config.

        Returns (names, node_idx, update_mode, max_step, blocks)
        where blocks maps each quantity to its column slice of the
        parameter vector, and node_idx contains the depth index of
        every node (concatenated in quantity order).

        Node counts are first snapped to the divisor list of the grid
        (the nearest smaller divisor that divides the grid evenly), so
        the node positions stay on an exact integer sub-grid.
        """
        names, idx, modes, caps = [], [], [], []
        blocks = {}
        ltau_dummy = torch.linspace(0.0, -1.0, nt, device=self.device)
        for q in ("T", "Pe", "B", "gamma", "phi", "vlos", "vmic", "vmac"):
            n = nodes_cfg.get(q, 0)
            if n <= 0:
                continue
            n = snap_node_count(n, nt)
            pos = node_positions(ltau_dummy, n)
            nn = len(pos)          # actual node count (may be < n when
                                   # n >= nt: clamped to the full grid)
            names.extend([q] * nn)
            idx.append(pos)
            mode = "multiplicative" if q in MULTIPLICATIVE else "additive"
            modes.extend([mode] * nn)
            cap = self.max_step.get(q)
            caps.extend([cap] * nn)
            blocks[q] = (len(names) - nn, len(names))
        node_idx = torch.cat(idx) if idx else torch.empty(0, dtype=torch.long)
        return names, node_idx, modes, caps, blocks

    # ------------------------------------------------------------------
    def _run_cycle(self, wavs, ltau, target, sigma, atmos, nodes_cfg,
                   auto_set, icycle):
        """One LM cycle.  Returns (atmos, chi2, history, iters, converged,
        final_counts).

        ``auto_set``: quantities whose node count is chosen automatically
        every iteration (``*`` -> m(i)=1000, from the chi2-derivative
        criterion).
        """
        nt = ltau.shape[0]
        nb = target.shape[0]

        # --- resolve the per-cycle node counts ------------------------------
        # Max nodes from the data (bounded by the number of constraints),
        # then snap each count onto the divisor list of the grid.
        counts = {}
        if auto_set:
            n_auto = len(auto_set)
            n_det = sum(int(max(v, 0)) for q, v in nodes_cfg.items()
                        if q not in auto_set)
            nfrecs = wavs.shape[0] * sum(
                1 for w in self.stokes_weights if float(w) != 0.0)
            nmax1 = int(round((min(200, nfrecs) - n_det) / (n_auto + 2)))
            for q in auto_set:
                cap = 2 * nmax1 if q == "T" else nmax1
                counts[q] = snap_node_count(max(cap, 1), nt)
            caps_snap = dict(counts)
        else:
            caps_snap = {}
        for q, v in nodes_cfg.items():
            if q not in auto_set:
                counts[q] = snap_node_count(int(v), nt) if int(v) > 0 else 0
        # round the counts into the layout
        names, node_idx, modes, caps, blocks = self._param_layout(counts, nt)
        nparams = len(names)

        if nparams == 0:
            chi2 = self._chi2(wavs, ltau, target, sigma, atmos)
            return atmos, chi2, [], 0, True, counts

        q_ids = {q: i for i, q in enumerate(
            ("T", "Pe", "B", "gamma", "phi", "vlos", "vmic", "vmac"))}

        def _groups(blocks_d):
            g = []
            for q in ("T", "Pe", "B", "gamma", "phi", "vlos", "vmic", "vmac"):
                if q in blocks_d:
                    g.extend([q_ids[q]] * (blocks_d[q][1] - blocks_d[q][0]))
            return g or None

        groups = _groups(blocks)

        # initial node values: direct sampling from the grid.  NOTE
        # (reference parity): the cycle baseline stays the RAW input
        # atmosphere -- the normalization projection is skipped on the
        # first call, so it0 uses the unprojected grid and the automatic
        # recount (which can change the counts before the first trial)
        # extracts the node values from the raw grid too.  An earlier
        # version replaced the baseline with the caps-count projection,
        # which changed the it0 derivatives and the first step direction.
        params = self._extract_nodes(ltau, atmos, blocks, node_idx, nt)
        pe_not_inverted = "Pe" not in blocks
        lamda = torch.full((nb,), self.lambda0, device=self.device,
                           dtype=self.dtype)
        chi2_best = self._chi2(wavs, ltau, target, sigma, atmos)
        atmos_best = atmos.clone()
        params_best = params.clone()
        layout_best = dict(counts)
        history = []                 # structured log, see _record()
        converged = False
        iters = 0
        # Loop state: chi0 = chi2 of the last accepted iteration,
        # ngu = accepted-iteration counter, rejected_streak = consecutive
        # rejections (7 in a row ends the cycle).
        chi0 = chi2_best.clone()
        ngu = 0
        rejected_streak = torch.zeros((nb,), device=self.device,
                                      dtype=torch.long)

        def _record(accepted):
            history.append({
                "cycle": icycle, "iteration": iters,
                "chi2": float(chi2_best.mean().item()),
                "lambda": float(lamda.mean().item()),
                "accepted": bool(accepted),
                "nparams": len(names),
                "nodes": dict(counts),
            })

        _record(True)                # starting point (cycle baseline)
        rf_method = self.synthesis.config["synthesis"]["rf_method"]

        def _grid_rf(atm):
            if rf_method == "analytic_chain":
                return self.synthesis._rf_analytic_chain(
                    wavs, ltau, atm, pe_not_inverted=pe_not_inverted)
            if rf_method == "fast":
                return self.synthesis._rf_fast(
                    wavs, ltau, atm, pe_not_inverted=pe_not_inverted)
            return self.synthesis._rf_analytic(
                wavs, ltau, atm, pe_not_inverted=pe_not_inverted)

        for it in range(self.max_iterations):
            iters += 1
            # layout-consistency guard: the batched automatic-node
            # reselection can transiently leave nparams and node_idx on
            # different versions of the layout (observed 26 vs 27 in a
            # batched cycle-4 run; _param_layout itself is consistent,
            # the desync comes from the interleaved reselection).  When
            # detected, rebuild the whole layout from `counts` so every
            # per-parameter structure (names/node_idx/blocks/modes/params)
            # agrees again before the forward step.
            if int(node_idx.numel()) != nparams:
                names, node_idx, modes, caps, blocks = \
                    self._param_layout(counts, nt)
                groups = _groups(blocks)
                nparams = len(names)
                params = self._extract_nodes(ltau, atmos, blocks,
                                             node_idx, nt)
                params_best = params.clone()
            # forward + response functions of the node parameters
            trial_atmos = self._atmos_from_params(ltau, atmos, params, blocks,
                                                  node_idx, nt, nb)
            ymod = self.synthesis._forward(wavs, ltau, trial_atmos)
            rf_grid = _grid_rf(trial_atmos)
            ymod_flat = ymod.reshape(nb, -1)
            target_flat = target.reshape(nb, -1)
            sig_flat = sigma.reshape(nb, -1)
            dy = (target_flat - ymod_flat) / sig_flat

            # Choose the effective node counts of the auto quantities
            # from the chi2 derivative: each depth carries the
            # correlation of the residual with the grid response
            # function (derivada(i) = sum_j difer(j)*rt(i,j),
            #  difer = (obs-mod)/sig**2, rt = normalized grid RF).
            counts_accept = dict(counts)
            if auto_set:
                derivs = {}
                for q in auto_set:
                    i0 = _QORDER[q] * nt
                    rfb = rf_grid[:, :, :, i0:i0 + nt].reshape(nb, -1, nt)
                    # (obs-mod)/sig^2 * rf  summed over all data points
                    derivs[q] = (rfb * (dy / sig_flat).unsqueeze(-1)).sum(1)
                new_counts = {q: auto_node_count(
                    derivs[q], caps_snap[q], nt) for q in auto_set}
                if any(new_counts[q] != counts[q] for q in auto_set):
                    counts.update(new_counts)
                    names, node_idx, modes, caps, blocks = \
                        self._param_layout(counts, nt)
                    groups = _groups(blocks)
                    nparams = len(names)
                    # Re-extract the node values from the accepted
                    # atmosphere with the new layout and rebuild the
                    # trial (one extra forward step).
                    params = self._extract_nodes(ltau, atmos, blocks,
                                                 node_idx, nt)
                    # the best-so-far bookkeeping lives in the OLD layout:
                    # reset it to the accepted state in the NEW layout
                    # (after the re-extraction, so the shapes agree)
                    params_best = params.clone()
                    atmos_best = atmos.clone()
                    layout_best = dict(counts)
                    trial_atmos = self._atmos_from_params(
                        ltau, atmos, params, blocks, node_idx, nt, nb)
                    ymod = self.synthesis._forward(wavs, ltau, trial_atmos)
                    rf_grid = _grid_rf(trial_atmos)
                    ymod_flat = ymod.reshape(nb, -1)
                    dy = (target_flat - ymod_flat) / sig_flat

            # project the grid RF onto the nodes through the spline matrix
            jac = torch.zeros((nb, ymod.shape[1] * 4, nparams),
                              device=self.device, dtype=self.dtype)
            for q, (i0, i1) in blocks.items():
                if q in ("vmic", "vmac"):
                    col = 6 * nt + (0 if q == "vmic" else 1)
                    jac[:, :, i0] = rf_grid[:, :, :, col].reshape(nb, -1)
                else:
                    w = node_grid_weights(ltau, node_idx[i0:i1])
                    blk = rf_grid[:, :, :, _QORDER[q] * nt:
                                  (_QORDER[q] + 1) * nt]
                    jac[:, :, i0:i1] = blk.reshape(nb, -1, nt) @ w
            jac_w = jac / sig_flat[:, :, None]
            # Parity of the normal-equation scalings (the y-factor of the
            # reference code):
            #   T/Pe/B/vlos(mult.):  y = the node value (in the vof frame
            #                        for vlos)  -> per-unit da (a*(1+da))
            #   gamma/phi (additive): y = 1.0 (gam1/fi1 = 1)
            #                        AND the internal RFs are per-RADIAN
            #                        while spot's are per-DEGREE, so the
            #                        block must be scaled by 180/pi to put
            #                        the angle channels on the SAME
            #                        singular-value footing (whose
            #                        normal equations are per radian).
            #                        Without this factor the angle
            #                        channels sit sqrt(180/pi) too small
            #                        in alpha and get truncated by the
            #                        SVD tolerance, killing the whole
            #                        first step (verified against the
            #                        dumped alpha/beta at state2).
            scale = torch.ones_like(params)
            if scale.shape[1] != nparams:
                # guard against the batched auto-node reselection once
                # leaving params on the previous layout (jac/scale
                # 23-vs-21 desync observed in a batched cycle-4 run):
                # re-extract against the CURRENT layout so every
                # per-parameter structure (params/modes/caps) agrees
                # with nparams before the normal-equation scaling.
                params = self._extract_nodes(ltau, atmos, blocks,
                                             node_idx, nt)
                scale = torch.ones_like(params)
            for j, mode in enumerate(modes):
                if mode == "multiplicative":
                    scale[:, j] = params[:, j].abs().clamp(min=1e-30)
                elif GAMMA_PHI_QUIRK and names[j] in ("gamma", "phi"):
                    scale[:, j] = params[:, j].abs().clamp(min=1e-30)
                elif names[j] in ("gamma", "phi"):
                    scale[:, j] = _RAD_TO_DEG
            jac_w = jac_w * scale[:, None, :]

            alpha = torch.matmul(jac_w.transpose(-2, -1), jac_w)
            beta = torch.matmul(jac_w.transpose(-2, -1), dy.unsqueeze(-1)).squeeze(-1)
            # NOTE: the normal equations are NOT normalised by chi2 -- the
            # Marquardt damping lambda acts on the *absolute* curvature
            # alpha(j,j)*(1+lambda).

            delta = solve_damped_svd(alpha, beta, lamda, self.svd_tolerance,
                                     groups if self.group_truncation else None)
            # the angle delta is per-RADIAN (scale above); the parameters
            # are per-DEGREE -> convert in the update
            angle_factor = torch.where(
                torch.tensor([n in ("gamma", "phi") for n in names],
                             device=self.device),
                torch.tensor(_RAD_TO_DEG, device=self.device,
                             dtype=self.dtype),
                torch.ones(1, device=self.device, dtype=self.dtype))
            trial_params = marquardt_step(params, delta, modes, caps,
                                          additive_factor=angle_factor)
            # guard against non-finite parameter values (reject the step)
            bad_param = ~torch.isfinite(trial_params)
            trial_params = torch.where(bad_param, params, trial_params)
            trial_atmos2 = self._atmos_from_params(ltau, atmos, trial_params, blocks,
                                                   node_idx, nt, nb)
            chi2_try = self._chi2(wavs, ltau, target, sigma, trial_atmos2)
            improved = chi2_try < chi2_best

            if torch.all(~improved):
                # Damping increase on rejection (absolute-curvature
                # scaling): x100 below 1e-3, x10 below 1e3, x2 beyond.
                lamda = torch.where(
                    lamda <= 1e-3, lamda * 100.0,
                    torch.where(lamda < 1e3, lamda * 10.0, lamda * 2.0))
                lamda = lamda.clamp(max=self.lambda_max)
                rejected_streak = rejected_streak + 1
                # Restore the node counts of the accepted state
                # (mnodosold) on rejection
                counts = counts_accept
                names, node_idx, modes, caps, blocks = \
                    self._param_layout(counts, nt)
                params = self._extract_nodes(ltau, atmos, blocks, node_idx,
                                             nt)
                _record(False)
                if self.verbose and it % 5 == 0:
                    print(f"    iter {it:3d} chi2={chi2_best.mean().item():.4e}"
                          f" (rejected, lamda={lamda.mean().item():.2e})")
                # 7 consecutive rejections end the cycle.  Batch: end
                # only when EVERY column hit 7 in a row (a single
                # stalled column must not stop the others).
                if torch.all(lamda >= self.lambda_max) or \
                        torch.all(rejected_streak >= 7):
                    break
                continue

            # accept the improved batches
            params = torch.where(improved[:, None], trial_params, params)
            atmos = self._atmos_from_params(ltau, atmos, params, blocks, node_idx,
                                            nt, nb)
            chi2_new = torch.where(improved, chi2_try, chi2_best)
            better = chi2_try < chi2_best
            atmos_best = torch.where(better[:, None],
                                     trial_atmos2, atmos_best)
            params_best = torch.where(better[:, None], trial_params, params_best)
            layout_best = dict(counts)
            chi2_best = torch.where(better, chi2_try, chi2_best)
            # Damping decrease on acceptance: x0.1 while lambda > 1e-4,
            # x0.5 below (avoid shrinking into the noise floor).
            lamda = torch.where(
                improved, torch.where(lamda > 1e-4, lamda * 0.1, lamda * 0.5),
                lamda).clamp(min=1e-12)
            rejected_streak = torch.where(improved,
                                          torch.zeros_like(rejected_streak),
                                          rejected_streak)
            ngu = ngu + 1
            _record(True)
            if self.verbose:
                print(f"    iter {it:3d} chi2={chi2_best.mean().item():.4e}"
                      f" lamda={lamda.mean().item():.2e}")

            # convergence: varchi = (chi0-chisq)/(chi0+chisq) of
            # consecutive *accepted* iterations against a threshold that
            # relaxes with the iteration count:
            #   ngu < 25 : cotavar = exp((-42+ngu)/3) - 1e-4 (cap 3.3e-3)
            #   ngu >= 25: cotavar = (ngu-24)*3e-3
            varchi = (chi0 - chi2_best) / (chi0 + chi2_best)
            if ngu < 25:
                cotavar = min(float(np.exp((-42.0 + ngu) / 3.0)) - 1e-4,
                              3.3e-3)
            else:
                cotavar = (ngu - 24) * 3e-3
            if bool((varchi <= cotavar).all()):
                converged = True
                break
            chi0 = chi2_best

        # final atmosphere: the accepted best state (exact, no rebuild)
        atmos = atmos_best
        chi2 = chi2_best
        return atmos, chi2, history, iters, converged, layout_best

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _chi2(self, wavs, ltau, target, sigma, atmos):
        ymod = self.synthesis._forward(wavs, ltau, atmos)
        return chi_square(target.reshape(target.shape[0], -1),
                          ymod.reshape(ymod.shape[0], -1),
                          sigma.reshape(sigma.shape[0], -1))

    def _extract_nodes(self, ltau, atmos, blocks, node_idx, nt):
        """Initial node values.

        Direct sampling: the node value of a quantity is simply the grid
        value at the (equally spaced) node depth -- NOT a least-squares
        spline projection.  The LS projection used before changes every
        node value when the node set grows between cycles (natural spline
        spaces are not nested), which inflated the cycle-to-cycle chi2
        baseline; direct sampling keeps the accepted atmosphere unchanged
        at the shared node depths.
        """
        parts = []
        for q, (i0, i1) in blocks.items():
            grid = _qslice(atmos, q, nt)
            if q in ("vmic", "vmac"):
                # scalar quantity with a single node: the value itself
                parts.append(grid)
            else:
                if q == "vlos":
                    # The vlos node values (and the count-1 mean) are
                    # stored in the OFFSET frame vof = v - voffset with
                    #   voffset = vmin - 7.0 km/s  (per atmosphere),
                    # so that the multiplicative LM update is well
                    # defined near v=0.
                    vmin = grid.amin(dim=1, keepdim=True)
                    voffset = vmin - 7.0            # spot vlos in km/s
                    grid = grid - voffset
                idx = node_idx[i0:i1]
                if len(idx) == 1:
                    # m(i)==1: the single-node value is the MEAN over the
                    # whole grid ("constant perturbation"), NOT the
                    # mid-layer grid value (an earlier version took the
                    # mid-layer value; the resulting constant baseline
                    # differs by a factor ~2 for T and changed the whole
                    # single-node LM step).
                    parts.append(grid.mean(dim=1, keepdim=True))
                else:
                    parts.append(grid[:, idx])
        return torch.cat(parts, dim=1)

    def _atmos_from_params(self, ltau, atmos, params, blocks, node_idx, nt, nb):
        """Full atmosphere with the inverted quantities replaced by the
        spline of the node values; the rest (Pe when not inverted, etc.)
        is kept from the input atmosphere.

        When Pe is not inverted and ``config['inversion']['hse_pg0']`` is
        > 0 (a gas-pressure boundary condition), the Pe stratification is
        recomputed from hydrostatic equilibrium with the trial T, exactly
        as done inside the inversion loop: the SAME T produces a
        DIFFERENT (T-consistent) Pe, which changes the forward synthesis
        and therefore the inversion path.

        Built out-of-place (concatenation) so that it stays composable
        with torch.func.vmap/jacfwd (no in-place slice assignment).
        """
        cols = []
        hse_pe = "Pe" not in blocks and self.hse_pg0 > 0
        for q in ("T", "Pe", "B", "gamma", "phi", "vlos", "vmic", "vmac"):
            if q in blocks:
                i0, i1 = blocks[q]
                if q in ("vmic", "vmac"):
                    grid = params[:, i0:i1]      # scalar: single node value
                else:
                    # Representation: the trial atmosphere is the CURRENT
                    # atmosphere displaced by the spline of the node DELTA
                    # (y(j)=(atmosr(kred)/pert(kred))-1;
                    # atmos += splines22(y*pert)  <=>  atmos + W.(a - a_old)),
                    # NOT the absolute spline of the node values.  The two
                    # share the same first derivative (the spline weights W
                    # are linear), but the trials stay close to the current
                    # atmosphere (high acceptance) while an absolute
                    # spline can jump far away on coarse node sets.
                    cur = _qslice(atmos, q, nt)
                    nn = node_idx[i0:i1].shape[0]
                    if q == "vlos":
                        # Offset frame: vof = v - voffset (see
                        # _extract_nodes); the delta spline must be
                        # built in the same frame on both sides.
                        vmin = cur.amin(dim=1, keepdim=True)
                        cur_a = cur - (vmin - 7.0)
                    else:
                        cur_a = cur
                    if nn == 1:
                        cur_a = cur_a.mean(dim=1, keepdim=True)
                    else:
                        cur_a = cur_a[:, node_idx[i0:i1]]
                    grid = cur + (nodes_to_grid(ltau, params[:, i0:i1],
                                                node_idx[i0:i1])
                                  - nodes_to_grid(ltau, cur_a,
                                                  node_idx[i0:i1]))
            else:
                grid = _qslice(atmos, q, nt)     # fixed quantity: keep as is
            cols.append(grid)
        atmos_new = torch.cat(cols, dim=1)
        if hse_pe:
            # Recompute Pe from (trial T, pg0) with the input Pe only as
            # the first estimate.
            from ..physics.hydrostatic import hydrostatic_pe_cont
            t_grid = atmos_new[:, 0:nt].unsqueeze(-1)
            pe_grid = atmos_new[:, nt:2 * nt].unsqueeze(-1).clamp(min=1e-12)
            pe_hse, _pg = hydrostatic_pe_cont(
                ltau, t_grid, pe_grid, self.hse_pg0,
                abundance=self.synthesis.abundance)
            atmos_new = torch.cat(
                [atmos_new[:, :nt], pe_hse.squeeze(-1),
                 atmos_new[:, 2 * nt:]], dim=1)
        return atmos_new

    def _model_and_jacobian(self, wavs, ltau, atmos, names, blocks, node_idx,
                            nt, ymod=None):
        """
        Model profile and its Jacobian w.r.t. the *node* parameters.

        With ``config['synthesis']['rf_method'] == 'analytic'`` the
        variational response functions are computed once on
        the full grid and projected onto the nodes through the spline
        weights (much faster than one full forward per parameter);
        otherwise the Jacobian is obtained by autograd through the chain
        params -> spline -> atmos -> synthesis.

        ``ymod`` may be passed to reuse the forward already computed on
        ``atmos`` (in the LM loop the HSE-recomputed trial atmosphere is
        built twice otherwise; the HSE recompute is the expensive part
        when ``hse_pg0`` > 0).
        """
        nb = atmos.shape[0]
        nparams = len(names)

        def forward_nodes(p):
            a = self._atmos_from_params(ltau, atmos, p, blocks, node_idx, nt, nb)
            return self.synthesis._forward(wavs, ltau, a)   # (Nb,Nw,4)

        params = self._extract_nodes(ltau, atmos, blocks, node_idx, nt)
        if ymod is None:
            ymod = forward_nodes(params)

        if self.synthesis.config["synthesis"]["rf_method"] in ("analytic",
                                                                "analytic_chain",
                                                                "fast"):
            from .nodes import node_grid_weights
            # grid response functions dI/dx (Nb, Nw, 4, Nx), one solve
            pe_not_inverted = "Pe" not in blocks
            rf_method = self.synthesis.config["synthesis"]["rf_method"]
            if rf_method == "analytic_chain":
                rf_grid = self.synthesis._rf_analytic_chain(
                    wavs, ltau, atmos, pe_not_inverted=pe_not_inverted)
            elif rf_method == "fast":
                rf_grid = self.synthesis._rf_fast(
                    wavs, ltau, atmos, pe_not_inverted=pe_not_inverted)
            else:
                rf_grid = self.synthesis._rf_analytic(
                    wavs, ltau, atmos, pe_not_inverted=pe_not_inverted)
            # project onto the nodes: J[:, node] = sum_col RF[:,col]*W[col,node]
            jac = torch.zeros((nb, ymod.shape[1] * 4, nparams),
                              device=self.device, dtype=self.dtype)
            for q, (i0, i1) in blocks.items():
                if q in ("vmic", "vmac"):
                    # scalar quantity: single node == the column
                    col = 6 * nt + (0 if q == "vmic" else 1)
                    jac[:, :, i0] = rf_grid[:, :, :, col].reshape(
                        nb, -1)
                else:
                    w = node_grid_weights(ltau, node_idx[i0:i1])  # (Nt, Nn)
                    block = rf_grid[:, :, :, _QORDER[q] * nt:
                                    (_QORDER[q] + 1) * nt]       # (Nb,Nw,4,Nt)
                    # (Nb, Nw*4, Nt) @ (Nt, Nn) -> (Nb, Nw*4, Nn)
                    jac[:, :, i0:i1] = block.reshape(nb, -1, nt) @ w
            return ymod, jac

        try:
            from torch.func import jacfwd, vmap

            def single(p):
                return forward_nodes(p[None])[0]           # (Nw,4)

            # forward-mode AD: cost ~ Nparams forward passes instead of
            # Nw*4 backward passes (params << outputs here)
            jac = vmap(jacfwd(single))(params.detach().requires_grad_(True))
            # vmap(jacfwd) gives (Nb, Np, Nw, 4): move the parameter dim
            # LAST before flattening (a plain reshape interleaves the Np
            # axis into the wavelength blocks).
            jac = jac.permute(0, 2, 3, 1).reshape(nb, -1, nparams)
            # (Nb, Nw*4, Np)
        except Exception:
            jac = torch.zeros((nb, ymod.shape[1] * 4, nparams),
                              device=self.device, dtype=self.dtype)
            p = params.detach().clone().requires_grad_(True)
            out = forward_nodes(p)                          # (Nb,Nw,4)
            for i in range(nb):
                for iw in range(ymod.shape[1]):
                    for is_ in range(4):
                        g = torch.autograd.grad(out[i, iw, is_], p,
                                                retain_graph=True)[0][i]
                        jac[i, iw * 4 + is_] = g
        return ymod, jac

    def _compute_errors(self, wavs, ltau, target, sigma, atmos, nodes_cfg):
        """
        Parameter errors from the covariance matrix at the solution.

        sigma_p^2 = chi2_reduced * diag( (J^T J)^-1 )  via SVD.
        """
        nt = ltau.shape[0]
        nb = target.shape[0]
        names, node_idx, modes, caps, blocks = self._param_layout(nodes_cfg, nt)
        nparams = len(names)
        errors = torch.zeros_like(atmos)
        if nparams == 0:
            return errors

        params = self._extract_nodes(ltau, atmos, blocks, node_idx, nt)
        ymod, jac = self._model_and_jacobian(wavs, ltau, atmos, names, blocks,
                                             node_idx, nt)
        sig_flat = sigma.reshape(nb, -1)
        jac_w = jac / sig_flat[:, :, None]
        alpha = torch.matmul(jac_w.transpose(-2, -1), jac_w)

        chi2 = chi_square(target.reshape(nb, -1), ymod.reshape(nb, -1),
                          sig_flat)
        dof = max(int(target.shape[1] * 4) - nparams, 1)
        chi2_red = chi2 / dof

        u, s, vh = torch.linalg.svd(alpha, full_matrices=False)
        smax = s[:, :1].clamp(min=1e-30)
        s_inv = torch.where(s <= self.svd_tolerance * smax,
                            torch.zeros_like(s), 1.0 / s)
        # diag of V diag(1/s^2) V^T
        cov_diag = (vh.transpose(-2, -1) ** 2 * (s_inv ** 2)[:, None, :]).sum(-1)
        node_err = torch.sqrt((chi2_red[:, None] * cov_diag).clamp(min=0.0))

        # propagate node errors to the grid through the spline
        from .nodes import node_grid_weights
        for q, (i0, i1) in blocks.items():
            if q in ("vmic", "vmac"):
                # scalar quantity: the grid error is the node error itself
                _set_qslice(errors, q, nt, node_err[:, i0:i1])
                continue
            w = node_grid_weights(ltau, node_idx[i0:i1])   # (Nt, Nn)
            # grid error = sqrt( sum_j W_ij^2 * node_err_j^2 )
            w2 = w ** 2
            err2 = w2 @ (node_err[:, i0:i1] ** 2).t()      # (Nt, Nb)
            _set_qslice(errors, q, nt, torch.sqrt(err2.t()))
        return errors


# ---------------------------------------------------------------------------
# quantity <-> column mapping helpers
# ---------------------------------------------------------------------------
_QORDER = {"T": 0, "Pe": 1, "B": 2, "gamma": 3, "phi": 4, "vlos": 5,
           "vmic": 6, "vmac": 7}


def _qslice(atmos, q, nt):
    """Columns of quantity q: (Nb, Nt) or (Nb, 1)."""
    o = _QORDER[q]
    if q in ("vmic", "vmac"):
        return atmos[:, 6 * nt + (0 if q == "vmic" else 1):][:, :1]
    return atmos[:, o * nt:(o + 1) * nt]


def _set_qslice(atmos, q, nt, values):
    o = _QORDER[q]
    if q in ("vmic", "vmac"):
        atmos[:, 6 * nt + (0 if q == "vmic" else 1):6 * nt + 2] = values
    else:
        atmos[:, o * nt:(o + 1) * nt] = values
