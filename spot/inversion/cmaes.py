# -*- coding: utf-8 -*-
"""
spot.inversion.cmaes — CMA-ES batched inversion (derivative-free)
===================================================================
A sibling solver of :class:`spot.inversion.Inversion` (Levenberg-
Marquardt).  It replaces the LM machinery with the Covariance Matrix
Adaptation Evolution Strategy (Hansen, N. 2016, arXiv:1604.00772, "The
CMA Evolution Strategy: A Tutorial"), which is *derivative-free*: the
only model evaluation is the forward synthesis, so **no response
functions are computed at all**.

Design properties
-----------------
* ``CmaesInversion`` subclasses :class:`Inversion` and reuses its
  configuration, node layout, sigma and cycle machinery, so the two
  solvers are drop-in interchangeable for the same config and can be
  used side by side without interfering with each other.
* The search space is the *node values* of the same node layout and the
  same delta representation as the LM solver (``_extract_nodes`` /
  ``_atmos_from_params``: the trial atmosphere is the current one
  displaced by the spline of the node delta), so both solvers walk the
  same parameter space; the initial state is the raw initial guess with
  a zero delta (exactly like the LM first trial).
* Fully vectorized and batched: :math:`N_b` independent inversions are
  solved in parallel.  Each batch column runs its OWN CMA-ES instance
  (its own mean / covariance / step size / evolution paths); every
  generation all :math:`N_b \\times \\lambda` offspring are synthesized
  in a single batched forward call, so the GPU batch parallelism is
  used exactly like in the synthesis itself.
* The CMA-ES state lives in a per-quantity normalized frame
  ``x = node_value / scale[quantity]`` so that the covariance matrix is
  well conditioned.  Hard physical bounds are supported through a
  quadratic penalty (no clipping of the search distribution).
* Same cycle progression: the node set grows cycle after
  cycle and each cycle restarts CMA-ES from the best atmosphere found
  so far (an elitist best point is tracked per column, so chi2 can
  never worsen).  Automatic nodes (``'*'``) are resolved to their
  *caps* (the maximum counts) and kept fixed — the dynamic recount
  needs response functions, which CMA-ES deliberately does not compute.

Configuration: the optional ``"cmaes"`` section of the config dict
(merged over :data:`spot.default.DEFAULT_CONFIG`) holds the strategy
parameters; everything else (nodes, cycles, sigma, weights, HSE, ...)
is shared with the LM solver.

Example
-------
::

    inv = CmaesInversion(cfg)      # cfg identical to the LM config
    res = inv.invert(wavs, ltau, target, initial="hot11")
    # res.atmos / res.chi2 / res.history — same container as Inversion
"""

import math
import time

import numpy as np
import torch

from .inversion import Inversion, InversionResult, _qslice, _set_qslice
from .marquardt import chi_square
from .nodes import nodes_to_grid

__all__ = ["CmaesInversion"]

# default per-quantity characteristic scales beta (x = value / beta) and
# physical bounds (quadratic penalty outside).  Both are configurable via
# config["cmaes"]["scale"] / config["cmaes"]["bounds"].
_DEFAULT_SCALE = {"T": 1000.0, "Pe": 1.0, "B": 500.0, "gamma": 30.0,
                  "phi": 60.0, "vlos": 1.0, "vmic": 1.0, "vmac": 1.0}
_DEFAULT_BOUNDS = {"T": [500.0, 30000.0], "Pe": [1e-8, 1e5],
                   "B": [-1000.0, 10000.0], "gamma": [0.0, 180.0],
                   "phi": [0.0, 360.0], "vlos": [-10.0, 10.0],
                   "vmic": [0.0, 10.0], "vmac": [-10.0, 10.0]}

# the CMA-ES bookkeeping always runs in float64 (small tensors; the
# covariance eigendecomposition and the step-size adaptation profit from
# the precision) even when the forward synthesis is float32.
_STATE_DTYPE = torch.float64


def _default_population(n):
    """Standard CMA-ES population size: 4 + floor(3 ln n), min 5."""
    return max(4 + int(3.0 * math.log(max(n, 2))), 5)


def _cma_strategy_params(n, lam):
    """Strategy parameters from Hansen (2016), Algorithm 1."""
    mu = max(lam // 2, 1)
    w_raw = [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)]
    w_sum = sum(w_raw)
    weights = [w / w_sum for w in w_raw]
    mu_eff = 1.0 / sum(w * w for w in weights)
    c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
    d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0)
    d_sigma += c_sigma
    c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
    c1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
    cmu = min(1.0 - c1,
              2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff))
    chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
    return dict(lam=lam, mu=mu, weights=weights, mu_eff=mu_eff,
                c_sigma=c_sigma, d_sigma=d_sigma, c_c=c_c, c1=c1, cmu=cmu,
                chi_n=chi_n)


class CmaesInversion(Inversion):
    """
    Batched derivative-free inversion with CMA-ES (Hansen 2016).

    Same constructor/config contract as :class:`Inversion`; the extra
    ``config['cmaes']`` section tunes the evolution strategy:

    ================  ==================================================
    key               meaning
    ================  ==================================================
    population        population size lambda (None -> 4 + floor(3 ln n))
    sigma0            initial step size in the normalized frame
    sigma0_cycles     per-cycle initial step sizes (coarse-to-fine
                      schedule, e.g. [1.5, 0.6, 0.3, 0.1]; None -> sigma0)
    min_sigma         step-size floor per column
    max_sigma         step-size cap
    scale             per-quantity scale beta (x = value / beta)
    bounds            per-quantity physical bounds (None/{} disables)
    penalty           quadratic penalty weight for bound violations
    covariance_mode   'full' (default) or 'sep' (diagonal covariance)
    stall             per-column generations without improvement -> done
    tol_fun           relative chi2 improvement regarded as improvement
    seed              RNG seed (None -> random)
    verbose           print per-generation progress
    eval_chunk        max rows per batched forward (None = all at once)
    hse_refresh       freeze the electron pressure (HSE Pe) during the
                      search — None (default): exact model, HSE Pe
                      recomputed for every candidate; int K >= 1: the
                      HSE-consistent Pe is recomputed once every K
                      generations (from the current best atmosphere) and
                      frozen over the span — a large speed-up when the
                      hydrostatic Pe recomputation dominates the forward
                      cost.  The delivered atmosphere/chi2 always use the
                      exact HSE model.
    ================  ==================================================
    """

    def __init__(self, config=None):
        super().__init__(config)
        cm = self.config.get("cmaes", {}) or {}
        self.cma_population = cm.get("population", None)
        self.cma_sigma0 = float(cm.get("sigma0", 1.0))
        self.cma_sigma0_cycles = cm.get("sigma0_cycles", None)
        if self.cma_sigma0_cycles is not None:
            # per-cycle initial step size (coarse-to-fine schedule)
            self.cma_sigma0_cycles = [float(v)
                                      for v in list(self.cma_sigma0_cycles)]
        self.cma_min_sigma = float(cm.get("min_sigma", 1e-8))
        self.cma_max_sigma = float(cm.get("max_sigma", 1e4))
        self.cma_scale = dict(_DEFAULT_SCALE, **(cm.get("scale", {}) or {}))
        self.cma_bounds = dict(_DEFAULT_BOUNDS, **(cm.get("bounds", {}) or {}))
        self.cma_penalty = float(cm.get("penalty", 1e12))
        self.cma_mode = str(cm.get("covariance_mode", "full")).lower()
        self.cma_stall = int(cm.get("stall", 5))
        self.cma_tol_fun = float(cm.get("tol_fun", 1e-8))
        self.cma_verbose = bool(cm.get("verbose", True))
        self.cma_seed = cm.get("seed", None)
        self.cma_eval_chunk = cm.get("eval_chunk", None)
        # HSE-Pe refresh interval (freeze the electron pressure between
        # refreshes): None = exact model (HSE Pe recomputed per candidate,
        # the LM-identical behavior); int K >= 1 = recompute the HSE-consistent
        # Pe once every K generations (from the current best atmosphere) and
        # FREEZE it for every candidate of the frozen span.  The forward is
        # then the true trial model only at the refresh points — a
        # derivative-free optimizer tolerates this well — and the final
        # atmosphere is always rebuilt with the exact (HSE) model.
        self.cma_hse_refresh = cm.get("hse_refresh", None)
        if self.cma_hse_refresh is not None:
            self.cma_hse_refresh = int(self.cma_hse_refresh)
            if self.cma_hse_refresh < 1:
                raise ValueError("cmaes.hse_refresh must be >= 1 or None")
        if self.cma_mode not in ("full", "sep"):
            raise ValueError("cmaes.covariance_mode must be 'full' or 'sep'")

    # ------------------------------------------------------------------
    # public API (same signature as Inversion.invert)
    # ------------------------------------------------------------------
    def invert(self, wavs, ltau, target, initial=None, sigma=None):
        """
        Batched CMA-ES inversion of a batch of Stokes profiles.

        The interface is identical to :meth:`Inversion.invert`; the
        search uses only forward evaluations (no response functions).
        """
        wavs = torch.as_tensor(wavs, dtype=self.dtype, device=self.device)
        ltau = torch.as_tensor(ltau, dtype=self.dtype, device=self.device)
        target = torch.as_tensor(target, dtype=self.dtype, device=self.device)
        if target.ndim == 2:
            target = target.unsqueeze(0)
        nb = target.shape[0]
        nt = ltau.shape[0]

        if initial is None:
            initial = self.config["atmosphere"]
        atmos = self.synthesis._normalize_atmos(initial, ltau)
        if atmos.shape[0] == 1 and nb > 1:
            atmos = atmos.repeat(nb, 1)
        if atmos.shape[0] != nb:
            raise ValueError(
                f"initial guess batch {atmos.shape[0]} != target batch {nb}")

        # NOTE: _wavs_cache is filled after _resolve_sigma exactly like
        # Inversion.invert, so the reference sigma convention follows the
        # identical resolution path.
        sigma = self._resolve_sigma(sigma, target)
        # normalize the sigma tensor to (Nb, Nw, 4) so the population
        # evaluation can expand it over the lambda offspring directly
        nw = target.shape[1]
        if sigma.ndim == 4:                        # (Nb,1,1,1) scalar tensor
            sigma = sigma.squeeze(1).squeeze(1)    # -> (Nb,1)
        if sigma.ndim == 3 and sigma.shape[1] == 4:   # (Nb,4,Nw) -> (Nb,Nw,4)
            sigma = sigma.permute(0, 2, 1).contiguous()
        if sigma.ndim == 2:                        # (Nb,1)
            sigma = sigma.reshape(nb, 1, 1)
        if tuple(sigma.shape[1:]) != (nw, 4):      # (Nb,1,1) -> full grid
            sigma = sigma.expand(nb, nw, 4).contiguous()
        self._wavs_cache = wavs

        if self.verbose:
            print(f"[cmaes] device={self.device} dtype={self.dtype} "
                  f"nb={nb}, solver=CMA-ES (no response functions)",
                  flush=True)
        t0 = time.time()
        if self.cma_seed is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = int(self.cma_seed)
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(seed)

        nodes_cfg = self._cycle_nodes()
        n_cycles = min(len(nodes_cfg), self.max_cycles)
        history = []
        best_atmos = None
        best_chi2 = None
        final_counts = None
        all_converged = True

        for icycle in range(n_cycles):
            n_cfg, auto_set = nodes_cfg[icycle]
            atmos, chi2, hist, iters, converged, counts = \
                self._run_cycle_cma(wavs, ltau, target, sigma, atmos,
                                    n_cfg, auto_set, icycle)
            history.extend(hist)
            all_converged = all_converged and converged
            final_counts = counts
            if best_chi2 is None:
                best_chi2 = chi2.clone()
                best_atmos = atmos.clone()
            else:
                # per-column best over the cycles (each column is an
                # independent optimization process)
                better = chi2 < best_chi2
                best_chi2 = torch.where(better, chi2, best_chi2)
                best_atmos = torch.where(better[:, None], atmos, best_atmos)
            if self.verbose:
                print(f"[cmaes cycle {icycle + 1}/{n_cycles}] chi2 = "
                      f"{chi2.mean().item():.4e}  ({iters} gens)",
                      flush=True)

        errors = None
        if self.compute_errors and final_counts is not None:
            # optional: parameter errors come from the RESPONSE functions
            # (covariance at the solution, as in the LM solver) — computed
            # only when explicitly requested via config['inversion']['compute_errors']
            errors = self._compute_errors(wavs, ltau, target, sigma,
                                          best_atmos, final_counts)

        return InversionResult(
            atmos=best_atmos, chi2=best_chi2, errors=errors, history=history,
            cycles=n_cycles, iterations=len(history), converged=all_converged,
            runtime=time.time() - t0)

    # ------------------------------------------------------------------
    # one CMA-ES cycle
    # ------------------------------------------------------------------
    def _run_cycle_cma(self, wavs, ltau, target, sigma, atmos, nodes_cfg,
                       auto_set, icycle):
        """
        One CMA-ES cycle with a fixed node layout.

        Returns (atmos, chi2, history, generations, converged, counts).
        """
        nt = ltau.shape[0]
        nb = target.shape[0]

        # --- node layout (automatic nodes -> caps, kept fixed) ---------
        counts = self._static_counts(nodes_cfg, auto_set, wavs.shape[0])
        names, node_idx, modes, caps, blocks = self._param_layout(counts, nt)
        nparams = len(names)
        if nparams == 0:
            chi2 = self._eval_params(wavs, ltau, target, sigma, atmos,
                                     p0=None, blocks=blocks, node_idx=node_idx,
                                     nt=nt, nb=nb)
            return atmos, chi2, [], 0, True, counts

        # --- parameter frame: x = node_value / beta ---------------------
        scale, lo_p, hi_p = self._param_frame(blocks, nparams)
        beta64 = torch.as_tensor(scale, dtype=_STATE_DTYPE, device=self.device)
        # the vlos nodes live in the offset frame vof = v - (vmin - 7);
        # the physical bounds must be expressed in the same frame (the
        # offset is per column, so the bounds become per-column boxes)
        vmin = self._vlos_vmin(ltau, atmos, blocks, nt)    # (nb,)
        lo_p64 = torch.as_tensor(lo_p, dtype=_STATE_DTYPE, device=self.device)
        hi_p64 = torch.as_tensor(hi_p, dtype=_STATE_DTYPE, device=self.device)
        if self.cma_bounds:
            vof_sh = torch.zeros(nb, nparams, dtype=_STATE_DTYPE,
                                 device=self.device)
            for q, (i0, i1) in blocks.items():
                if q == "vlos":
                    vof_sh[:, i0:i1] = 7.0 - vmin[:, None]
            lo_x = (lo_p64 + vof_sh) / beta64        # (nb, nparam)
            hi_x = (hi_p64 + vof_sh) / beta64
            has_bnd = torch.isfinite(lo_x[0]) & torch.isfinite(hi_x[0])
        else:
            lo_x = torch.full((nb, nparams), -1e300, dtype=_STATE_DTYPE,
                              device=self.device)
            hi_x = torch.full((nb, nparams), 1e300, dtype=_STATE_DTYPE,
                              device=self.device)
            has_bnd = torch.zeros(nparams, dtype=torch.bool,
                                  device=self.device)
        # baseline node values in the LM frame (vof for vlos)
        p0 = self._extract_nodes(ltau, atmos, blocks, node_idx, nt)

        # --- frozen-Pe mode (user speed-up option) ------------------------
        # builder: atmosphere construction used for candidate evaluation
        #   exact  -> _atmos_from_params (HSE Pe per candidate, LM-identical)
        #   frozen -> _atmos_from_params_frozen (Pe fixed from `ref_base`)
        # ref_base: the per-column reference atmosphere whose Pe columns are
        # frozen; built/refreshed with the EXACT HSE model at the start of
        # every span of `hse_refresh` generations (and at the cycle end,
        # the final atmosphere is rebuilt exactly).
        frozen = self.cma_hse_refresh is not None
        Kh = self.cma_hse_refresh if frozen else 0
        builder = (self._atmos_from_params_frozen if frozen
                   else self._atmos_from_params)
        ref_base = None
        if frozen:
            # initial refresh: the HSE-consistent Pe of the cycle baseline
            # (the baseline point then evaluates EXACTLY as before)
            ref_base = self._atmos_from_params(ltau, atmos, p0, blocks,
                                               node_idx, nt, nb)
            if self.cma_verbose:
                print(f"    [cmaes] hse_refresh={Kh} — HSE Pe frozen "
                      f"between refreshes", flush=True)

        # --- strategy parameters ----------------------------------------
        lam = int(self.cma_population) if self.cma_population else \
            _default_population(nparams)
        sp = _cma_strategy_params(nparams, lam)
        mu = sp["mu"]
        weights = torch.tensor(sp["weights"], dtype=_STATE_DTYPE,
                               device=self.device)
        c_sigma, d_sigma, c_c, c1, cmu = (sp["c_sigma"], sp["d_sigma"],
                                          sp["c_c"], sp["c1"], sp["cmu"])
        chi_n = sp["chi_n"]
        sqrt_cc = math.sqrt(c_c * (2.0 - c_c) * sp["mu_eff"])
        sqrt_cs = math.sqrt(c_sigma * (2.0 - c_sigma) * sp["mu_eff"])

        # --- state (float64, per column) --------------------------------
        x_mean = (p0.to(dtype=_STATE_DTYPE) / beta64)
        if self.cma_sigma0_cycles is not None \
                and icycle < len(self.cma_sigma0_cycles):
            s0 = self.cma_sigma0_cycles[icycle]
        else:
            s0 = self.cma_sigma0
        step = torch.full((nb,), s0, dtype=_STATE_DTYPE, device=self.device)
        if self.cma_mode == "full":
            C = torch.eye(nparams, dtype=_STATE_DTYPE, device=self.device)
            C = C.unsqueeze(0).expand(nb, nparams, nparams).contiguous()
        else:
            C = torch.ones(nb, nparams, dtype=_STATE_DTYPE,
                           device=self.device)
        pc = torch.zeros(nb, nparams, dtype=_STATE_DTYPE, device=self.device)
        ps = torch.zeros(nb, nparams, dtype=_STATE_DTYPE, device=self.device)

        # elitist best point: the cycle baseline itself (zero delta), so
        # chi2 can never worsen for any column
        chi2_0 = self._eval_params(wavs, ltau, target, sigma,
                                   ref_base if frozen else atmos, p0=p0,
                                   blocks=blocks, node_idx=node_idx, nt=nt,
                                   nb=nb, frozen=frozen)
        best_x = x_mean.clone()
        best_f = chi2_0.to(dtype=_STATE_DTYPE).clone()
        stall = torch.zeros(nb, dtype=_STATE_DTYPE, device=self.device)
        done = torch.zeros(nb, dtype=torch.bool, device=self.device)

        history = []
        iters = 0
        converged = False

        if self.cma_verbose:
            print(f"  [cmaes cycle {icycle + 1}] nodes={dict(counts)} "
                  f"nparams={nparams} lam={lam} "
                  f"chi2(init)={chi2_0.mean().item():.4e}", flush=True)

        def _record(gen):
            rec = {
                "cycle": icycle, "iteration": gen,
                "chi2": float(best_f.mean().item()),
                "sigma": float(step.mean().item()),
                "accepted": True,
                "nparams": nparams,
                "nodes": dict(counts),
            }
            if gen == 0:
                rec["chi2_col"] = [float(v) for v in chi2_0.cpu()]
            history.append(rec)

        _record(0)

        def _exact_refresh():
            """Re-log the tracked best with the EXACT (HSE) model.

            Returns True if a column had to be restored to the last
            finite best (its frozen-objective best failed the exact
            evaluation — possible when the frozen Pe lets the search
            wander into HSE-unstable states).  The frozen Pe columns are
            refreshed from the exact atmosphere of the good columns.
            """
            nonlocal best_x, best_f, best_x_ok, best_f_ok, ref_base, stall
            bp = (best_x * beta64).to(dtype=self.dtype)
            full = self._atmos_from_params(ltau, atmos, bp, blocks,
                                           node_idx, nt, nb)
            y = self.synthesis._forward(wavs, ltau, full)
            c_true = chi_square(target.reshape(nb, -1),
                                y.reshape(nb, -1),
                                sigma.reshape(nb, -1)) \
                .to(dtype=_STATE_DTYPE)
            fin = torch.isfinite(c_true)
            bad = ~fin
            best_x = torch.where(bad[:, None], best_x_ok, best_x)
            best_f = torch.where(bad, best_f_ok, c_true)
            best_x_ok = best_x.clone()
            best_f_ok = best_f.clone()
            new_pe = _qslice(full, "Pe", nt)
            old_pe = _qslice(ref_base, "Pe", nt)
            _set_qslice(ref_base, "Pe", nt,
                        torch.where(fin[:, None], new_pe, old_pe))
            stall = torch.zeros_like(stall)
            if bool(bad.any()) and self.cma_verbose:
                tb = _qslice(full, "T", nt)[bad]
                print(f"      [cmaes] exact re-evaluation failed for "
                      f"{int(bad.sum().item())}/{nb} column(s); restored "
                      f"last finite best (bad T range "
                      f"{float(tb.min()):.1f}..{float(tb.max()):.1f} K)",
                      flush=True)
            return bool(bad.any())

        # the per-column best that passed the exact model (init: the
        # cycle baseline, whose exact chi2 is chi2_0)
        best_x_ok = x_mean.clone()
        best_f_ok = chi2_0.to(dtype=_STATE_DTYPE).clone()

        for gen in range(self.max_iterations):
            iters = gen + 1
            act = ~done
            if not bool(act.any()):
                break
            idx_a = act.nonzero(as_tuple=False).squeeze(-1)
            a = idx_a.shape[0]
            m_a = x_mean[idx_a]
            s_a = step[idx_a]

            # 0. periodic HSE refresh: recompute the frozen Pe from the
            # current best atmosphere and re-log the tracked best with
            # the exact (HSE) model
            if frozen and gen > 0 and gen % Kh == 0:
                _exact_refresh()
                if self.cma_verbose:
                    print(f"    [cmaes] gen {gen}: HSE Pe refreshed "
                          f"(chi2={best_f.mean().item():.4e})", flush=True)

            # 1. sample the population: x = m + sigma * B D z  (C = B D^2 B')
            z = torch.randn(a, lam, nparams, generator=self._gen,
                            device=self.device).to(dtype=_STATE_DTYPE)
            if self.cma_mode == "full":
                evals, evecs = torch.linalg.eigh(C[idx_a])
                d = evals.sqrt().clamp(min=1e-14)          # (a,n)
                y = torch.einsum("aij,akj->aki", evecs, d.unsqueeze(1) * z)
                inv_d = 1.0 / d
            else:
                sd = torch.sqrt(C[idx_a].clamp(min=1e-30))
                y = sd * z
                inv_d = 1.0 / sd
            X = m_a.unsqueeze(1) + s_a[:, None, None] * y   # (a,lam,n)

            # 2. evaluate every offspring (one batched forward call)
            P = (X * beta64).to(dtype=self.dtype)
            chi2_raw, chi2_pen = self._eval_population(
                wavs, ltau, target, sigma, ref_base if frozen else atmos,
                P, blocks, node_idx, nt, nb, idx_a, beta64, lo_x[idx_a],
                hi_x[idx_a], has_bnd, frozen=frozen)

            # 3. sort by penalized fitness
            f_sort, sort_idx = torch.sort(chi2_pen, dim=1)
            x_sort = X.gather(1, sort_idx[:, :, None]
                              .expand(a, lam, nparams))
            gen_best_raw = chi2_raw.gather(1, sort_idx[:, :1]).squeeze(1)

            # 4. elitist best tracking (per column, non-increasing)
            imp = (best_f[idx_a] - gen_best_raw) > \
                self.cma_tol_fun * best_f[idx_a].abs().clamp(min=1.0)
            best_f[idx_a] = torch.where(imp, gen_best_raw, best_f[idx_a])
            best_x[idx_a] = torch.where(imp[:, None], x_sort[:, 0],
                                        best_x[idx_a])
            stall[idx_a] = torch.where(imp, torch.zeros_like(stall[idx_a]),
                                       stall[idx_a] + 1.0)

            # 5. recombination + evolution paths (Hansen alg. 1)
            x_mu = x_sort[:, :mu]                          # (a,mu,n)
            y_mu = (x_mu - m_a.unsqueeze(1)) / s_a[:, None, None]
            x_mean_new = (x_mu * weights[None, :, None]).sum(dim=1)
            y_w = (y_mu * weights[None, :, None]).sum(dim=1)   # (a,n)
            if self.cma_mode == "full":
                # C^{-1/2} y_w = B D^{-1} B' y_w
                y_w_inv = torch.einsum(
                    "aij,aj->ai", evecs,
                    torch.einsum("aij,aj->ai", evecs, y_w) * inv_d)
            else:
                y_w_inv = y_w * inv_d
            ps[idx_a] = (1.0 - c_sigma) * ps[idx_a] + sqrt_cs * y_w_inv
            pc[idx_a] = (1.0 - c_c) * pc[idx_a] + sqrt_cc * y_w

            # 6. step-size adaptation (CSA) + h_sigma correction
            ps_norm = torch.linalg.norm(ps[idx_a], dim=1)
            expo = math.sqrt(1.0 - (1.0 - c_sigma) ** (2.0 * (gen + 1)))
            h_sig = (ps_norm / expo) < (1.4 + 2.0 / (nparams + 1)) * chi_n
            s_new = s_a * torch.exp(
                (c_sigma / d_sigma) * (ps_norm / chi_n - 1.0))
            s_new = torch.nan_to_num(s_new, nan=self.cma_min_sigma,
                                     posinf=self.cma_max_sigma,
                                     neginf=self.cma_min_sigma)
            step[idx_a] = s_new.clamp(self.cma_min_sigma, self.cma_max_sigma)

            # 7. covariance update (rank-1 + rank-mu)
            pc_a = pc[idx_a]
            rank1 = torch.einsum("ai,aj->aij", pc_a, pc_a)
            h_f = h_sig.to(_STATE_DTYPE)
            if self.cma_mode == "full":
                C_a = ((1.0 - c1 - cmu) * C[idx_a]
                       + cmu * torch.einsum("ami,amj->aij",
                                            y_mu * weights[None, :, None],
                                            y_mu)
                       + c1 * (rank1
                               + (1.0 - h_f)[:, None, None]
                               * c_c * (2.0 - c_c) * C[idx_a]))
                # SPD-ify: symmetrize + clamp the eigenvalues
                C_a = 0.5 * (C_a + C_a.mT)
                ev, evec = torch.linalg.eigh(C_a)
                C[idx_a] = torch.einsum("aij,aj,akj->aik", evec,
                                        ev.clamp(min=1e-14), evec)
            else:
                C[idx_a] = ((1.0 - c1 - cmu) * C[idx_a]
                            + cmu * (y_mu * weights[None, :, None]).pow(2)
                            .sum(dim=1)
                            + c1 * (pc_a * pc_a
                                    + (1.0 - h_f)[:, None]
                                    * c_c * (2.0 - c_c) * C[idx_a]))
            x_mean[idx_a] = x_mean_new

            # 8. per-column stopping
            done[idx_a] = done[idx_a] | (stall[idx_a] >= self.cma_stall) \
                | (step[idx_a] < self.cma_min_sigma)
            _record(gen + 1)
            if self.cma_verbose and (gen < 4 or (gen + 1) % 10 == 0):
                print(f"    gen {gen + 1:3d} chi2={best_f.mean().item():.4e} "
                      f"sigma={step.mean().item():.3e} "
                      f"done={int(done.sum().item())}/{nb}", flush=True)
            if bool(done.all()):
                converged = True
                break

        # final atmosphere: delta-frame rebuild from the best points
        # with the EXACT model; the reported chi2 is the exact-model chi2
        # of this delivered atmosphere (also correct in frozen mode, where
        # one last exact refresh guarantees the best is HSE-finite)
        if frozen:
            _exact_refresh()
        params_best = (best_x * beta64).to(dtype=self.dtype)
        atmos_best = self._atmos_from_params(ltau, atmos, params_best,
                                             blocks, node_idx, nt, nb)
        chi2_best = best_f.to(dtype=self.dtype)
        return atmos_best, chi2_best, history, iters, converged, counts

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _static_counts(self, nodes_cfg, auto_set, nw):
        """Fixed node counts of a cycle; automatic quantities take the cap
        counts (the same maxima the LM cycle starts from), since the
        dynamic recount used by the LM solver needs response functions."""
        counts = {}
        if auto_set:
            n_auto = len(auto_set)
            n_det = sum(int(max(v, 0)) for q, v in nodes_cfg.items()
                        if q not in auto_set)
            nfrecs = nw * sum(1 for w in self.stokes_weights
                              if float(w) != 0.0)
            nmax1 = int(round((min(200, nfrecs) - n_det) / (n_auto + 2)))
            for q in auto_set:
                cap = 2 * nmax1 if q == "T" else nmax1
                counts[q] = max(int(cap), 1)
        for q, v in nodes_cfg.items():
            if q not in auto_set:
                counts[q] = int(v)
        return counts

    def _param_frame(self, blocks, nparams):
        """Per-parameter scale (beta) and PHYSICAL bound box.

        Returns (scale (n,), lo_p (n,), hi_p (n,)).  (The vlos bounds are
        converted into the offset frame per column by the caller.)
        """
        scale = np.zeros(nparams)
        lo = np.full(nparams, -np.inf)
        hi = np.full(nparams, np.inf)
        for q, (i0, i1) in blocks.items():
            s = float(self.cma_scale.get(q, 1.0))
            scale[i0:i1] = s
            if self.cma_bounds and q in self.cma_bounds:
                blo, bhi = self.cma_bounds[q]
                lo[i0:i1] = blo
                hi[i0:i1] = bhi
        return scale, lo, hi

    @staticmethod
    def _vlos_vmin(ltau, atmos, blocks, nt):
        """Per-column minimum LOS velocity of the baseline atmosphere
        (needed to map the vlos bounds into the offset frame)."""
        if "vlos" not in blocks:
            return None
        grid = _qslice(atmos, "vlos", nt)          # (nb, Nt) km/s
        return grid.amin(dim=1).to(dtype=_STATE_DTYPE)

    def _atmos_from_params_frozen(self, ltau, atmos, params, blocks,
                                  node_idx, nt, nb):
        """Inversion._atmos_from_params WITHOUT the HSE-Pe recompute.

        The Pe stratification is taken from ``atmos`` (the frozen
        reference), everything else is identical to the delta
        representation of the LM solver — used when ``cmaes.hse_refresh``
        is set.
        """
        cols = []
        for q in ("T", "Pe", "B", "gamma", "phi", "vlos", "vmic", "vmac"):
            if q in blocks:
                i0, i1 = blocks[q]
                if q in ("vmic", "vmac"):
                    grid = params[:, i0:i1]      # scalar column
                else:
                    cur = _qslice(atmos, q, nt)
                    nn = node_idx[i0:i1].shape[0]
                    if q == "vlos":
                        # offset frame (identical to _atmos_from_params)
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
                grid = _qslice(atmos, q, nt)      # fixed quantity: keep
            cols.append(grid)
        return torch.cat(cols, dim=1)

    def _eval_params(self, wavs, ltau, target, sigma, base, p0, blocks,
                     node_idx, nt, nb, frozen=False):
        """chi2 (nb,) of one parameter-vector evaluation (no population)."""
        if p0 is None:
            p0 = self._extract_nodes(ltau, base, blocks, node_idx, nt)
        builder = self._atmos_from_params_frozen if frozen \
            else self._atmos_from_params
        trial = builder(ltau, base, p0, blocks, node_idx, nt, nb)
        ymod = self.synthesis._forward(wavs, ltau, trial)
        sig = sigma.reshape(nb, -1)
        return chi_square(target.reshape(nb, -1), ymod.reshape(nb, -1), sig)

    def _eval_population(self, wavs, ltau, target, sigma, base, P, blocks,
                         node_idx, nt, nb, idx_a, beta_a, lo_a, hi_a,
                         has_bnd, frozen=False):
        """Forward + chi2 of a population P (a, lam, nparams).

        ``base`` is the per-column reference atmosphere ((nb, nx); in
        frozen mode its Pe columns are the frozen electron pressure).
        Returns (chi2_raw, chi2_pen) both (a, lam); the penalty adds a
        quadratic bound violation measured in the normalized frame.
        """
        builder = self._atmos_from_params_frozen if frozen \
            else self._atmos_from_params
        a, lam, _nd = P.shape
        rows = a * lam
        chi2_a = torch.empty(rows, dtype=self.dtype, device=self.device)
        base = base[idx_a]
        base2 = base.unsqueeze(1).expand(a, lam, -1).reshape(rows, -1)
        tgt = target[idx_a].unsqueeze(1).expand(a, lam, -1, -1) \
            .reshape(rows, -1)
        sig = sigma[idx_a].unsqueeze(1).expand(a, lam, -1, -1) \
            .reshape(rows, -1)
        chunk = int(self.cma_eval_chunk) if self.cma_eval_chunk else rows
        P2 = P.reshape(rows, _nd)
        for i0 in range(0, rows, chunk):
            sl = slice(i0, min(i0 + chunk, rows))
            trial = builder(ltau, base2[sl], P2[sl], blocks, node_idx, nt,
                            sl.stop - sl.start)
            ymod = self.synthesis._forward(wavs, ltau, trial)
            chi2_a[sl] = chi_square(tgt[sl], ymod.reshape(sl.stop - sl.start,
                                                          -1),
                                    sig[sl])
        chi2_raw = chi2_a.reshape(a, lam)
        # NaN / inf protection: a non-finite evaluation becomes a huge
        # chi2, worse than any bound violation, so it is never selected
        chi2_raw = torch.where(torch.isfinite(chi2_raw), chi2_raw,
                               torch.full_like(chi2_raw, 1e30))
        # quadratic bound penalty in the normalized frame
        if bool(has_bnd.any()):
            xn = P.to(dtype=_STATE_DTYPE) / beta_a
            excess = (xn - hi_a[:, None, :]).clamp(min=0.0) \
                + (lo_a[:, None, :] - xn).clamp(min=0.0)
            excess = excess * has_bnd.to(_STATE_DTYPE)
            pen = (self.cma_penalty * (excess * excess).sum(dim=-1)) \
                .to(dtype=self.dtype)
            pen = pen.reshape(a, lam)
        else:
            pen = torch.zeros(a, lam, dtype=self.dtype, device=self.device)
        return chi2_raw, chi2_raw + pen
