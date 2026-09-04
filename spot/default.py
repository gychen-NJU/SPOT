# -*- coding: utf-8 -*-
"""
spot.default — default configuration dictionary
=================================================

All tunable options of spot live in one dictionary.  Users provide
their own (possibly partial) configuration dict; every key that is
missing is filled from :data:`DEFAULT_CONFIG` with a *recursive* merge
(see :func:`spot.config.merge_config`).  In this way the user only
needs to specify the options that differ from the defaults.

Units used across the whole package
-----------------------------------
* wavelength ``wavs``            : [Angstrom]
* optical depth ``ltau``         : log10(tau5000), dimensionless
* temperature ``T``              : [K]
* electron pressure ``Pe``       : [dyn/cm^2]
* magnetic field ``B``           : [Gauss]
* inclination ``gamma``          : [degrees]
* azimuth ``phi``                : [degrees]
* LOS velocity ``vlos``          : [km/s]
* microturbulence ``vmic``       : [km/s]
* macroturbulence ``vmac``       : [km/s]

Atmosphere layout (batched)
---------------------------
The atmosphere of a single pixel is a flat vector of length
``Nx = 6*Nt + 2``:

    [T(0..Nt-1), Pe(0..Nt-1), B(0..Nt-1), gamma(0..Nt-1),
     phi(0..Nt-1), vlos(0..Nt-1), vmic, vmac]

where ``Nt = len(ltau)`` and ``vmic`` / ``vmac`` are depth-independent.
A batch of ``Nb`` pixels is stored as a tensor of shape ``(Nb, Nx)``.
"""

import os

# ---------------------------------------------------------------------------
# Device / dtype
# ---------------------------------------------------------------------------
def _guess_device():
    """Pick the best available device without importing torch at import time."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEFAULT_CONFIG = {
    # ------------------------------------------------------------------
    # global execution options
    # ------------------------------------------------------------------
    "device": _guess_device(),          # 'cpu', 'cuda' or 'cuda:0'
    "dtype": "float32",                 # 'float32' or 'float64'

    # ------------------------------------------------------------------
    # data files (names of packaged presets, or absolute/relative paths)
    # ------------------------------------------------------------------
    "abundance": "thevenin",            # preset: 'thevenin' (from default/THEVENIN)
    "lines": "default",                 # preset: 'default' (from default/LINES)
    # initial-guess atmosphere of an inversion when no explicit `initial`
    # is passed: 'auto' = run the bundled neural operator
    # (config['network_preset'], e.g. hinode_sp) on the target profiles
    # first and use its atmospheric parameters as the initial guess
    # (recommended: an order of magnitude faster convergence to the same
    # chi2 — see the demo 08 configuration search); any preset name
    # (e.g. 'hot11') or an explicit path is used directly instead.
    "atmosphere": "auto",
    # neural-operator preset used by the 'auto' initial guess
    "network_preset": "hinode_sp",

    # ------------------------------------------------------------------
    # synthesis options
    # ------------------------------------------------------------------
    "synthesis": {
        "solver": "hermitian",          # RTE formal solver (case-insensitive):
                                        # 'hermitian' (Bellot Rubio et al.
                                        # 1998; 4th order, default/reference),
                                        # 'cn' (A-stable Crank-Nicolson in
                                        # linear tau, pyPRT-dsh rk4_solve port
                                        # — pyPRT misnames it "RK4"),
                                        # 'delo' (Rees et al. 1989 DELO,
                                        # pyPRT-dsh delo_solve port)
        "continuum_opacity": "mihalas", # continuum opacity recipe:
                                        # 'mihalas' (Mihalas-type formula
                                        # opacities, default), 'atlas'
                                        # (ATLAS solar ODF Rosseland
                                        # table) or 'opacity_project'
                                        # (Opacity Project tables; both
                                        # ported from pyPRT-dsh)
        "refractive_index": "auto",     # air refractive index at the line:
                                        # 'auto' (default) = ground-based
                                        # Edlén-type air index; a NUMBER is
                                        # used as given (e.g. 1.0 for
                                        # spaceborne observations like
                                        # Hinode, where the vacuum
                                        # wavelength equals the air
                                        # wavelength of the line list)
        "include_stimulated": True,     # stimulated-emission correction (1-e^-x)
        "macroturbulence": True,        # convolve profiles with a Gaussian vmac
        "normalize_continuum": True,    # divide Stokes I by the HSRA continuum
                                        # intensity at the first line (the
                                        # reference continuum normalization)
        "stokes": [True, True, True, True],   # compute I, Q, U, V
        "return_rf": False,             # if True, also return response functions
        "rf_method": "fast",            # 'fast' (default): pyPRT-style
                                        # wavelength-expanded autograd +
                                        # analytic macroturbulence-
                                        # convolution correction —
                                        # mathematically identical to
                                        # 'autograd' but ~30x cheaper
                                        # (and batch-friendly);
                                        # 'autograd' (exact Jacobian via
                                        # vmap+jacrev), 'analytic' (analytic
                                        # variational forward-mode via
                                        # jacfwd),
                                        # 'analytic_chain' (full
                                        # hand-written chain rule, zero
                                        # autograd — used by the benchmark
                                        # inversion/demo)
                                        # or 'finite_diff'
        "rf_eps": 1e-4,                 # relative step for 'finite_diff'
        "rf_scale": {                   # absolute perturbation scale per block
            "T": 1.0, "Pe": 1.0, "B": 1.0, "gamma": 0.01, "phi": 0.01,
            "vlos": 0.01, "vmic": 0.01, "vmac": 0.01,
        },
    },

    # ------------------------------------------------------------------
    # inversion options
    #
    # Defaults = the RECOMMENDED fast pipeline (demo 08 configuration
    # search): the 'auto' network initial guess (see "atmosphere" above)
    # followed by the classical 4-cycle node schedule with fast response
    # functions, S/N-based sigma and NO hydrostatic-Pe boundary
    # condition — reaches chi2 ~6e-2 in ~100 s on the Hinode SP test
    # case (vs ~2500 s for the previous hot11-guess + hse setup).
    # ------------------------------------------------------------------
    "inversion": {
        # number of nodes per physical quantity (0 = fixed, not inverted).
        # each value may be an int (same for every cycle) or a list of
        # ints (one entry per cycle); "auto" = the node count of that
        # cycle is chosen automatically from the chi2 derivative
        # (final cycle of the 4-cycle schedule below).
        "nodes": {
            "T": [2, 3, 4, "auto"],
            "Pe": 0,          # Pe not inverted (kept from the initial guess)
            "B": [1, 2, 3, "auto"],
            "gamma": [1, 2, 2, "auto"],
            "phi": [1, 2, 2, "auto"],
            "vlos": [1, 2, 3, "auto"],
            "vmic": 0,
            "vmac": 0,
        },
        "max_cycles": 4,                # number of cycles (node sets)
        "max_iterations": 80,           # LM iterations per cycle
        "chi2_tolerance": 1e-10,        # relative chi2 change for convergence
        "lambda0": 1e-3,                # initial Marquardt damping
        "lambda_max": 1e10,
        "lambda_factor": 10.0,          # damping increase on rejection
        "lambda_decrease": 0.1,         # damping decrease on acceptance
        "max_step": {                   # max relative step per quantity
            # the reference solver imposes no step caps (only Pe clamps
            # the update to +-0.25); leave the rest unlimited so the
            # LM damping controls the step size.
            "T": None, "Pe": 0.25, "B": None, "gamma": None, "phi": None,
            "vlos": None, "vmic": None, "vmac": None,
        },
        "svd_tolerance": 1e-4,          # singular-value threshold (relative),
                                        # the reference default (tol = 1e-4)
        "sigma": "sir",                 # noise per Stokes sample: 'auto'
                                        # (0.118*sqrt(max|I|) for I,
                                        # 0.204*sqrt(max|I|) for Q/U/V,
                                        # per column), 'sir' (derive the
                                        # noise from S/N and the Stokes
                                        # weights, per column),
                                        # or an array/tensor of shape (4, Nw)
                                        # or (Nb, 4, Nw)
        "snr": 1000.0,                  # S/N used by the sigma='sir' scheme
        "stokes_weights": [1.0, 5.0, 5.0, 10.0],   # I:Q:U:V weights used by
                                                  # the sigma='sir' scheme
        "compute_errors": True,         # compute parameter errors from the
                                        # covariance matrix at the solution
        # hydrostatic gas-pressure boundary condition: gas pressure
        # [dyn/cm^2] at the SURFACE (top) of the grid.  When > 0 and Pe is
        # not inverted, every trial atmosphere has its Pe stratification
        # recomputed from hydrostatic equilibrium (e.g. a surface gas
        # pressure of 500 dyn/cm^2); 0 = keep Pe as the boundary condition
        # (left unchanged).  With the 'auto' network initial guess the
        # network Pe already matches the target closely, so the default 0
        # is both faster (no per-iteration hydrostatic recompute) and
        # accurate — the previous hot11-guess defaults needed hse to fix
        # the Pe mismatch (see the demo 08 search).
        "hse_pg0": 0.0,
        "verbose": True,
        "log_history": True,            # store chi2 per iteration
    },

    # ------------------------------------------------------------------
    # derivative-free (CMA-ES) inversion options — used only by
    # spot.inversion.cmaes.CmaesInversion; the LM solver ignores them
    # ------------------------------------------------------------------
    "cmaes": {
        # population size lambda (None -> 4 + floor(3*ln(nparams)))
        "population": None,
        # initial step size in the normalized frame x = value/scale
        "sigma0": 1.0,
        # optional per-cycle initial step sizes (coarse-to-fine schedule,
        # e.g. [1.5, 0.6, 0.3, 0.1]); None -> sigma0 for every cycle
        "sigma0_cycles": None,
        "min_sigma": 1e-8,              # per-column stopping (step size)
        "max_sigma": 1e4,
        # characteristic scale beta per quantity (x = value / beta)
        "scale": {"T": 1000.0, "Pe": 1.0, "B": 500.0, "gamma": 30.0,
                  "phi": 60.0, "vlos": 1.0, "vmic": 1.0, "vmac": 1.0},
        # physical bounds enforced by a quadratic penalty
        # (set to None or {} to disable)
        "bounds": {"T": [500.0, 30000.0], "Pe": [1e-8, 1e5],
                   "B": [-1000.0, 10000.0], "gamma": [0.0, 180.0],
                   "phi": [0.0, 360.0], "vlos": [-10.0, 10.0],
                   "vmic": [0.0, 10.0], "vmac": [-10.0, 10.0]},
        "penalty": 1e12,                 # quadratic penalty weight
        "covariance_mode": "full",       # 'full' or 'sep' (diagonal)
        "stall": 5,                      # per-column stop: generations
                                         # without a chi2 improvement
        "tol_fun": 1e-8,                 # relative chi2 improvement
        "seed": None,                    # RNG seed (None -> random)
        "verbose": True,
        "eval_chunk": None,              # max rows per batched forward
        # freeze the electron pressure (HSE Pe) during the search:
        # None (default) = exact model (HSE Pe per candidate, LM-identical);
        # int K >= 1     = recompute the HSE-consistent Pe from the current
        #                  best atmosphere every K generations and freeze it
        #                  in between (fast when the hydrostatic Pe recompute
        #                  dominates; the delivered result is always exact).
        "hse_refresh": None,
    },

    # ------------------------------------------------------------------
    # preset model handling
    # ------------------------------------------------------------------
    "models": {
        # interpolation order used when the user's ltau grid differs from
        # the preset grid: 'linear' or 'cubic' (extrapolation supported)
        "interpolation": "linear",
    },
}
