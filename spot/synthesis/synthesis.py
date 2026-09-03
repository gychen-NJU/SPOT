# -*- coding: utf-8 -*-
"""
spot.synthesis.synthesis — the Synthesis class
================================================
Batch (Nb), vectorized synthesis of Stokes IQUV spectral-line profiles on
CPU or CUDA.  The forward model carries an atmosphere through the full
line-synthesis chain:

    atmosphere -> ionization equilibrium -> continuum & line opacity
               -> Zeeman (Voigt/Faraday) profiles -> propagation matrix K
               -> Hermitian RTE -> Stokes (normalized by HSRA continuum)
               -> macroturbulence convolution

The forward model is written as a *pure* differentiable function of the
atmosphere vector, so response functions dI/dx are obtained either with
autograd (exact, default) or with central finite differences.

Quick start
-----------
>>> from spot import Synthesis
>>> synth = Synthesis({"device": "cpu"})
>>> stokes = synth(wavs, ltau, atmos)      # stokes.shape == (Nb, Nw, 4)

Atmosphere layout
-----------------
``atmos`` has shape ``(Nb, Nx)`` with ``Nx = 6*Nt + 2``:

    [T (Nt), Pe (Nt), B (Nt), gamma (Nt), phi (Nt), vlos (Nt),
     vmic (1), vmac (1)]

Units: T [K], Pe [dyn/cm^2], B [G], gamma/phi [deg], vlos/vmic/vmac
[km/s].  ``ltau`` is log10(tau5000), strictly *decreasing* (index 0 =
deepest layer).
"""

import warnings

import numpy as np
import torch

from ..config import merge_config
from ..default import DEFAULT_CONFIG
from ..physics import opacity as _opacity
from ..physics.absorption import absorption_matrix
from ..physics.atoms import (ATOM_WEIGHTS, IONIZATION_ENERGY_1,
                             IONIZATION_ENERGY_2, AbundanceTable,
                             element_symbol_to_number)
from ..physics.line import (component_profiles, component_profiles_batched,
                            damping_parameter, line_opacity, zeeman_pattern)
from ..physics.pressure import ionization_equilibrium
from ..physics.rte import (hermite_solve, hermite_solve_general,
                           propagation_operator, delo_solve, cn_solve,
                           resolve_rte_solver)
from ..physics.thermodynamics import (air_refractive_index, hsra_continuum,
                                      planck_intensity)
from ..utils.data_io import load_atmosphere, load_lines

_LOG10 = 2.3025851

# One-shot warning flag for non-finite response functions (float32
# extreme-state robustness, see _sanitize_rf).
_WARN_RF_NONFINITE = [True]

__all__ = ["Synthesis"]


def _sanitize_rf(rf):
    """
    Replace non-finite entries of a response-function tensor with 0.

    Float32 back-propagation can overflow at non-physical extreme
    atmosphere states (e.g. a cold surface layer with tiny Pe makes
    ``1/kappa5`` ~ 1e28, whose chain-rule factors exceed the float32
    range): the forward stays finite but the gradient comes out as
    NaN/Inf.  This keeps the run usable — the affected directions lose
    their gradient (set to 0) — and emits ONE RuntimeWarning per
    process, recommending float64 for reliable gradients.

    Float64 forward models are never affected in practice (float64
    intermediate range ~1e308), so this is a no-op check there.
    """
    if torch.isfinite(rf).all():
        return rf
    if _WARN_RF_NONFINITE[0]:
        _WARN_RF_NONFINITE[0] = False
        warnings.warn(
            "rf: non-finite response-function values detected (float32 "
            "overflow/underflow at an extreme atmosphere state); the "
            "affected entries were replaced by 0. Those directions are "
            "NOT constraint-free — for reliable gradients use float64 "
            "(the reference float64 precision) or keep the parameters "
            "bounded.",
            RuntimeWarning, stacklevel=3)
    return torch.nan_to_num(rf, nan=0.0, posinf=0.0, neginf=0.0)


class Synthesis:
    """
    Batched spectral-line synthesis (Stokes IQUV).

    Parameters
    ----------
    config : dict, optional
        User configuration merged over :data:`DEFAULT_CONFIG` (only the
        differing keys need to be given).  Relevant keys: ``device``,
        ``dtype``, ``lines``, ``abundance`` and the ``synthesis``
        subsection.
    """

    def __init__(self, config=None):
        self.config = merge_config(config or {}, DEFAULT_CONFIG)
        synth = self.config["synthesis"]
        self.device = self._resolve_device(self.config["device"])
        self.dtype = getattr(torch, self.config["dtype"])

        # ---- data files ---------------------------------------------------
        lines_cfg = self.config["lines"]
        if isinstance(lines_cfg, (list, tuple)):
            # list of indices into the default line list
            all_lines = load_lines("default")
            self.lines = [all_lines[i] for i in lines_cfg]
        else:
            self.lines = load_lines(lines_cfg)
        self.abundance = AbundanceTable.from_default()
        if isinstance(self.config["abundance"], str):
            from ..utils.data_io import load_abundance
            table = load_abundance(self.config["abundance"])
            self.abundance = AbundanceTable(table["abundance_log12"])

        self._rte = resolve_rte_solver(synth.get("solver", "hermitian"))
        # continuum opacity recipe: 'mihalas' (the default, a resolved
        # per-wavelength continuum opacity model), 'atlas' (ATLAS solar
        # ODF Rosseland table) or 'opacity_project' (Opacity Project
        # tables).  The table recipes are wavelength-averaged Rosseland
        # opacities ported from pyPRT-dsh synthesis.py.
        self._cont_opacity = str(synth.get("continuum_opacity",
                                           "mihalas")).strip().lower()
        if self._cont_opacity not in ("mihalas", "atlas", "opacity_project",
                                      "opacity project"):
            raise ValueError(
                f"unknown continuum_opacity {self._cont_opacity!r}; "
                f"available: ['mihalas', 'atlas', 'opacity_project']")
        self._normalize = synth.get("normalize_continuum", True)
        self._use_mac = synth.get("macroturbulence", True)
        # refractive index at the line: 'auto' (or None) = ground-based
        # air index (standard air-refraction formula); any explicit number
        # (e.g. 1.0 for spaceborne observations such as Hinode) is used
        # AS GIVEN.
        self._refidx = synth.get("refractive_index", "auto")
        self._include_stim = synth.get("include_stimulated", True)

        # ---- precompute per-line static data ------------------------------
        self._line_static = []
        for rec in self.lines:
            static = self._prepare_line(rec)
            self._line_static.append(static)

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(device):
        if device in ("auto", None):
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _prepare_line(self, rec):
        """Static (depth-independent) data of one spectral line."""
        # the generic symbol XX (and the already-iron FE) maps to Z=26
        atom = rec["atom"].upper()
        atom_z = 26 if atom in ("XX", "FE") else element_symbol_to_number(atom)
        wl_air = rec["wavelength"]                    # Angstrom (air)
        if self._refidx in ("auto", None):
            # ground-based reference: standard air refractive index;
            # keep the historical 1.0004 fallback for UV lines (< 1800 A)
            nair = air_refractive_index(wl_air * 1e-4) \
                if wl_air >= 1800.0 else 1.0004
        else:
            nair = float(self._refidx)                # explicit (1.0 = space)
        wvac = nair * wl_air * 1e-8                   # vacuum wavelength [cm]
        wc = wvac / 2.99792458e10                     # 1/frequency [s]
        dl0 = 4.6686e-5 * wvac * wvac                 # Zeeman unit [cm/G]
        shifts, strengths, npi, nr = zeeman_pattern(
            rec["mult_l"], rec["design_l"], rec["tam_l"],
            rec["mult_u"], rec["design_u"], rec["tam_u"], dl0)

        return {
            "rec": rec,
            "wvac": wvac,
            "wl_air": wl_air,                 # Angstrom (air); used for the
                                            # Planck source function and the
                                            # HSRA reference continuum
                                            # normalization wavelength
            "wc": wc,
            "weight": float(ATOM_WEIGHTS[atom_z - 1]),
            "gf_abu": (10.0 ** rec["loggf"]) * float(self.abundance[atom_z]),
            "chi1": float(IONIZATION_ENERGY_1[atom_z - 1]),
            "chi2": float(IONIZATION_ENERGY_2[atom_z - 1]),
            "shifts": [torch.as_tensor(s, dtype=self.dtype)
                       for s in (shifts[:npi], shifts[npi:npi + nr],
                                 shifts[npi + nr:])],
            "strengths": [torch.as_tensor(s, dtype=self.dtype)
                          for s in (strengths[:npi], strengths[npi:npi + nr],
                                    strengths[npi + nr:])],
            # concatenated (pi, sigma_r, sigma_l) arrays + group sizes for
            # the batched Zeeman profile evaluation (one Voigt call per
            # component block instead of one per component)
            "shifts_all": torch.as_tensor(shifts, dtype=self.dtype),
            "strengths_all": torch.as_tensor(strengths, dtype=self.dtype),
            "npi": int(npi), "nr": int(nr),
        }

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def __call__(self, wavs, ltau, atmos, return_rf=None):
        """
        Synthesize Stokes profiles.

        Parameters
        ----------
        wavs : torch.Tensor (Nw,)
            Wavelength grid [Angstrom].
        ltau : torch.Tensor (Nt,)
            log10(tau5000), strictly decreasing (deepest first).
        atmos : torch.Tensor (Nb, Nx)
            Atmosphere vectors (see module docstring); may also be a
            preset-model name (str), in which case it is loaded and
            interpolated onto ``ltau``.
        return_rf : bool, optional
            If True, also return the response functions.  Defaults to
            ``config['synthesis']['return_rf']``.

        Returns
        -------
        stokes : torch.Tensor (Nb, Nw, 4)
            Emergent Stokes (I, Q, U, V) at the given wavelengths.
        rf : torch.Tensor (Nb, Nw, 4, Nx), optional
            Response functions dI/dx (only if ``return_rf``).
        """
        if return_rf is None:
            return_rf = self.config["synthesis"]["return_rf"]
        wavs = torch.as_tensor(wavs, dtype=self.dtype, device=self.device)
        ltau = torch.as_tensor(ltau, dtype=self.dtype, device=self.device)
        atmos = self._normalize_atmos(atmos, ltau)

        if not torch.all(ltau[:-1] > ltau[1:]):
            raise ValueError(
                "ltau must be strictly DECREASING (deepest layer first), "
                "as in standard model files")

        if return_rf:
            if atmos.shape[1] == 7 * ltau.shape[0] + 1:
                raise NotImplementedError(
                    "Response functions currently support the scalar-vmic "
                    "layout (Nx = 6*Nt+2) only; use per-depth vmic with "
                    "return_rf=False (the RTE/Saha part differs only in "
                    "the vmic dependence of vdop, which is analytic and "
                    "could be added on request).")
            rf = self.response_functions(wavs, ltau, atmos)
            stokes = self._forward(wavs, ltau, atmos)
            return stokes, rf
        return self._forward(wavs, ltau, atmos)

    def _normalize_atmos(self, atmos, ltau):
        """Accept (Nb,Nx) tensors, preset names, or dicts from load_atmosphere.

        Microturbulence is a SINGLE parameter (one node; the ``.mod``
        file stores per-layer rows but the value is the same in every
        depth).  Two atmosphere layouts are therefore supported:

        * ``Nx = 6*Nt + 2`` — depth-INDEPENDENT microturbulence
          ``[T, Pe, B, gamma, phi, vlos, vmic, vmac]`` (the historical
          layout; response functions and the inversion also use this
          one; constant per-layer vmic profiles collapse onto it);
        * ``Nx = 7*Nt + 1`` — depth-DEPENDENT microturbulence
          ``[T, Pe, B, gamma, phi, vlos, vmic(0..Nt-1), vmac]``, for
          model files (e.g. old pyprt Bifrost data) whose microturbulence
          genuinely varies with depth; the inversion still treats vmic
          as one parameter (the depth profile is averaged into a scalar
          seed) and response functions require the scalar layout.
        """
        if isinstance(atmos, str) or (
                isinstance(atmos, np.ndarray) and atmos.ndim == 1
                and atmos.dtype.kind in "SU"):
            model = load_atmosphere(atmos, ltau=ltau.cpu().numpy(),
                                    interpolation=self.config["models"]["interpolation"])
            return self._model_to_atmos(model)
        if isinstance(atmos, dict):
            model = dict(atmos)
            import numpy as _np
            src = _np.asarray(model["ltau"], dtype=float)
            tgt = ltau.cpu().numpy()
            if not _np.allclose(src, tgt):
                from ..utils.interpolation import interp_to_grid
                for key in ("T", "Pe", "vmic", "B", "vlos", "gamma", "phi"):
                    model[key] = interp_to_grid(src, model[key], tgt)
                model["ltau"] = tgt
            return self._model_to_atmos(model)
        atmos = torch.as_tensor(atmos, dtype=self.dtype, device=self.device)
        if atmos.ndim == 1:
            atmos = atmos.unsqueeze(0)
        ncols = atmos.shape[1]
        if ncols not in (6 * ltau.shape[0] + 2, 7 * ltau.shape[0] + 1):
            raise ValueError(
                f"atmos must have Nx = 6*Nt + 2 = {6 * ltau.shape[0] + 2} "
                f"(scalar vmic) or Nx = 7*Nt + 1 = {7 * ltau.shape[0] + 1} "
                f"(per-depth vmic, the per-layer .mod-convention layout), "
                f"got {ncols}")
        return atmos

    def _model_to_atmos(self, model):
        """Pack a load_atmosphere dict (velocities in km/s) into (Nb, Nx).

        By convention the microturbulence is ONE value even though the
        ``.mod`` file stores it per layer (the same number appears in
        every depth row; a single node is used for its inversion): a
        ``vmic`` array of length ``nt`` whose values are (nearly)
        constant is therefore collapsed to the scalar ``6*Nt+2`` layout.
        Only a genuinely depth-VARYING ``vmic`` profile (e.g. the old
        pyprt Bifrost data, where every layer differs) keeps the
        ``7*Nt+1`` per-depth layout.
        """
        import numpy as _np
        import torch as _torch
        nt = len(model["ltau"])
        # work in numpy (handles both numpy and torch model values)
        def asnp(v):
            return v.detach().cpu().numpy() if _torch.is_tensor(v) else _np.asarray(v)
        ltau_arr = asnp(model["ltau"])
        order = _np.argsort(ltau_arr)[::-1].copy()  # deepest first
        vmic_full = _np.asarray(asnp(model["vmic"]), dtype=float)
        if vmic_full.size > 2:
            spread = float(vmic_full.max() - vmic_full.min())
            scale = max(float(np.max(np.abs(vmic_full))), 1.0)
            depth_vmic = spread > 1e-4 * scale      # nearly-constant -> scalar
        else:
            depth_vmic = False
        vmic_arr = vmic_full[order] if depth_vmic else vmic_full
        parts = [asnp(model["T"])[order], asnp(model["Pe"])[order],
                 asnp(model["B"])[order],
                 asnp(model["gamma"])[order], asnp(model["phi"])[order],
                 asnp(model["vlos"])[order]]
        if depth_vmic:
            parts.append(np.asarray(vmic_arr, dtype=float).reshape(-1))
        else:
            parts.append(_np.full(1, float(_np.mean(vmic_full))))
        parts.append(_np.full(1, 0.0))
        x = torch.as_tensor(_np.concatenate(parts), dtype=self.dtype,
                            device=self.device).unsqueeze(0)
        return x

    # ------------------------------------------------------------------
    # forward model (differentiable)
    # ------------------------------------------------------------------
    def _forward(self, wavs, ltau, atmos):
        """Pure differentiable forward: (Nb, Nx) -> (Nb, Nw, 4)."""
        nt = ltau.shape[0]
        nw = wavs.shape[0]
        nb = atmos.shape[0]

        # unpack the atmosphere (all in [km/s] -> cm/s internally)
        t, pe, b, gamma, phi, vlos, vmic, vmac = self._unpack(atmos, nt)

        theta = 5040.0 / t
        pp = ionization_equilibrium(theta, pe, abundance=self.abundance)

        k_matrix, svec, wl1_cm, wl1_air = self._K_source(
            wavs, t, pe, b, gamma, phi, vlos, vmic, theta, pp, nt, nw)

        stokes = self._rte(ltau, k_matrix, svec)

        # normalize the emergent Stokes vector by the HSRA continuum at
        # the AIR wavelength of the region's first line, so the output is
        # relative intensity (I/Ic) at each spectral point.
        if self._normalize:
            cont_norm = hsra_continuum(float(wl1_air))
            stokes = stokes / cont_norm

        # macroturbulence convolution (per line region; Gaussian width
        # sigma = lambda0 * vmac / c  with lambda0 the AIR wavelength of
        # the region's first line).  Always applied when enabled: with
        # vmac ~ 0 the Gaussian kernel collapses to a delta (sigma
        # clamped), keeping the forward free of data-dependent control
        # flow (vmap-compatible).
        if self._use_mac:
            stokes = self._macroturbulence(stokes, vmac, wavs, wl1_air)
        return stokes

    @staticmethod
    def _unpack(atmos, nt):
        """Split the (Nb, 6Nt+2) atmosphere into its physical quantities."""
        t = atmos[:, 0:nt].unsqueeze(-1)                        # (Nb,Nt,1) K
        pe = atmos[:, nt:2 * nt].unsqueeze(-1).clamp(min=1e-10) # dyn/cm^2
        b = atmos[:, 2 * nt:3 * nt].unsqueeze(-1)               # G
        gamma = (atmos[:, 3 * nt:4 * nt].unsqueeze(-1)
                 * (np.pi / 180.0))                             # rad
        phi = (atmos[:, 4 * nt:5 * nt].unsqueeze(-1)
               * (np.pi / 180.0))                               # rad
        vlos = atmos[:, 5 * nt:6 * nt].unsqueeze(-1) * 1e5      # cm/s
        if atmos.shape[1] == 7 * nt + 1:
            # per-layer .mod convention: microturbulence given per depth (km/s)
            vmic = atmos[:, 6 * nt:7 * nt].unsqueeze(-1) * 1e5  # (Nb,Nt,1) cm/s
            vmac = atmos[:, 7 * nt] * 1e5                       # (Nb,) cm/s
        else:
            vmic = atmos[:, 6 * nt:6 * nt + 1].unsqueeze(-1) * 1e5  # cm/s
            vmac = atmos[:, 6 * nt + 1] * 1e5                   # (Nb,) cm/s
        return t, pe, b, gamma, phi, vlos, vmic, vmac

    def _nair_for(self, wavs):
        """Air refractive index at each wavelength, for the
        continuum-opacity evaluation.

        The input wavelength grid is given in air (as measured by a
        ground-based instrument), while the radiative-transfer geometry
        and the opacity are expressed on the vacuum wavelength scale.
        nair = c_air/c_vac converts between them: the default ('auto')
        uses the standard air-refraction formula (nair ~ 1.0003, with a
        fixed 1.0004 fallback below 1800 A), and an explicit
        refractive_index (e.g. 1.0 for spaceborne observations) is used
        as given, for parity with the line-wavelength conversion.

        Vectorized (torch-only arithmetic, no .numpy()/Python loop) so it
        can be called under the torch.func transforms used by the
        analytic response functions.
        """
        wl_a = torch.as_tensor(wavs, dtype=self.dtype, device=self.device)
        if self._refidx in ("auto", None):
            from ..physics.thermodynamics import air_refractive_index_torch
            n = air_refractive_index_torch(wl_a * 1e-4)
            n = torch.where(wl_a >= 1800.0, n, torch.full_like(wl_a, 1.0004))
            return n
        return torch.full_like(wl_a, float(self._refidx))

    def _K_source(self, wavs, t, pe, b, gamma, phi, vlos, vmic, theta, pp,
                  nt, nw):
        """Propagation matrix K and source vector S of the atmosphere.

        Returns (k_matrix (Nb,Nt,Nw,4,4), svec (Nb,Nt,Nw,4), wl1_cm).
        Pure function of the physical quantities (no Hermite solve), so
        it can be differentiated to obtain the analytic response
        functions.
        """
        # continuum opacity: kappa(lambda) and kappa(5000 A) per H nucleon,
        # each evaluated with the wavelength-appropriate air refractive
        # index so the opacity is computed on the vacuum wavelength scale.
        nair_w = self._nair_for(wavs)
        nair_5 = self._nair_for(torch.tensor([5000.0], dtype=self.dtype,
                                             device=self.device))
        if self._cont_opacity in ("mihalas",):
            kappa5 = _opacity.continuum_opacity(
                torch.tensor([5000.0], dtype=self.dtype, device=self.device),
                t, pe, pp, refractive_index=nair_5)             # (Nb,Nt,1)
            kappa_w = _opacity.continuum_opacity(
                wavs, t, pe, pp, refractive_index=nair_w)       # (Nb,Nt,Nw)
        else:
            # ATLAS / Opacity Project: wavelength-averaged Rosseland
            # coefficient per H nucleon.  The kappa(5000) normalisation
            # is the same per-H value, so t0 = kappa_w / kappa5 = 1
            # (tables carry no wavelength dependence): the continuum
            # enters the RTE as the tau5000 scale exactly as in pyPRT.
            vmic_km = vmic / 1e5                       # cm/s -> km/s (ATLAS grid)
            kappa5 = _opacity.continuum_opacity_recipe(
                self._cont_opacity,
                torch.tensor([5000.0], dtype=self.dtype,
                             device=self.device),
                t, pe, pp, vmic=vmic_km)               # (Nb,Nt,1)
            kappa_w = kappa5.expand(*t.shape[:2], nw)  # (Nb,Nt,Nw)
        t0 = kappa_w / kappa5                                   # (Nb,Nt,Nw)

        # source function at the first line's wavelength (the vacuum
        # wavelength of the region's reference line)
        wl1_cm = self._line_static[0]["wvac"]
        wl1_air = self._line_static[0]["wl_air"]
        # if the grid does not overlap the first line, use the first line
        # whose wavelength falls inside the grid (each spectral region is
        # synthesized around its own reference line); tensor-only
        # comparisons keep the forward vmap-compatible
        wl1_sel = None
        wlo, whi = wavs.min(), wavs.max()
        for static in self._line_static:
            wl = static["wvac"] * 1e8
            if bool((wlo <= wl) & (wl <= whi)):
                wl1_sel = static["wvac"]
                break
        if wl1_sel is not None:
            wl1_cm = wl1_sel
            wl1_air = next(s["wl_air"] for s in self._line_static
                           if s["wvac"] == wl1_sel)
        # Planck source at the first line's wavelength (converted from
        # the AIR Angstrom value to cm); dominated by thermal emission,
        # so only the I component of svec is non-zero.
        source = planck_intensity(t, wl1_air * 1e-8)        # (Nb,Nt,1)
        source = source.expand(-1, -1, nw)                      # (Nb,Nt,Nw)
        svec = torch.stack([source, torch.zeros_like(source),
                            torch.zeros_like(source),
                            torch.zeros_like(source)], dim=-1)  # (Nb,Nt,Nw,4)

        # accumulate the line contribution to K over all lines
        k_line = torch.zeros((t.shape[0], nt, nw, 4, 4), device=self.device,
                             dtype=self.dtype)
        for static in self._line_static:
            k_line = k_line + self._line_matrix(
                static, wavs, t, pe, b, gamma, phi, vlos, vmic, theta, pp,
                kappa5, nt, nw)

        # full propagation matrix: continuum + lines (per unit tau5000)
        k_matrix = _continuum_matrix(t0) + k_line
        return k_matrix, svec, wl1_cm, wl1_air

    def _line_matrix(self, static, wavs, t, pe, b, gamma, phi, vlos, vmic,
                     theta, pp, kappa5, nt, nw):
        """Propagation-matrix contribution of a single spectral line."""
        rec = static["rec"]
        weight = static["weight"]
        wvac = static["wvac"]
        wc = static["wc"]

        # Doppler velocity (cm/s) and Doppler width (cm)
        croot = 1.66286e8 / weight
        vdop = torch.sqrt(croot * t + vmic ** 2)               # (Nb,Nt,1)
        dldop = wc * vdop                                      # cm

        # partition functions of the element
        u0, u1, u2 = _partition(rec["atom"], t)

        # line opacity per unit tau5000
        eta0 = line_opacity(
            theta, t, pe, rec["energy_low"], rec["loggf"], static["gf_abu"],
            wvac, weight, u0, u1, u2, static["chi1"], static["chi2"],
            vdop, kappa5, stimulated=self._include_stim,
            istage=rec["stage"], ne=pp["ne"])

        # damping parameter
        a = damping_parameter(
            t, theta, pe, pp, wvac, vdop, vmic, weight,
            rec["alfa"], rec["sigma"], rec["zeff"], rec["stage"],
            rec["energy_low"], static["chi1"], static["chi2"])

        # cap the damping at 3: the maximum is taken over the LAYERS of
        # each individual model (i.e. over the depth axis only), so the
        # cap is per-column and batch-size independent.  This keeps the
        # Voigt damping within the regime where the line profile is well
        # behaved (an unphysically large a would flatten the line).
        amax = a.amax(dim=1, keepdim=True)      # (Nb,1,1), per model
        scale = torch.where(amax > 3.0, 3.0 / amax,
                            torch.ones_like(amax))
        a = a * scale

        # wavelength shift in Doppler units
        dlamda = (wavs[None, None, :] - rec["wavelength"]) * 1e-8  # cm
        v = dlamda / dldop - vlos / vdop                            # (Nb,Nt,Nw)
        hh = b / dldop                                              # B/dldop

        # Zeeman component profiles: (pi, sigma_r, sigma_l) — evaluated in
        # ONE batched Voigt/Faraday pass over an extra component axis
        # (component blocks, memory-bounded) instead of one call per
        # Zeeman component; identical mathematics to the per-component
        # loop, with the dispersive (Faraday) profiles coupled to the
        # absorption ones through the Voigt function.
        (eta_pi, rho_pi, eta_r, rho_r,
         eta_l, rho_l) = component_profiles_batched(
            v, a, static["shifts_all"], static["strengths_all"],
            static["npi"], static["nr"], hh)

        t3 = eta0.expand(-1, -1, nw)                                # kappaL/kappa5
        return absorption_matrix(torch.zeros_like(t3), t3,
                                 eta_pi, eta_r, eta_l,
                                 rho_pi, rho_r, rho_l, gamma, phi)

    # ------------------------------------------------------------------
    # full-analytic chain-rule response functions
    # ------------------------------------------------------------------
    def _K_source_derivatives(self, wavs, ltau, atmos, pp, dpp, ddpp):
        """
        K and its 7 analytic derivatives dK/dx (hand-written chain rule).

        Returns ``(k_matrix, svec, dk_dict)`` where ``dk_dict[name]`` is
        (Nb, Nt, Nw, 4, 4) for name in
        T, Pe, B, gamma, phi, vlos, vmic.  Units: T [K], Pe [dyn/cm^2],
        B [G], gamma/phi [rad], vlos/vmic [km/s] (vlos/vmic get the
        final 1e5 rescale, exactly like the jacfwd path).
        """
        from ..physics.derivatives import (damping_derivatives,
                                           line_opacity_derivatives,
                                           mvoigt_derivatives,
                                           partition_derivatives)
        nt = ltau.shape[0]
        nw = wavs.shape[0]
        nb = atmos.shape[0]
        t, pe, b, gamma, phi, vlos, vmic, vmac = self._unpack(atmos, nt)
        theta = 5040.0 / t

        # kappa(lambda)/kappa(5000) and its dT/dPe derivatives.
        # NOTE: the continuum-opacity recipes follow _K_source and hard-code
        # refractive_index=1.0 (the reference behaviour); the configured
        # line index only affects the line-centre / Doppler / Zeeman
        # vacuum wavelengths.
        if self._cont_opacity != "mihalas":
            raise ValueError(
                "analytic response functions (rf_method='analytic' / "
                "'analytic_chain') are only defined for the mihalas "
                "continuum opacity; use rf_method='fast' or 'autograd' "
                f"with continuum_opacity={self._cont_opacity!r}")
        kappa5, dkappa5, ddkappa5 = _opacity.continuum_opacity_derivatives(
            torch.tensor([5000.0], dtype=self.dtype, device=self.device),
            t, pe, pp, dpp, ddpp, refractive_index=1.0)
        kappa_w, dkappa_w, ddkappa_w = _opacity.continuum_opacity_derivatives(
            wavs, t, pe, pp, dpp, ddpp, refractive_index=1.0)
        t0 = kappa_w / kappa5                                     # (Nb,Nt,Nw)
        dlnk5 = dkappa5 / kappa5                                  # d ln k5/dT
        ddlnk5 = ddkappa5 / kappa5                                # d ln k5/dPe
        # d(kappa_w/kappa5)/dx = (dkappa_w - t0*dkappa5)/kappa5
        t1 = (dkappa_w - t0 * dkappa5) / kappa5                   # d t0/dT
        t2 = (ddkappa_w - t0 * ddkappa5) / kappa5                 # d t0/dPe

        # source vector (Planck, I component only)
        wl1_cm = self._line_static[0]["wvac"]
        wl1_sel = None
        wlo, whi = wavs.min(), wavs.max()
        for static in self._line_static:
            wl = static["wvac"] * 1e8
            if bool((wlo <= wl) & (wl <= whi)):
                wl1_sel = static["wvac"]
                break
        if wl1_sel is not None:
            wl1_cm = wl1_sel
        source = planck_intensity(t, wl1_cm)
        source = source.expand(-1, -1, nw)
        svec = torch.stack([source, torch.zeros_like(source),
                            torch.zeros_like(source),
                            torch.zeros_like(source)], dim=-1)

        # accumulate line contributions
        k_line = torch.zeros((nb, nt, nw, 4, 4), device=self.device,
                             dtype=self.dtype)
        dk = {name: torch.zeros((nb, nt, nw, 4, 4), device=self.device,
                                dtype=self.dtype)
              for name in ("T", "Pe", "B", "gamma", "phi", "vlos", "vmic")}
        for iline, static in enumerate(self._line_static):
            # the continuum opacity ratio and its derivatives (t0/t1/t2)
            # enter only through the FIRST line of the blend; the
            # remaining lines contribute line opacity only.  Passing
            # t0=0 and t1=t2=0 for the other lines reproduces exactly
            # that (the continuum matrix is added below through dk0/k0,
            # once).
            t0i = t0 if iline == 0 else torch.zeros_like(t0)
            t1i = t1 if iline == 0 else torch.zeros_like(t1)
            t2i = t2 if iline == 0 else torch.zeros_like(t2)
            k1, dk1 = self._line_matrix_derivatives(
                static, wavs, t, pe, b, gamma, phi, vlos, vmic, theta, pp,
                dpp, ddpp, kappa5, t0i, t1i, t2i, dlnk5, ddlnk5, nt, nw)
            k_line = k_line + k1
            for name in dk:
                dk[name] = dk[name] + dk1[name]

        k_matrix = _continuum_matrix(t0) + k_line
        return k_matrix, svec, dk

    def _line_matrix_derivatives(self, static, wavs, t, pe, b, gamma, phi,
                                 vlos, vmic, theta, pp, dpp, ddpp, kappa5,
                                 t0, t1, t2, dlnk5, ddlnk5, nt, nw):
        """Line contribution to K and its 7 derivatives."""
        from ..physics.absorption import profile_components
        from ..physics.derivatives import (damping_derivatives,
                                           line_opacity_derivatives,
                                           mvoigt_derivatives,
                                           partition_derivatives)
        rec = static["rec"]
        weight = static["weight"]
        wvac = static["wvac"]
        wc = static["wc"]
        nb = t.shape[0]

        croot = 1.66286e8 / weight
        vdop = torch.sqrt(croot * t + vmic ** 2)               # (Nb,Nt,1)
        dvdop = croot / vdop / 2.0                             # d vdop/dT
        mvdop = vmic / vdop                                    # d vdop/dvmic
        dldop = wc * vdop
        t13 = dvdop / vdop                                     # d ln vdop/dT
        t14 = mvdop / vdop                                     # d ln vdop/dvmic
        w1 = wc / dldop                                        # 1/vdop

        # partition functions + d ln u/dT
        u0, u1, u2 = _partition(rec["atom"], t)
        du0, du1, du2 = partition_derivatives(rec["atom"], t)

        # eta0 (line opacity per unit tau5000) and its derivatives
        eta0, deta0, ddeta0, meta0 = line_opacity_derivatives(
            theta, t, pe, rec["energy_low"], rec["loggf"], static["gf_abu"],
            wvac, u0, u1, u2, du0, du1, du2, static["chi1"], static["chi2"],
            vdop, dvdop, mvdop, kappa5, dlnk5, ddlnk5,
            stimulated=self._include_stim, istage=rec["stage"], ne=pp["ne"])

        # damping parameter and its derivatives
        a, da, dda, ma = damping_derivatives(
            t, theta, pe, pp, dpp, ddpp, wvac, vdop, dvdop, mvdop, weight,
            rec["alfa"], rec["sigma"], rec["zeff"], rec["stage"],
            rec["energy_low"], static["chi1"], static["chi2"])
        # cap the damping at 3: the maximum is per MODEL (over the depth
        # axis), so the cap is per-column and batch-size independent;
        # scale a and its derivatives accordingly.
        amax = a.amax(dim=1, keepdim=True)      # (Nb,1,1), per model
        scale = torch.where(amax > 3.0, 3.0 / amax, torch.ones_like(amax))
        a = a * scale
        da = da * scale
        dda = dda * scale
        ma = ma * scale

        # wavelength shift and its dependence on vdop
        dlamda = (wavs[None, None, :] - rec["wavelength"]) * 1e-8
        v = dlamda / dldop - vlos / vdop                       # (Nb,Nt,Nw)
        hh = b / dldop

        sg = torch.sin(gamma)
        cg = torch.cos(gamma)
        s2g = sg * sg
        c2g = 1.0 + cg * cg
        scg = sg * cg
        sf = torch.sin(2.0 * phi)
        cf = torch.cos(2.0 * phi)

        t3 = eta0.expand(-1, -1, nw)

        # --- Voigt/Faraday derivatives for the three Zeeman groups (pi, sigma_r, sigma_l) ---
        mv = []
        for group in range(3):
            mv.append(mvoigt_derivatives(
                v, a, static["shifts"][group], static["strengths"][group],
                hh, w1, t13, t14, dldop))
        (etar, vetar, getar, ettar, ettvr, ettmr,
         esar, vesar, gesar, essar, essvr, essmr) = mv[1]
        (etal, vetal, getal, ettal, ettvl, ettml,
         esal, vesal, gesal, essal, essvl, essml) = mv[2]
        (etap, vetap, getap, ettap, ettvp, ettmp,
         esap, vesap, gesap, essap, essvp, essmp) = mv[0]

        # ---------- dK/dT ------------------------------------------------
        detar = deta0.expand(-1, -1, nw) * etar + t3 * (ettar * da.expand(-1, -1, nw) + ettvr)
        detal = deta0.expand(-1, -1, nw) * etal + t3 * (ettal * da.expand(-1, -1, nw) + ettvl)
        detap = deta0.expand(-1, -1, nw) * etap + t3 * (ettap * da.expand(-1, -1, nw) + ettvp)
        desar = deta0.expand(-1, -1, nw) * esar + t3 * (essar * da.expand(-1, -1, nw) + essvr)
        desal = deta0.expand(-1, -1, nw) * esal + t3 * (essal * da.expand(-1, -1, nw) + essvl)
        desap = deta0.expand(-1, -1, nw) * esap + t3 * (essap * da.expand(-1, -1, nw) + essvp)
        tm = 0.5 * (detar + detal)
        tn = 0.5 * (detar - detal)
        sm = 0.5 * (desar + desal)
        sn = 0.5 * (desar - desal)
        tpm = 0.5 * (detap - tm)
        spm = 0.5 * (desap - sm)
        fi = t1.expand(-1, -1, nw) + 0.5 * (detap * s2g + tm * c2g)
        fq = tpm * s2g * cf
        fu = tpm * s2g * sf
        fv = tn * cg
        fq1 = spm * s2g * cf
        fu1 = spm * s2g * sf
        fv1 = sn * cg
        dkT = _matabs(fi, fq, fu, fv, fq1, fu1, fv1)

        # ---------- dK/dPe -----------------------------------------------
        detar = ddeta0.expand(-1, -1, nw) * etar + t3 * (ettar * dda.expand(-1, -1, nw))
        detal = ddeta0.expand(-1, -1, nw) * etal + t3 * (ettal * dda.expand(-1, -1, nw))
        detap = ddeta0.expand(-1, -1, nw) * etap + t3 * (ettap * dda.expand(-1, -1, nw))
        desar = ddeta0.expand(-1, -1, nw) * esar + t3 * (essar * dda.expand(-1, -1, nw))
        desal = ddeta0.expand(-1, -1, nw) * esal + t3 * (essal * dda.expand(-1, -1, nw))
        desap = ddeta0.expand(-1, -1, nw) * esap + t3 * (essap * dda.expand(-1, -1, nw))
        tm = 0.5 * (detar + detal)
        tn = 0.5 * (detar - detal)
        sm = 0.5 * (desar + desal)
        sn = 0.5 * (desar - desal)
        tpm = 0.5 * (detap - tm)
        spm = 0.5 * (desap - sm)
        fi = t2.expand(-1, -1, nw) + 0.5 * (detap * s2g + tm * c2g)
        fq = tpm * s2g * cf
        fu = tpm * s2g * sf
        fv = tn * cg
        fq1 = spm * s2g * cf
        fu1 = spm * s2g * sf
        fv1 = sn * cg
        dkPe = _matabs(fi, fq, fu, fv, fq1, fu1, fv1)

        # ---------- dK/dvmic ----------------------------------------------
        detar = meta0.expand(-1, -1, nw) * etar + t3 * (ettar * ma.expand(-1, -1, nw) + ettmr)
        detal = meta0.expand(-1, -1, nw) * etal + t3 * (ettal * ma.expand(-1, -1, nw) + ettml)
        detap = meta0.expand(-1, -1, nw) * etap + t3 * (ettap * ma.expand(-1, -1, nw) + ettmp)
        desar = meta0.expand(-1, -1, nw) * esar + t3 * (essar * ma.expand(-1, -1, nw) + essmr)
        desal = meta0.expand(-1, -1, nw) * esal + t3 * (essal * ma.expand(-1, -1, nw) + essml)
        desap = meta0.expand(-1, -1, nw) * esap + t3 * (essap * ma.expand(-1, -1, nw) + essmp)
        tm = 0.5 * (detar + detal)
        tn = 0.5 * (detar - detal)
        sm = 0.5 * (desar + desal)
        sn = 0.5 * (desar - desal)
        tpm = 0.5 * (detap - tm)
        spm = 0.5 * (desap - sm)
        fi = 0.5 * (detap * s2g + tm * c2g)
        fq = tpm * s2g * cf
        fu = tpm * s2g * sf
        fv = tn * cg
        fq1 = spm * s2g * cf
        fu1 = spm * s2g * sf
        fv1 = sn * cg
        dkVmic = _matabs(fi, fq, fu, fv, fq1, fu1, fv1)

        # ---------- dK/dB --------------------------------------------------
        tm = 0.5 * (getar + getal)
        tn = 0.5 * (getar - getal)
        sm = 0.5 * (gesar + gesal)
        sn = 0.5 * (gesar - gesal)
        tpm = 0.5 * (getap - tm)
        spm = 0.5 * (gesap - sm)
        fi = t3 * 0.5 * (getap * s2g + tm * c2g)
        fq = t3 * tpm * s2g * cf
        fu = t3 * tpm * s2g * sf
        fv = t3 * tn * cg
        fq1 = t3 * spm * s2g * cf
        fu1 = t3 * spm * s2g * sf
        fv1 = t3 * sn * cg
        dkB = _matabs(fi, fq, fu, fv, fq1, fu1, fv1)

        # ---------- dK/dvlos ----------------------------------------------
        tm = 0.5 * (vetar + vetal)
        tn = 0.5 * (vetar - vetal)
        sm = 0.5 * (vesar + vesal)
        sn = 0.5 * (vesar - vesal)
        tpm = 0.5 * (vetap - tm)
        spm = 0.5 * (vesap - sm)
        fi = t3 * 0.5 * (vetap * s2g + tm * c2g)
        fq = t3 * tpm * s2g * cf
        fu = t3 * tpm * s2g * sf
        fv = t3 * tn * cg
        fq1 = t3 * spm * s2g * cf
        fu1 = t3 * spm * s2g * sf
        fv1 = t3 * sn * cg
        dkVlos = _matabs(fi, fq, fu, fv, fq1, fu1, fv1)

        # ---------- dK/dgamma ---------------------------------------------
        fi = 2.0 * t3 * _tpm0(etap, etar, etal) * scg
        fq = fi * cf
        fu = fi * sf
        fv = -t3 * tn0(etar, etal) * sg
        fi1 = 2.0 * t3 * _spm0(esap, esar, esal) * scg
        fq1 = fi1 * cf
        fu1 = fi1 * sf
        fv1 = -t3 * sn0(esar, esal) * sg
        dkGamma = _matabs(fi, fq, fu, fv, fq1, fu1, fv1)

        # ---------- dK/dphi -----------------------------------------------
        fi = torch.zeros_like(t3)
        fq = -2.0 * t3 * _tpm0(etap, etar, etal) * s2g * sf
        fu = 2.0 * t3 * _tpm0(etap, etar, etal) * s2g * cf
        fv = torch.zeros_like(t3)
        fq1 = -2.0 * t3 * _spm0(esap, esar, esal) * s2g * sf
        fu1 = 2.0 * t3 * _spm0(esap, esar, esal) * s2g * cf
        fv1 = torch.zeros_like(t3)
        dkPhi = _matabs(fi, fq, fu, fv, fq1, fu1, fv1)

        # vlos/vmic enter as cm/s inside the K builder, but the atmosphere
        # (and the RF column convention) is in km/s, so convert the
        # derivative to per-km/s units.
        dkVlos = dkVlos * 1e5
        dkVmic = dkVmic * 1e5

        k_line = absorption_matrix(torch.zeros_like(t3), t3,
                                   etap, etar, etal,
                                   esap, esar, esal,
                                   gamma, phi)
        return k_line, {"T": dkT, "Pe": dkPe, "B": dkB,
                        "gamma": dkGamma, "phi": dkPhi,
                        "vlos": dkVlos, "vmic": dkVmic}

    def _rf_analytic_chain(self, wavs, ltau, atmos, pe_not_inverted=False):
        """
        Full-analytic response functions: hand-written chain rule.

        Ionization-equilibrium derivatives -> continuum/line opacity and
        damping derivative tables -> dK/dx per quantity -> variational
        solve ``dR/dtau = K R + Q`` (hermite_solve_general).

        No autograd and no jacfwd: every derivative is the analytic
        formula.  Returns (Nb, Nw, 4, Nx) as :meth:`_rf_analytic`.
        """
        from ..physics.pressure import ionization_equilibrium_derivatives
        nt = ltau.shape[0]
        nw = wavs.shape[0]
        nb = atmos.shape[0]
        nx = atmos.shape[1]
        t, pe, b, gamma, phi, vlos, vmic, vmac = self._unpack(atmos, nt)
        theta = 5040.0 / t

        # ionization-equilibrium derivatives (d ln p/dT, d ln p/dPe)
        pp, dpp, ddpp = ionization_equilibrium_derivatives(
            theta, pe, abundance=self.abundance)

        k_matrix, svec, dk = self._K_source_derivatives(
            wavs, ltau, atmos, pp, dpp, ddpp)

        i_top = hermite_solve(ltau, k_matrix, svec)
        if self._normalize:
            wl1_cm = self._line_static[0]["wvac"]
            wlo, whi = wavs.min(), wavs.max()
            for static in self._line_static:
                wl = static["wvac"] * 1e8
                if bool((wlo <= wl) & (wl <= whi)):
                    wl1_cm = static["wvac"]
                    break
            cont_norm = hsra_continuum(float(wl1_cm * 1e8))
        else:
            cont_norm = 1.0

        i_minus_s = self._i_minus_s(ltau, k_matrix, svec, i_top)

        # Q(x,i) = (dK/dx)(I-S) - K(dS/dx), per-dlogtau units (ln10*tau)
        fac = _LOG10 * (10.0 ** ltau)[None, :, None, None]
        q_names = ["T", "Pe", "B", "gamma", "phi", "vlos", "vmic"]
        i0_of = {"T": 0, "Pe": nt, "B": 2 * nt, "gamma": 3 * nt,
                 "phi": 4 * nt, "vlos": 5 * nt, "vmic": 6 * nt}
        q_all = []
        for name in q_names:
            dkx = dk[name]                                   # (Nb,Nt,Nw,4,4)
            dk_is = torch.einsum("ntwxy,ntwy->ntwx", dkx, i_minus_s)
            if name == "T":
                dsource = self._dsource_dt(t, wl1_cm).squeeze(-1)
                dsource = dsource.unsqueeze(-1).expand(-1, -1, nw)
                dsvec = torch.stack(
                    [dsource, torch.zeros_like(dsource),
                     torch.zeros_like(dsource), torch.zeros_like(dsource)],
                    dim=-1)
                kds = torch.einsum("btwxy,btwy->btwx", k_matrix, dsvec)
                qj = dk_is - kds
            else:
                qj = dk_is
            qj = qj * fac
            if name == "vmic":
                q_all.append(qj.unsqueeze(1))
            else:
                q_all.append(qj.unsqueeze(1).expand(
                    nb, nt, nt, nw, 4).contiguous()
                    .masked_fill(~torch.eye(nt, dtype=torch.bool,
                                            device=self.device)
                                 .view(1, nt, nt, 1, 1), 0.0))
        q = torch.cat(q_all, dim=1)

        rf_int = hermite_solve_general(ltau, k_matrix, q)

        rf = torch.zeros((nb, nw, 4, nx), device=self.device, dtype=self.dtype)
        src = 0
        for name in q_names:
            if name == "vmic":
                rfq = rf_int[:, src]
                src += 1
                i0 = i0_of[name]
                if self._normalize:
                    rfq = rfq / cont_norm
                rf[:, :, :, i0:i0 + 1] = rfq.unsqueeze(-1)
            else:
                block = rf_int[:, src:src + nt]
                src += nt
                if self._normalize:
                    block = block / cont_norm
                i0 = i0_of[name]
                # NOTE (units): _unpack feeds gamma/phi to the K builder in
                # RADIANS, so the hand-written dK blocks (and the whole
                # dI/d(gamma) column) are per-radian; the atmosphere and
                # the node Jacobian convention are per-DEGREE.  _rf_analytic
                # (jacfwd) applies _unit_scale at reassembly; the chain
                # must apply the same rad->deg factor here (vlos/vmic are
                # already carried in km/s by the dK blocks, T/Pe/B are
                # unit-less).
                unit = float(np.pi / 180.0) if name in ("gamma", "phi") \
                    else 1.0
                rf[:, :, :, i0:i0 + nt] = block.permute(0, 2, 3, 1) * unit
        if pe_not_inverted:
            rf = self._hse_rf_correction(ltau, t, pe, pp, dpp, ddpp, rf, nt)
        return rf

    # ------------------------------------------------------------------
    # 'fast' response functions: wavelength-expanded autograd with a
    # macroturbulence-convolution correction (pyPRT-style; see
    # pyprt/synth/response_function.py).
    #
    # Idea (pyPRT): without the macroturbulence convolution every
    # (batch, wavelength, Stokes) element of the synthesis is an
    # independent computation of the same atmosphere, so the atmosphere
    # can be broadcast to (Nb, Nx, Nw) and the full Jacobian
    # dI/dx -> (Nb, Nw, 4, Nx) is obtained with ONE forward + 4
    # backward passes of torch.autograd (no vmap/jacrev overhead).
    # The wavelength-coupling introduced by the macroturbulence
    # convolution is then corrected analytically afterwards.
    # ------------------------------------------------------------------
    def _forward_fast_nomac(self, wavs, ltau, xx):
        """
        Differentiable forward WITHOUT macroturbulence on the
        wavelength-expanded atmosphere ``xx`` of shape (Nb, Nx, Nw):
        every (batch, wavelength) pair is an independent synthesis
        (identical physics/units as :meth:`_forward`, only the final
        convolution step is omitted).

        Returns ``(stokes (Nb, Nw, 4), wl1_air)``.
        """
        nt = ltau.shape[0]
        nw = wavs.shape[0]
        # unpack (per-wavelength) with the SAME chain factors as
        # _unpack: angles deg -> rad, velocities km/s -> cm/s
        t = xx[:, :nt, :]                                   # (Nb,Nt,Nw)
        pe = xx[:, nt:2 * nt, :].clamp(min=1e-10)
        b = xx[:, 2 * nt:3 * nt, :]
        gamma = xx[:, 3 * nt:4 * nt, :] * (np.pi / 180.0)
        phi = xx[:, 4 * nt:5 * nt, :] * (np.pi / 180.0)
        vlos = xx[:, 5 * nt:6 * nt, :] * 1e5
        vmic = xx[:, 6 * nt:6 * nt + 1, :] * 1e5            # (Nb,1,Nw)
        theta = 5040.0 / t
        pp = ionization_equilibrium(theta, pe, abundance=self.abundance)
        k_matrix, svec, wl1_cm, wl1_air = self._K_source(
            wavs, t, pe, b, gamma, phi, vlos, vmic, theta, pp, nt, nw)
        stokes = self._rte(ltau, k_matrix, svec)        # (Nb,Nw,4)
        if self._normalize:
            cont_norm = hsra_continuum(float(wl1_air))      # AIR lambda1
            stokes = stokes / cont_norm
        return stokes, wl1_air

    def _rf_fast(self, wavs, ltau, atmos, pe_not_inverted=False):
        """
        pyPRT-style fast response functions (rf_method == 'fast').

        1. broadcast the atmosphere (Nb, Nx) to (Nb, Nx, Nw): the same
           parameter set, one copy per wavelength;
        2. run the forward synthesis WITHOUT macroturbulence (per
           wavelength independent);
        3. one autograd pass per Stokes component yields the full
           Jacobian (Nb, Nw, 4, Nx) of the *un*-convolved profiles;
        4. correct the macroturbulence convolution analytically
           (:meth:`_fix_fast_mac`): the wavelength coupling moves the
           derivatives dS/dx through the kernel and the vmac column is
           the kernel-width derivative.

        Returns (Nb, Nw, 4, Nx), identical to :meth:`_rf_autograd`
        (same columns/layout/units as the other rf_methods).
        """
        nt = ltau.shape[0]
        nw = wavs.shape[0]
        nb = atmos.shape[0]
        nx = atmos.shape[1]

        xx = atmos.unsqueeze(-1).expand(nb, nx, nw)
        xx = xx.contiguous().requires_grad_(True)
        stokes, wl1_air = self._forward_fast_nomac(wavs, ltau, xx)

        rf = torch.empty((nb, nw, 4, nx), device=self.device, dtype=self.dtype)
        for s_ in range(4):
            # each (b, w) element depends only on its own copy xx[b, :, w],
            # so the summed gradient is the per-element Jacobian
            g = torch.autograd.grad(
                stokes[..., s_], xx, torch.ones_like(stokes[..., s_]),
                retain_graph=True)[0]
            rf[:, :, s_, :] = g.transpose(1, 2)           # (Nb,Nx,Nw)->(Nb,Nw,Nx)
        del xx

        if self._use_mac:
            rf = self._fix_fast_mac(wavs, stokes.detach(), rf, atmos, wl1_air)
        if pe_not_inverted:
            from ..physics.pressure import ionization_equilibrium_derivatives
            t, pe, b, gamma, phi, vlos, vmic, vmac = self._unpack(atmos, nt)
            theta = 5040.0 / t
            pp, dpp, ddpp = ionization_equilibrium_derivatives(
                theta, pe, abundance=self.abundance)
            rf = self._hse_rf_correction(ltau, t, pe, pp, dpp, ddpp, rf, nt)
        return _sanitize_rf(rf)

    def _fix_fast_mac(self, wavs, stokes, rf, atmos, wl1_air):
        """
        Analytic macroturbulence correction of the un-convolved RF.

        With Ŝ = conv(S, K_vmac) (K = normalized Gaussian kernel,
        sigma = lambda0 * vmac / c, in the pixel convention of
        :meth:`_macroturbulence`):

          (a) for every parameter x != vmac the kernel is independent
              of x, so  dŜ/dx = conv(dS/dx, K);
          (b) for x = vmac the kernel width itself moves, and the exact
              result for the discrete kernel
                  dK_k/dsigma = K_k (k^2 - kbar^2) / sigma^3,
              kbar^2 = sum_k K_k k^2, gives
                  dŜ/dvmac = (dsigma/dvmac) [conv(S, K k^2) - kbar^2 Ŝ]
                              / sigma^3.

        All convolutions use the same kernel/radius/replicate-padding as
        :meth:`_macroturbulence`, so the result reproduces the exact
        autograd gradient of the full forward model.
        """
        # vmac arrives in cm/s (vmac_cm = atmos * 1e5) -> use c in cm/s,
        # matching _macroturbulence (fix 372b107); a c in km/s made the
        # correction kernel ~1e5 px wide and broke fast vs autograd.
        c_cm = 2.99792458e10
        nx = rf.shape[-1]
        icol = nx - 1                                     # vmac column
        vmac_cm = atmos[:, icol] * 1e5                     # (Nb,)
        lambda0 = float(wl1_air)                       # AIR
        paso = float(torch.diff(wavs).abs().min())
        sigma_pix = lambda0 * vmac_cm / c_cm / paso
        sigma_used = sigma_pix.clamp(min=1e-6)
        assert lambda0 > 0 and paso > 0

        radius = wavs.shape[0] - 1
        x = torch.arange(-radius, radius + 1, device=self.device,
                         dtype=self.dtype)                 # (Nk,)
        kernels = torch.exp(-0.5 * (x[None, :] / sigma_used[:, None]) ** 2)
        kernels = kernels / kernels.sum(dim=1, keepdim=True)   # (Nb, Nk)

        # (a) wavelength coupling of every parameter column
        # (rf: (Nb,Nw,4,Nx) -> (Nb,4,Nx,Nw), convolve along Nw, back)
        rf = self._convolve_w(rf.permute(0, 2, 3, 1), kernels, radius) \
            .permute(0, 3, 1, 2)

        # (b) vmac column: kernel-width derivative
        kbar2 = (kernels * x[None, :] ** 2).sum(dim=1)         # (Nb,)
        s_obs = self._convolve_w(stokes.permute(0, 2, 1),
                                 kernels, radius).permute(0, 2, 1)
        s2 = self._convolve_w(stokes.permute(0, 2, 1),
                              kernels * x[None, :] ** 2,
                              radius).permute(0, 2, 1)
        # d sigma_pix / d (vmac in km/s): sigma = lambda0*vmac_cm/(c_cm*paso),
        # vmac_cm = atmos[:, icol] * 1e5  ->  factor 1e5*lambda0/(c_cm*paso)
        dsigma = torch.where(
            sigma_pix > 1e-6,
            torch.full_like(sigma_pix, lambda0 * 1e5 / (c_cm * paso)),
            torch.zeros_like(sigma_pix))
        d_obs_sigma = (s2 - kbar2[:, None, None] * s_obs) \
            / sigma_used[:, None, None] ** 3
        rf[..., icol] = d_obs_sigma * dsigma[:, None, None]
        return rf

    def _convolve_w(self, x, kernels, radius):
        """
        Convolve ``x`` of shape (Nb, ..., Nw) along the wavelength axis
        with per-batch kernels (Nb, 2*radius+1) using replicate padding
        — exactly the discretization of :meth:`_macroturbulence`.
        """
        nb = x.shape[0]
        nw = x.shape[-1]
        lead = x.reshape(nb, -1, nw)                        # (Nb, C, Nw)
        padded = torch.nn.functional.pad(
            lead, (radius, radius), mode="replicate")
        unfolded = padded.unfold(-1, 2 * radius + 1, 1)
        # NOTE: kernels must be (Nb, 1, 1, Nk) — a 3-D (Nb, 1, Nk) tensor
        # aligns its batch dim against the *channel* dim of the 4-D
        # unfolded tensor and breaks for Nb > 1 (it only works by chance
        # for Nb = 1, where the batch dim is a trailing singleton).
        out = (unfolded * kernels[:, None, None, :]).sum(-1)
        return out.reshape(x.shape)

    def _macroturbulence(self, stokes, vmac, wavs, wl1_air):
        """Gaussian convolution with width sigma = lambda0 * vmac / c.

        sigma [pixels] = lambda0 * vmac / (c * dwavelength) with lambda0
        the AIR wavelength of the region's first line, vmac the
        macroturbulent velocity and c the speed of light; the kernel is
        normalized to unit area so the convolution preserves the line
        flux.
        """
        # vmac arrives in cm/s (see _unpack) -> use c in cm/s.  Dividing
        # by c in km/s instead makes sigma_pix ~3e5 px for any non-tiny
        # vmac and the kernel degenerates into a full-band average that
        # flattens every spectral line (observed with xnet vmac ~ 4 km/s
        # and even 0.1 km/s on the Hinode grid).
        c_cm = 2.99792458e10
        lambda0 = float(wl1_air)                     # Angstrom (AIR)
        paso = float(torch.diff(wavs).abs().min())
        sigma_pix = (lambda0 * vmac / c_cm / paso)       # (Nb,) in pixels
        # fixed kernel radius (maximum possible): vmap-safe (no .item() on
        # batched data); cheap since the kernel is tiny vs Nw
        rmax = wavs.shape[0] - 1
        x = torch.arange(-rmax, rmax + 1, device=self.device, dtype=self.dtype)
        kernels = torch.exp(-0.5 * (x[None, :] / sigma_pix.clamp(min=1e-6)[:, None]) ** 2)
        kernels = kernels / kernels.sum(dim=1, keepdim=True)   # (Nb, 2r+1)

        # per-batch convolution via unfold (differentiable, vmap-safe)
        padded = torch.nn.functional.pad(
            stokes.permute(0, 2, 1), (rmax, rmax), mode="replicate")  # (Nb,4,Nw+2r)
        unfolded = padded.unfold(-1, 2 * rmax + 1, 1)       # (Nb,4,Nw,2r+1)
        out = (unfolded * kernels[:, None, None, :]).sum(-1)
        return out.permute(0, 2, 1)                          # (Nb,Nw,4)

    def _hse_rf_correction(self, ltau, t, pe, pp, dpp, ddpp, rf, nt):
        """
        Hydrostatic coupling applied to a grid response function.

        When Pe is not a free parameter of the inversion, the atmosphere
        is kept in hydrostatic equilibrium: a temperature change at
        depth kk drags the electron pressure along, dPe(j)/dT(kk) = M.
        The effective T response function is
        ``grt(kk) = rt4(isv,kk) + sum_j grp(j) M(j,kk)``, i.e.

            rf_T_new = rf_T + rf_Pe @ M

        M is assembled by :func:`~spot.physics.hydrostatic.
        hydrostatic_pe_coupling` from the ionization-equilibrium
        derivatives at the current state; kappa5/dkappa5/ddkappa5 are
        the 5000 A continuum opacity derivatives (converted per-mass
        inside the operator).

        Parameters
        ----------
        rf : torch.Tensor (Nb, Nw, 4, Nx)
            Grid response function whose columns are ordered T, Pe, B,
            gamma, phi, vlos, vmic, vmac (each depth block of Nt plus
            the two scalar velocity columns).

        Returns the corrected rf (same shape).
        """
        from ..physics.hydrostatic import hydrostatic_pe_coupling
        kappa5, dkappa5, ddkappa5 = _opacity.continuum_opacity_derivatives(
            torch.tensor([5000.0], dtype=self.dtype, device=self.device),
            t, pe, pp, dpp, ddpp, refractive_index=1.0)
        m = hydrostatic_pe_coupling(ltau, t, pe, pp, dpp, ddpp,
                                    kappa5, dkappa5, ddkappa5,
                                    abundance=self.abundance)
        rf_pe = rf[..., nt:2 * nt]                        # (Nb,Nw,4,Nt)
        rf_t = (rf[..., :nt]
                + torch.einsum("bwsi,bij->bwsj", rf_pe, m))
        return torch.cat([rf_t, rf[..., nt:]], dim=-1)

    # ------------------------------------------------------------------
    # response functions
    # ------------------------------------------------------------------
    def response_functions(self, wavs, ltau, atmos, pe_not_inverted=False):
        """
        Response functions dI/dx of shape (Nb, Nw, 4, Nx).

        With ``rf_method == 'autograd'`` the exact Jacobian of the
        forward model is computed; with ``'analytic'`` the variational
        (linearized) equation is solved (faster: the propagation matrix
        is computed once and shared by every parameter); with
        ``'analytic_chain'`` the same variational equation is solved but
        the source derivatives dK/dx and dS/dx come from the hand-written
        chain rule (derivatives.py + _K_source_derivatives +
        _line_matrix_derivatives, zero autograd) — the path used by the
        benchmark inversion; with ``'finite_diff'`` a central difference
        scheme with the configurable relative step ``rf_eps`` and
        per-quantity absolute scales ``rf_scale`` is used (most
        memory-efficient for very large batches); with ``'fast'`` the
        pyPRT-style wavelength-expanded autograd is used: the
        atmosphere is broadcast to (Nb, Nx, Nw), the forward is run
        once WITHOUT macroturbulence (all wavelength points
        independent), 4 autograd backward passes give the exact
        Jacobian of the un-convolved profiles, and the
        macroturbulence-convolution coupling is corrected analytically
        afterwards (:meth:`_rf_fast`; mathematically identical to
        ``'autograd'``, but only a single graph is built instead of a
        vmap(jacrev) — much cheaper, the pyPRT 'fast' mode).
        """
        method = self.config["synthesis"]["rf_method"]
        if method in ("analytic", "analytic_chain") \
                and self._cont_opacity != "mihalas":
            raise ValueError(
                f"analytic response functions (rf_method='{method}') are "
                "only defined for the mihalas continuum opacity; use "
                "rf_method='fast' or 'autograd' with "
                f"continuum_opacity={self._cont_opacity!r}")
        if method == "autograd":
            return self._rf_autograd(wavs, ltau, atmos)
        if method == "fast":
            return self._rf_fast(wavs, ltau, atmos,
                                 pe_not_inverted=pe_not_inverted)
        if method in ("analytic", "analytic_chain"):
            if method == "analytic_chain":
                return self._rf_analytic_chain(
                    wavs, ltau, atmos, pe_not_inverted=pe_not_inverted)
            return self._rf_analytic(wavs, ltau, atmos,
                                     pe_not_inverted=pe_not_inverted)
        return self._rf_finite_diff(wavs, ltau, atmos)

    def _rf_analytic(self, wavs, ltau, atmos, pe_not_inverted=False):
        """
        Analytic response functions.

        The RF satisfy the variational (linearized) RTE

            dR/dtau = K R + Q,     Q = (dK/dx)(I - S) - dS/dx

        Because the equation is linear, the solution is
        ``RF(x) = sum_i OO(i) Q(x,i)`` where ``OO(i)`` is the
        propagation (Green) operator from depth i to the surface.  OO
        depends only on K and is computed once; the local sources
        Q(x,i) are obtained by forward-mode differentiation (jacfwd) of
        the K-building function only (NOT of the whole forward, which
        would cost one full forward per parameter).

        Returns (Nb, Nw, 4, Nx) with the same column order as atmos.
        """
        from torch.func import jacfwd, vmap

        nt = ltau.shape[0]
        nw = wavs.shape[0]
        nb = atmos.shape[0]
        nx = atmos.shape[1]

        # unpack in internal units (rad, cm/s), as in _forward
        t, pe, b, gamma, phi, vlos, vmic, vmac = self._unpack(atmos, nt)
        theta = 5040.0 / t
        pp = ionization_equilibrium(theta, pe, abundance=self.abundance)

        # one forward pass: K, S and the full Stokes solution I
        k_matrix, svec, wl1_cm, wl1_air = self._K_source(
            wavs, t, pe, b, gamma, phi, vlos, vmic, theta, pp, nt, nw)
        i_top = hermite_solve(ltau, k_matrix, svec)   # (Nb,Nw,4)
        if self._normalize:
            cont_norm = hsra_continuum(float(wl1_air))    # AIR lambda1
        else:
            cont_norm = 1.0

        # per-layer (I - S) (fixed forward solution, not differentiated)
        i_minus_s = self._i_minus_s(ltau, k_matrix, svec, i_top)  # (Nb,Nt,Nw,4)

        q_names = ["T", "Pe", "B", "gamma", "phi", "vlos", "vmic"]

        def kfun_batch(tq, pq, bq, gq, phiq, vq, mq):
            return self._K_source(wavs, tq, pq, bq, gq, phiq, vq, mq,
                                  theta, pp, nt, nw)[0]

        def kfun_sample(tq, pq, bq, gq, phiq, vq, mq):
            # NOTE: velocities are fed to _K_source in their input units
            # (km/s) although _K_source expects cm/s; the Jacobian is
            # therefore dK/d(cm/s) and the reassembly multiplies by 1e5
            # to recover dK/d(km/s).  (Scaling vq*1e5 inside the vmap
            # breaks the other argnums' Jacobians in torch.func.)
            return kfun_batch(tq[None], pq[None], bq[None], gq[None],
                              phiq[None], vq[None], mq[None])[0]

        inputs = (t, pe, b, gamma, phi, vlos, vmic)

        def per_sample_jac(*xs):
            return jacfwd(kfun_sample, argnums=tuple(range(7)))(*xs)

        jacs = vmap(per_sample_jac)(*inputs)

        rf = torch.zeros((nb, nw, 4, nx), device=self.device, dtype=self.dtype)
        i0_of = {"T": 0, "Pe": nt, "B": 2 * nt, "gamma": 3 * nt,
                 "phi": 4 * nt, "vlos": 5 * nt, "vmic": 6 * nt}
        q_all = []
        for name, jac in zip(q_names, jacs):
            # jac: (Nb, 1, Nt, Nw, 4, 4, Nt, 1) for depth quantities and
            #      (Nb, 1, Nt, Nw, 4, 4, 1, 1) for vmic
            j6 = jac.squeeze(1).squeeze(-1)          # (Nb, Nt, Nw, 4, 4, Nt|1)
            if j6.shape[-1] == 1:
                # vmic: a single scalar input, affects every depth
                dk = j6[:, :, :, :, :, 0]            # (Nb, Nt, Nw, 4, 4)
            else:
                # depth quantity: dK/dx is diagonal in depth (K is
                # layer-local), extract the diagonal along dims 1 and 5
                dk = torch.diagonal(j6, dim1=1, dim2=5)  # (Nb, Nw, 4, 4, Nt)
                dk = dk.permute(0, 4, 1, 2, 3)       # (Nb, Nt, Nw, 4, 4)
            # local source Q(i) = +(dK/dx)(I-S) - K(dS/dx) at depth i
            # (variational equation: dR/dtau = K R + Q with
            # Q = (dK/dx)(I-S) - K(dS/dx); the K*dS term is non-zero
            # only for T, where dS/dT is the Planck derivative)
            dk_is = torch.einsum("btwak,btwk->btwa", dk, i_minus_s)
            if name == "T":
                dsource = self._dsource_dt(t, wl1_cm).squeeze(-1)  # (Nb,Nt)
                dsource = dsource.unsqueeze(-1).expand(-1, -1, nw)  # (Nb,Nt,Nw)
                dsvec = torch.stack(
                    [dsource, torch.zeros_like(dsource),
                     torch.zeros_like(dsource), torch.zeros_like(dsource)],
                    dim=-1)                                       # (Nb,Nt,Nw,4)
                kds = torch.einsum("btwxy,btwy->btwx", k_matrix, dsvec)
                qj = dk_is - kds
            else:
                qj = dk_is
            # per-dlogtau units: multiply by ln10 * tau (K' = K ln10 tau)
            fac = _LOG10 * (10.0 ** ltau)[None, :, None, None]
            qj = qj * fac
            if name == "vmic":
                # vmic is a depth-independent scalar: one source affecting
                # every depth (the whole column moves together)
                q_all.append(qj.unsqueeze(1))          # (Nb,1,Nt,Nw,4)
            else:
                # per-depth response: one source per depth (only that
                # layer is perturbed), giving dI/dx_i for each column i
                q_all.append(qj.unsqueeze(1).expand(
                    nb, nt, nt, nw, 4).contiguous()
                    .masked_fill(~torch.eye(nt, dtype=torch.bool,
                                            device=self.device)
                                 .view(1, nt, nt, 1, 1), 0.0))
        q = torch.cat(q_all, dim=1)                    # (Nb, Np_src, Nt, Nw, 4)

        # batched variational solve: all sources share K (the layer
        # matrices and icontorno are computed once inside the solver)
        rf_int = hermite_solve_general(ltau, k_matrix, q)   # (Nb,Np_src,Nw,4)

        src = 0
        for j, name in enumerate(q_names):
            if name == "vmic":
                rfq = rf_int[:, src]               # (Nb,Nw,4) whole-column
                src += 1
                i0 = i0_of[name]
                if self._normalize:
                    rfq = rfq / cont_norm
                rf[:, :, :, i0:i0 + 1] = rfq.unsqueeze(-1) \
                    * self._unit_scale(name, nt)
            else:
                block = rf_int[:, src:src + nt]    # (Nb,Nt,Nw,4) per-depth
                src += nt
                if self._normalize:
                    block = block / cont_norm
                i0 = i0_of[name]
                # (Nb, Nt, Nw, 4) -> (Nb, Nw, 4, Nt)
                rf[:, :, :, i0:i0 + nt] = block.permute(0, 2, 3, 1) \
                    * self._unit_scale(name, nt)
        if pe_not_inverted:
            from ..physics.pressure import ionization_equilibrium_derivatives
            pp, dpp, ddpp = ionization_equilibrium_derivatives(
                theta, pe, abundance=self.abundance)
            rf = self._hse_rf_correction(ltau, t, pe, pp, dpp, ddpp, rf, nt)
        return rf

    def _i_minus_s(self, ltau, k_matrix, svec, i_top=None):
        """Per-layer (I - S) of the forward solution (analytic)."""
        if i_top is None:
            i_top = hermite_solve(ltau, k_matrix, svec)   # (Nb,Nw,4)
        _, layers = hermite_solve(ltau, k_matrix, svec, return_layers=True)
        return layers - svec                      # (Nb,Nt,Nw,4)

    @staticmethod
    def _dsource_dt(t, wl1_cm):
        """d(Planck)/dT at the first line's wavelength (analytic Planck
        derivative used in the source term of the response function)."""
        # B(T) = 2 h c^2 / lambda^5 / (exp(hc/lambda kT) - 1)
        h = 6.6262e-27
        c = 2.99792458e10
        k = 1.38054e-16
        lam = wl1_cm
        x = h * c / (lam * k * t)
        ex = torch.exp(x)
        b = 2.0 * h * c * c / (lam ** 5) / (ex - 1.0)
        db = b * x / t * ex / (ex - 1.0)                # dB/dT
        return db                                       # (Nb,Nt,1)

    @staticmethod
    def _unit_scale(name, nt):
        """Chain rule factor from the Jacobian's units to input units.

        The Jacobian is dK/d(input value as fed to _K_source): for the
        velocities the input is in km/s but _K_source interprets it as
        cm/s, so dK/d(km/s) = dK/d(cm/s) * 1e5.  Angles are fed in rad
        (already converted by _unpack) while the atmos stores degrees,
        so dK/d(deg) = dK/d(rad) * pi/180.
        """
        if name in ("gamma", "phi"):
            return np.pi / 180.0                         # rad -> deg
        if name in ("vlos", "vmic"):
            return 1e5                                   # cm/s -> km/s
        return 1.0                                       # T, Pe, B

    def _rf_autograd(self, wavs, ltau, atmos):
        """Exact Jacobian via torch.autograd (batched)."""
        try:
            from torch.func import jacrev, vmap

            def single(x):
                return self._forward(wavs, ltau, x[None])[0]     # (Nw,4)

            rf = vmap(jacrev(single))(atmos.detach().requires_grad_(True))
            return _sanitize_rf(rf.detach())
        except Exception:
            # fallback: per-sample autograd
            rf = torch.zeros((atmos.shape[0], wavs.shape[0], 4, atmos.shape[1]),
                             device=self.device, dtype=self.dtype)
            for i in range(atmos.shape[0]):
                xi = atmos[i].detach().clone().requires_grad_(True)
                out = self._forward(wavs, ltau, xi[None])[0]     # (Nw,4)
                for iw in range(wavs.shape[0]):
                    for is_ in range(4):
                        g = torch.autograd.grad(
                            out[iw, is_], xi, retain_graph=True)[0]
                        rf[i, iw, is_] = g
            return rf

    def _rf_finite_diff(self, wavs, ltau, atmos):
        """Central finite-difference response functions.

        The step is adaptive: ``step = max(scale, |x|) * eps`` per
        element, so quantities with a small local value (e.g. electron
        pressure in the upper atmosphere) still get a well-resolved
        perturbation instead of a huge relative one.  The differences
        are evaluated internally in float64 to avoid the catastrophic
        cancellation that a float32 Stokes vector would suffer.
        """
        eps = float(self.config["synthesis"]["rf_eps"])
        scales = self.config["synthesis"]["rf_scale"]
        nt = ltau.shape[0]
        block = {
            "T": (0, nt), "Pe": (nt, 2 * nt), "B": (2 * nt, 3 * nt),
            "gamma": (3 * nt, 4 * nt), "phi": (4 * nt, 5 * nt),
            "vlos": (5 * nt, 6 * nt), "vmic": (6 * nt, 6 * nt + 1),
            "vmac": (6 * nt + 1, 6 * nt + 2),
        }
        rf = torch.zeros((atmos.shape[0], wavs.shape[0], 4, atmos.shape[1]),
                         device=self.device, dtype=self.dtype)
        for name, (i0, i1) in block.items():
            scale = float(scales[name])
            for j in range(i0, i1):
                step = torch.maximum(torch.full_like(atmos[:, j], scale * eps),
                                     atmos[:, j].abs() * eps).clamp(min=1e-30)
                xp = atmos.clone().double()
                xm = atmos.clone().double()
                xp[:, j] = xp[:, j] + step.double()
                xm[:, j] = xm[:, j] - step.double()
                rf[:, :, :, j] = (self._forward(wavs.double(), ltau.double(), xp)
                                  - self._forward(wavs.double(), ltau.double(), xm)
                                  ) / (2 * step.double())[:, None, None]
        return rf


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _atomic_weight(z):
    from ..physics.atoms import ATOM_WEIGHTS
    return float(ATOM_WEIGHTS[z - 1])


def _partition(atom, t):
    from ..physics.partition import partition_functions
    z = 26 if atom.upper() in ("XX", "FE") else element_symbol_to_number(atom)
    return partition_functions(z, t)


def _continuum_matrix(t0):
    """Diagonal continuum absorption matrix (etaI only)."""
    zero = torch.zeros_like(t0)
    return torch.stack([
        torch.stack([t0, zero, zero, zero], dim=-1),
        torch.stack([zero, t0, zero, zero], dim=-1),
        torch.stack([zero, zero, t0, zero], dim=-1),
        torch.stack([zero, zero, zero, t0], dim=-1),
    ], dim=-2)


def _continuum_matrix_derivatives(t1, t2):
    """Derivatives of the continuum matrix wrt T (t1) and Pe (t2)."""
    return {"T": _continuum_matrix(t1), "Pe": _continuum_matrix(t2)}


def _matabs(fi, fq, fu, fv, fq1, fu1, fv1):
    """4x4 propagation matrix from the 7 elements.

    NOTE on signs: spot's forward ``absorption_matrix`` uses positive
    Faraday (rho) profiles and compensates by negating the dispersion
    elements (``fq1 = -t3*fq1`` etc., see absorption.py), whereas the
    reference formulation's rho carries the implicit minus sign
    internally and feeds the raw ``fq1/fu1/fv1`` into the matrix.  This
    helper applies the same negation here so that the analytic dK
    matrices match the forward convention (and hence the
    jacfwd/autograd K-derivatives) exactly: the mirrored entries carry
    the reference layout signs while the input dispersion elements are
    already sign-flipped.
    """
    fq1, fu1, fv1 = -fq1, -fu1, -fv1
    return torch.stack([
        torch.stack([fi, fq, fu, fv], dim=-1),
        torch.stack([fq, fi, -fv1, fu1], dim=-1),
        torch.stack([fu, fv1, fi, -fq1], dim=-1),
        torch.stack([fv, -fu1, fq1, fi], dim=-1),
    ], dim=-2)


# --- derivative helpers: profile combinations ------------------------------
def _tpm0(etap, etar, etal):
    """tpm = 0.5*(etap - tm), tm = 0.5*(etar+etal)."""
    tm = 0.5 * (etar + etal)
    return 0.5 * (etap - tm)


def _spm0(esap, esar, esal):
    """spm = 0.5*(esap - sm), sm = 0.5*(esar+esal)."""
    sm = 0.5 * (esar + esal)
    return 0.5 * (esap - sm)


def tn0(etar, etal):
    """tn = 0.5*(etar - etal)."""
    return 0.5 * (etar - etal)


def sn0(esar, esal):
    """sn = 0.5*(esar - esal)."""
    return 0.5 * (esar - esal)
