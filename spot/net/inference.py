"""spot.net.inference — Convenient inference interface for a trained StokesPHNO model.

Quick start
-----------
    from spot.net import StokesInference

    infer = StokesInference(preset="hinode_sp")    # bundled Hinode SP model
    result = infer.predict(stokes_tensor)          # dict of physical-unit tensors
    phys = infer.predict_numpy(stokes_np)          # same, as numpy arrays

    # or point to your own run:
    infer = StokesInference(model_path=..., norm_path=..., config_path=...)

Presets
-------
  "hinode_sp": the current SPOT network (v3, trained from scratch on 2.36M BIFROST
               1-D atmospheres + 519k SIR-inverted Hinode SP profiles; 50 epochs).
               The input Stokes intensities are expected in PHYSICAL units and are
               divided by I_c,ref = 8.257986e14 (k * I_C(HSRA, 6301.508)) inside
               predict() automatically; pass input_scale=None for already
               continuum-normalised input.  See
               spot/net/models/hinode_sp/MODEL_CARD.md.

The azimuth 'f' is returned wrapped to (-180, 180] degrees; velocities (v, m, M)
are in cm/s as stored in the dataset; t in K, p in dyn/cm^2, b in G, g in deg.
The depth axis is index 0 = shallowest (log tau5000 = -4) ... 63 = deepest (+2).
"""
from __future__ import annotations

import glob
import os
import time
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch

from .config import OUTPUT_ROOT, TrainingConfig
from .data import ALL_PARAMS, DEPTH_PARAMS, N_COMPONENTS, N_WAVELENGTH
from .model import StokesPHNO
from .normalization import Normalizer

# --------------------------------------------------------------- presets ----
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

PRESETS: Dict[str, dict] = {
    "hinode_sp": {
        "model_version": "v3-50ep (2026-09-12)",
        "description": (
            "Current SPOT network: trained from scratch on 2.36M BIFROST 1-D "
            "atmospheres + 519k SIR-inverted Hinode SP profiles (50 epochs; "
            "azimuth folded to [0,180)). Inputs are expected in physical "
            "intensity units; they are divided by I_c,ref = 8.257986e14 "
            "automatically. See models/hinode_sp/MODEL_CARD.md."
        ),
        "model": os.path.join(MODELS_DIR, "hinode_sp", "best_model.pt"),
        "norm": os.path.join(MODELS_DIR, "hinode_sp", "norm_stats.pt"),
        "config": os.path.join(MODELS_DIR, "hinode_sp", "config.json"),
        "input_scale": 8.25798607858631e14,   # I_c,ref = k * I_C(HSRA, 6301.508)
    },
}
# NOTE: the original fine-tuned checkpoint (bundled up to v1.1.1) is kept under
# models/deprecated/hinode_sp_v1/ for provenance only -- no preset points there
# and it is not shipped in the wheel.


def list_presets() -> Dict[str, str]:
    """Return the available presets with one-line descriptions."""
    return {k: v["description"] for k, v in PRESETS.items()}


def find_latest_run(root: Optional[str] = None) -> str:
    """Path of the most recently modified run directory containing best_model.pt."""
    # spot.net sits one level deeper than the package root (SPOT/spot/), so
    # the default training-output root is the project root's outputs/runs.
    root = root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        OUTPUT_ROOT)
    runs = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
    runs = [d for d in runs if os.path.exists(os.path.join(d, "checkpoints", "best_model.pt"))]
    if not runs:
        raise FileNotFoundError(
            f"no trained run found under {root}; pass model_path explicitly"
        )
    return max(runs, key=os.path.getmtime)


class StokesInference:
    """Loads model + normaliser once and predicts physical parameters on demand.

    Args:
      preset: named configuration from PRESETS (e.g. "hinode_sp"); explicit
        model_path/norm_path/config_path still override the preset values.
      model_path / norm_path / config_path: direct paths (defaults: newest run).
      device: "cuda:N" / "cpu"; None = auto.
      batch_size: default batch size for predict().
      input_scale: divide the input Stokes by this constant before inference
        (set automatically from the preset; None disables).
    """

    def __init__(
        self,
        preset: Optional[str] = None,
        model_path: Optional[str] = None,
        norm_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        input_scale: Optional[float] = None,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.input_scale = input_scale
        if preset is not None:
            if preset not in PRESETS:
                raise ValueError(
                    f"unknown preset '{preset}'. Available: {sorted(PRESETS)}"
                )
            p = PRESETS[preset]
            model_path = model_path or p["model"]
            norm_path = norm_path or p["norm"]
            config_path = config_path or p["config"]
            if self.input_scale is None:
                self.input_scale = p.get("input_scale")
            if self.verbose:
                print(f"[StokesInference] preset='{preset}' "
                      f"(input scale = {self.input_scale:.4e})")
        if model_path is None:
            run = find_latest_run()
            model_path = os.path.join(run, "checkpoints", "best_model.pt")
            norm_path = norm_path or os.path.join(run, "norm_stats.pt")
            config_path = config_path or os.path.join(run, "config.json")
            if self.verbose:
                print(f"[StokesInference] auto-discovered run: {run}")
        # when only model_path is given, derive the run directory from it
        run_dir = os.path.dirname(os.path.dirname(model_path))
        norm_path = norm_path or os.path.join(run_dir, "norm_stats.pt")
        config_path = config_path or os.path.join(run_dir, "config.json")
        cfg = TrainingConfig.load(config_path)

        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.normalizer = Normalizer().load(norm_path)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = StokesPHNO(
            d_model=cfg.d_model, n_fno_layers=cfg.n_fno_layers,
            n_transformer_layers=cfg.n_transformer_layers, n_heads=cfg.n_heads,
            d_ff=cfg.d_ff, dropout=cfg.dropout, se_reduction=cfg.se_reduction,
            fno_modes=cfg.fno_modes, out_dims=cfg.out_dims,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.default_batch_size = batch_size
        if self.verbose:
            print(f"[StokesInference] device={self.device}, model params={self.model.count_params():,}, "
                  f"epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss')}")

    # ------------------------------------------------------------- predict --
    def _as_tensor(self, stokes: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(stokes, np.ndarray):
            stokes = torch.from_numpy(stokes.copy())
        if not isinstance(stokes, torch.Tensor):
            stokes = torch.tensor(stokes, dtype=torch.float32)
        stokes = stokes.to(torch.float32)
        if stokes.dim() == 2:
            stokes = stokes.unsqueeze(0)     # single profile (n_wavelength, 4)
        if stokes.dim() != 3 or stokes.shape[1] != N_WAVELENGTH or stokes.shape[2] != N_COMPONENTS:
            raise ValueError(
                f"expected (B, {N_WAVELENGTH}, {N_COMPONENTS}), got {tuple(stokes.shape)}"
            )
        return stokes

    @torch.no_grad()
    def predict(
        self,
        stokes: Union[np.ndarray, torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Invert Stokes profiles -> physical parameters in original units."""
        stokes = self._as_tensor(stokes)
        if self.input_scale is not None:
            stokes = stokes / self.input_scale
        B = stokes.shape[0]
        bs = batch_size or self.default_batch_size or B
        vprint = print if self.verbose else (lambda *a, **k: None)
        vprint(f"[predict] {B} profiles, batch_size={bs}")

        stokes_norm = self.normalizer.normalize_stokes(stokes)
        results = {k: [] for k in ALL_PARAMS}
        t0 = time.time()
        for i in range(0, B, bs):
            batch = stokes_norm[i:i + bs].to(self.device)
            pred = self.model(batch)
            for k in ALL_PARAMS:
                results[k].append(pred[k].cpu())
        results = {k: torch.cat(v, dim=0) for k, v in results.items()}
        for k in ALL_PARAMS:
            results[k] = self.normalizer.denormalize_param(k, results[k])
        vprint(f"[predict] done in {time.time() - t0:.2f}s "
               f"({1000 * (time.time() - t0) / B:.2f} ms/profile)")
        return results

    def predict_numpy(
        self,
        stokes: Union[np.ndarray, torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Same as predict() but returns numpy arrays."""
        return {k: v.numpy() for k, v in self.predict(stokes, batch_size).items()}


def main() -> None:  # small CLI: python -m spot.net.inference --stokes x.npy --out y.npz
    import argparse
    import json

    parser = argparse.ArgumentParser(description="spot.net.inference: invert Stokes profiles with a trained StokesPHNO model")
    parser.add_argument("--stokes", required=True, help="input Stokes array (.npy / .npz / .pt)")
    parser.add_argument("--out", required=True, help="output .npz path")
    parser.add_argument("--preset", default=None,
                        help=f"named model preset, one of {sorted(PRESETS)}")
    parser.add_argument("--model", default=None, help="best_model.pt path (default: latest run)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.stokes.endswith(".npy"):
        s = np.load(args.stokes)
    elif args.stokes.endswith(".npz"):
        s = np.load(args.stokes)["s"]
    else:
        s = torch.load(args.stokes, map_location="cpu", weights_only=False)
        s = s["s"] if isinstance(s, dict) else s
        s = s.numpy()
    infer = StokesInference(preset=args.preset, model_path=args.model, device=args.device,
                            batch_size=args.batch_size)
    out = infer.predict_numpy(s, batch_size=args.batch_size)
    np.savez(args.out, **out)
    print(f"Saved {args.out} with keys {sorted(out)}")
    for k, v in out.items():
        print(f"  {k:3s} {v.shape}  min={v.min():.4g} max={v.max():.4g}")


if __name__ == "__main__":
    main()
