"""spot.net.normalization — Normalisation of Stokes spectra and atmospheric parameters.

Conventions (validated against the raw BandH.pt fields):
  t (K), p (dyn/cm^2), b (G)  -> log10 -> z-score
  g (inclination, deg)        -> g / 180  -> [0, 1]
  f (azimuth, deg, periodic)  -> (sin(f), cos(f))  2-D angular encoding
  v (LOS velocity, cm/s)      -> z-score (possible negative and large)
  m, M (turbulent velocities, cm/s) -> log10 -> z-score
  Stokes I,Q,U,V              -> per-channel z-score
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
import torch

LOG_FLOOR = 1e-8          # clamp for log10 (positive quantities)


class Normalizer:
    """Fits univariate statistics on a training subset and transforms arrays."""

    def __init__(
        self,
        depth_params: Sequence[str] = ("t", "p", "b", "g", "f", "v"),
        scalar_params: Sequence[str] = ("m", "M"),
        angular_params: Sequence[str] = ("g",),
        sincos_params: Sequence[str] = ("f",),
        log_params: Sequence[str] = ("t", "p", "b", "m", "M"),
        std_params: Sequence[str] = ("v",),
    ):
        self.depth_params = tuple(depth_params)
        self.scalar_params = tuple(scalar_params)
        self.angular_params = tuple(angular_params)
        self.sincos_params = tuple(sincos_params)
        self.log_params = tuple(log_params)
        self.std_params = tuple(std_params)
        self.stats: Dict[str, dict] = {}

    # ------------------------------------------------------------- fitting --
    def fit(self, data: Dict[str, torch.Tensor], indices: Optional[torch.Tensor] = None) -> "Normalizer":
        """Compute stats from data rows `indices` (default: all rows)."""
        s = data["stokes"]
        if indices is not None:
            s = s[indices]
        s64 = s.to(torch.float64)
        sm = s64.mean(dim=(0, 1))
        ss = s64.std(dim=(0, 1))
        self.stats["stokes"] = {"mean": sm, "std": ss + 1e-8}

        for name in list(self.depth_params) + list(self.scalar_params):
            v = data[name]
            if indices is not None:
                v = v[indices]
            v = v.to(torch.float64)
            if not torch.isfinite(v).all():
                raise ValueError(f"non-finite values in parameter '{name}'")
            st = {}
            if name in self.log_params:
                lv = torch.log10(torch.clamp(v, min=LOG_FLOOR))
                st["transform"] = "log10"
                st["mean"] = lv.mean().item()
                st["std"] = lv.std().item() + 1e-8
            elif name in self.sincos_params:
                st["transform"] = "sincos"
            elif name in self.angular_params:
                st["transform"] = "linear"           # g: deg -> unit
                st["scale"] = 180.0
            else:                                    # z-score params
                st["transform"] = "std"
                st["mean"] = v.mean().item()
                st["std"] = v.std().item() + 1e-8
            self.stats[name] = st
        return self

    # ---------------------------------------------------------- transforms --
    def normalize_stokes(self, s: torch.Tensor) -> torch.Tensor:
        m = self.stats["stokes"]["mean"].to(s.device, dtype=torch.float32)
        d = self.stats["stokes"]["std"].to(s.device, dtype=torch.float32)
        return (s.to(torch.float32) - m) / d

    def normalize_param(self, name: str, v: torch.Tensor) -> torch.Tensor:
        st = self.stats[name]
        v = v.to(torch.float32)
        if st["transform"] == "log10":
            z = (torch.log10(v.clamp(min=LOG_FLOOR)) - st["mean"]) / st["std"]
            return z
        if st["transform"] == "sincos":
            rad = v * (math.pi / 180.0)
            return torch.stack([torch.sin(rad), torch.cos(rad)], dim=-1)
        if st["transform"] == "linear":
            return v / st["scale"]
        return (v - st["mean"]) / st["std"]

    def denormalize_param(self, name: str, z: torch.Tensor) -> torch.Tensor:
        st = self.stats[name]
        z = z.to(torch.float32)
        if st["transform"] == "log10":
            return torch.pow(10.0, z * st["std"] + st["mean"])
        if st["transform"] == "sincos":
            rad = torch.atan2(z[..., 0], z[..., 1])
            return rad * (180.0 / math.pi)
        if st["transform"] == "linear":
            return z * st["scale"]
        return z * st["std"] + st["mean"]

    # ------------------------------------------------- virtualization API --
    def normalize_all(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Vectorised transform of the whole dataset (used to build tensors)."""
        out = {"stokes": self.normalize_stokes(data["stokes"])}
        for name in list(self.depth_params) + list(self.scalar_params):
            out[name] = self.normalize_param(name, data[name])
        return out

    def denormalize_all(self, pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: self.denormalize_param(k, v) for k, v in pred.items()}

    # ----------------------------------------------------------------- I/O --
    def save(self, path: str) -> None:
        torch.save(self.stats, path)

    def load(self, path: str) -> "Normalizer":
        self.stats = torch.load(path, map_location="cpu", weights_only=True)
        return self

    def describe(self) -> str:
        lines = []
        lines.append("stokes: per-channel z-score")
        for name, st in self.stats.items():
            if name == "stokes":
                continue
            if st["transform"] == "log10":
                lines.append(f"{name}: log10-zscore  mean={st['mean']:.4f} std={st['std']:.4f}")
            elif st["transform"] == "sincos":
                lines.append(f"{name}: sin/cos(deg->rad) angular encoding")
            elif st["transform"] == "linear":
                lines.append(f"{name}: /{st['scale']} (angular, deg)")
            else:
                lines.append(f"{name}: z-score  mean={st['mean']:.4g} std={st['std']:.4g}")
        return "\n".join(lines)
