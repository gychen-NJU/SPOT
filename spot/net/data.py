"""spot.net.data — BandH dataset: Bifrost + Hinode mixed Stokes (s) and atmospheric parameters.

The raw file `BandH.pt` stores the physical fields (top-level keys):
  s  : (N, 112, 4)  Stokes I,Q,U,V profiles (continuum normalised)
  t  : (N, 64)      temperature        [K]
  p  : (N, 64)      electron pressure  [dyn/cm^2]
  b  : (N, 64)      magnetic field     [G]
  g  : (N, 64)      inclination        [deg]
  f  : (N, 64)      azimuth            [deg] (unwrapped, may exceed +-360)
  v  : (N, 64)      LOS velocity       [cm/s]
  m  : (N,)         micro turbulence   [cm/s]
  M  : (N,)         macro turbulence   [cm/s]
(and log10 / radian / km-s variants: lgt, lgp, lgb, rdg, rdf, kmv, kmm, kmM)

The inversion task learned here:  s -> t, p, b, g, f, v, m, M.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

DEFAULT_DATA_PATH = os.path.join(
    os.path.expanduser("~"), "Documents", "2026", "Hinode", "data", "BandH.pt"
)

DEPTH_PARAMS = ("t", "p", "b", "g", "f", "v")
SCALAR_PARAMS = ("m", "M")
ALL_PARAMS = DEPTH_PARAMS + SCALAR_PARAMS
N_WAVELENGTH = 112
N_COMPONENTS = 4
N_DEPTH = 64


def load_bandh(
    data_path: Optional[str] = None,
    drop_nonfinite: bool = True,
) -> Dict[str, torch.Tensor]:
    """Load BandH.pt and keep only rows with finite targets and profiles.

    Returns a dict with keys 'stokes' (N,112,4) and ALL_PARAMS (physical units).
    """
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    print(f"[load_bandh] loading {data_path}")
    raw = torch.load(data_path, map_location="cpu", weights_only=False)

    data = {k: raw[k].to(torch.float32) for k in ALL_PARAMS}
    data["stokes"] = raw["s"].to(torch.float32)
    N = data["stokes"].shape[0]
    print(f"[load_bandh] {N} samples")

    if drop_nonfinite:
        mask = torch.ones(N, dtype=torch.bool)
        for k, v in data.items():
            fin = torch.isfinite(v)
            while fin.dim() > 1:                    # all dims -> row-level mask
                fin = fin.all(dim=-1)
            mask &= fin
        n_bad = (~mask).sum().item()
        if n_bad:
            print(f"[load_bandh] dropping {n_bad} rows with non-finite values "
                  f"({100.0 * n_bad / N:.3f}%)")
            data = {k: v[mask] for k, v in data.items()}

    for k, v in data.items():
        print(f"  {k:7s} {tuple(v.shape)}  min={v.min():.4g}  max={v.max():.4g}")
    return data


def split_indices(
    n: int, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic 80/10/10 split (same convention as the v7 pipeline)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


class BandHDataset(Dataset):
    """Pre-normalised dataset: slices of fully normalised tensors."""

    def __init__(
        self,
        stokes_norm: torch.Tensor,
        params_norm: Dict[str, torch.Tensor],
        indices: Optional[Sequence[int]] = None,
    ):
        self.stokes = stokes_norm if indices is None else stokes_norm[indices]
        self.params = {
            k: (v if indices is None else v[indices]) for k, v in params_norm.items()
        }

    def __len__(self) -> int:
        return self.stokes.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        out = {"stokes": self.stokes[idx]}
        for k, v in self.params.items():
            out[k] = v[idx]
        return out


def build_split_dataloaders(
    data: Dict[str, torch.Tensor],
    normalizer,
    batch_size: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 2,
    pin_memory: bool = False,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.Tensor]:
    """Fit stats on the train split, normalise once, return (train, val, test_idx)."""
    N = data["stokes"].shape[0]
    train_idx, val_idx, test_idx = split_indices(N, train_ratio, val_ratio, seed)
    normalizer.fit(data, train_idx)
    norm = normalizer.normalize_all(data)

    train_ds = BandHDataset(norm["stokes"], norm, train_idx)
    val_ds = BandHDataset(norm["stokes"], norm, val_idx)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    print(f"[data] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
          f"steps/epoch={len(train_dl)}")
    return train_dl, val_dl, test_idx
