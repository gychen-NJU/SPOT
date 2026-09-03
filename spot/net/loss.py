"""spot.net.loss — training loss and regression/angular metrics.

The training loss is computed in normalised space:
    L = sum_i w_i * (MSE_i + 0.05 * RRMSE_i)
where RRMSE_i = sqrt(MSE_i / var(target_i)) is a scale-invariant relative term.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .data import DEPTH_PARAMS, SCALAR_PARAMS

DEFAULT_WEIGHTS: Dict[str, float] = {
    "t": 1.5, "p": 1.5, "b": 1.5, "g": 1.0, "f": 1.0, "v": 1.0, "m": 2.0, "M": 2.0,
}


def compute_loss(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Weighted (MSE + 0.05*RRMSE) summed over atmospheric parameters; returns (total_loss, per-param MSE)."""
    weights = weights or DEFAULT_WEIGHTS
    total = torch.zeros((), device=next(iter(pred.values())).device)
    per_param: Dict[str, float] = {}
    for name in list(DEPTH_PARAMS) + list(SCALAR_PARAMS):
        p = pred[name].float()
        t = target[name].float()
        mse = F.mse_loss(p, t)
        rrmse = torch.sqrt(mse / t.var().clamp(min=1e-8))
        total = total + weights.get(name, 1.0) * (mse + 0.05 * rrmse)
        per_param[name] = mse.item()
    return total, per_param


# ------------------------------------------------------------------ metrics --
def _mse(x, y):
    return float(np.mean((x - y) ** 2))


def regression_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """Compute MSE / RRMSE / MAE / R2 / Pearson correlation on flattened arrays.

    Non-finite pairs are dropped and counted in 'n_finite'; when no finite
    pair remains, every metric is set to NaN.
    """
    pred = pred.ravel().astype(np.float64)
    true = true.ravel().astype(np.float64)
    ok = np.isfinite(pred) & np.isfinite(true)
    n = int(ok.sum())
    if n == 0:
        return {"MSE": float("nan"), "RRMSE": float("nan"), "MAE": float("nan"),
                "R2": float("nan"), "Corr": float("nan"), "n_finite": 0}
    pred, true = pred[ok], true[ok]
    mse = _mse(pred, true)
    var = true.var() + 1e-10
    mae = float(np.mean(np.abs(pred - true)))
    r2 = 1.0 - mse / var
    corr = float(np.corrcoef(pred, true)[0, 1]) if var > 0 else float("nan")
    return {"MSE": mse, "RRMSE": float(np.sqrt(mse / var)),
            "MAE": mae, "R2": r2, "Corr": corr, "n_finite": n}


def angular_metrics_deg(pred_deg: np.ndarray, true_deg: np.ndarray) -> Dict[str, float]:
    """Regression metrics for a periodic angle in degrees, using the circular distance."""
    d = np.mod(pred_deg - true_deg + 180.0, 360.0) - 180.0
    mse = float(np.mean(d ** 2))
    var = true_deg.var() + 1e-10
    return {
        "MSE": mse,
        "RRMSE": float(np.sqrt(mse / var)),
        "MAE": float(np.mean(np.abs(d))),
        "R2": 1.0 - mse / var,
        "Corr": float(np.corrcoef(pred_deg.ravel(), true_deg.ravel())[0, 1]),
    }
