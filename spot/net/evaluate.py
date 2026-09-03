"""spot.net.evaluate — held-out test-set evaluation: normalised and physical metrics, plus diagnostic figures."""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .config import TrainingConfig
from .data import (
    ALL_PARAMS,
    DEPTH_PARAMS,
    SCALAR_PARAMS,
    BandHDataset,
    load_bandh,
    split_indices,
)
from .loss import angular_metrics_deg, regression_metrics
from .normalization import Normalizer
from .plots import plot_depth_profiles, plot_scatter


@torch.no_grad()
def predict_testset(
    model, normalizer, data, test_idx, device, batch_size: int = 512,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Predict on the test split, returning predictions and targets in both physical and normalised units as numpy arrays (pred_phys, true_phys, pred_norm, true_norm)."""
    norm = normalizer.normalize_all(data)
    ds = BandHDataset(norm["stokes"], norm, test_idx)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    model.eval()
    pred_norm = {k: [] for k in ALL_PARAMS}
    true_norm = {k: [] for k in ALL_PARAMS}
    for batch in loader:
        stokes = batch["stokes"].to(device)
        pred = model(stokes)
        for k in ALL_PARAMS:
            pred_norm[k].append(pred[k].cpu())
            true_norm[k].append(batch[k])
    pred_norm = {k: torch.cat(v).numpy() for k, v in pred_norm.items()}
    true_norm = {k: torch.cat(v).numpy() for k, v in true_norm.items()}
    pred_phys = normalizer.denormalize_all(
        {k: torch.from_numpy(v) for k, v in pred_norm.items()}
    )
    pred_phys = {k: v.numpy() for k, v in pred_phys.items()}
    true_phys = normalizer.denormalize_all(
        {k: torch.from_numpy(v) for k, v in true_norm.items()}
    )
    true_phys = {k: v.numpy() for k, v in true_phys.items()}
    return pred_phys, true_phys, pred_norm, true_norm


def evaluation_metrics(pred_phys, true_phys, pred_norm, true_norm) -> Dict:
    results: Dict = {}
    for k in ALL_PARAMS:
        results[k] = {
            "normalized": regression_metrics(pred_norm[k], true_norm[k]),
            "physical": regression_metrics(pred_phys[k], true_phys[k]),
        }
    if "f" in ALL_PARAMS:
        results["f"]["physical_angular"] = angular_metrics_deg(
            pred_phys["f"], true_phys["f"]
        )
    return results


def run_evaluation(
    run_dir: str,
    device: Optional[torch.device] = None,
    batch_size: int = 512,
    max_points: int = 200_000,
) -> Dict:
    """Load a run's best model and normalizer, then evaluate on the held-out test split."""
    cfg = TrainingConfig.load(os.path.join(run_dir, "config.json"))
    device = device or cfg.resolve_device()
    normalizer = Normalizer().load(os.path.join(run_dir, "norm_stats.pt"))

    print(f"[eval] loading data from {cfg.data_path}")
    data = load_bandh(cfg.data_path)
    N = data["stokes"].shape[0]
    _, _, test_idx = split_indices(N, cfg.train_ratio, cfg.val_ratio, cfg.seed)

    from .model import StokesPHNO
    ckpt = torch.load(
        os.path.join(run_dir, "checkpoints", "best_model.pt"),
        map_location="cpu", weights_only=False,
    )
    model = StokesPHNO(
        d_model=cfg.d_model, n_fno_layers=cfg.n_fno_layers,
        n_transformer_layers=cfg.n_transformer_layers, n_heads=cfg.n_heads,
        d_ff=cfg.d_ff, dropout=cfg.dropout, se_reduction=cfg.se_reduction,
        fno_modes=cfg.fno_modes, out_dims=cfg.out_dims,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"[eval] best model: epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss'):.4e}, "
          f"params={model.count_params():,}")

    t0 = time.time()
    pred_phys, true_phys, pred_norm, true_norm = predict_testset(
        model, normalizer, data, test_idx, device, batch_size
    )
    print(f"[eval] predicted {len(test_idx)} test samples in {time.time() - t0:.0f}s")

    results = evaluation_metrics(pred_phys, true_phys, pred_norm, true_norm)
    _print_metrics_table(results)

    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    log_params = ["t", "p", "b", "m", "M"]
    for k in ALL_PARAMS:
        if k == "f":
            continue
        plot_scatter(
            pred_phys[k], true_phys[k], f"{k} (physical)",
            os.path.join(fig_dir, f"scatter_phys_{k}.png"),
            corr=results[k]["physical"]["Corr"],
            log_scale=(k in log_params),
            max_points=max_points,
        )
        plot_scatter(
            pred_norm[k], true_norm[k], f"{k} (normalized)",
            os.path.join(fig_dir, f"scatter_norm_{k}.png"),
            corr=results[k]["normalized"]["Corr"],
            max_points=max_points,
        )
    # f is the periodic azimuth: scatter it separately and report angular metrics in degrees
    plot_scatter(
        pred_phys["f"], true_phys["f"], "f (physical, deg)",
        os.path.join(fig_dir, "scatter_phys_f.png"),
        corr=results["f"]["physical_angular"]["Corr"],
        max_points=max_points,
    )
    plot_scatter(
        pred_norm["f"], true_norm["f"], "f (normalized sin/cos)",
        os.path.join(fig_dir, "scatter_norm_f.png"),
        corr=results["f"]["normalized"]["Corr"],
        max_points=max_points,
    )
    plot_depth_profiles(
        {k: pred_phys[k] for k in DEPTH_PARAMS if k != "f"},
        {k: true_phys[k] for k in DEPTH_PARAMS if k != "f"},
        os.path.join(fig_dir, "test_profiles.png"),
    )
    print(f"[eval] figures -> {fig_dir}")

    with open(os.path.join(run_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    torch.save(
        {"pred": pred_phys, "true": true_phys}, os.path.join(run_dir, "test_predictions.pt")
    )
    print(f"[eval] saved test_results.json and test_predictions.pt")
    return results


def _print_metrics_table(results: Dict) -> None:
    print(f"\n{'Param':>4s} | {'MSE':>10s} {'RRMSE':>8s} {'MAE':>10s} {'R2':>8s} "
          f"{'Corr':>8s} || {'phys MAE':>10s} {'phys R2':>8s} {'phys Corr':>8s}")
    print("-" * 110)
    for k in ALL_PARAMS:
        n = results[k]["normalized"]
        p = results[k]["physical"]
        extra = ""
        if "physical_angular" in results[k]:
            a = results[k]["physical_angular"]
            extra = f" (angular MAE={a['MAE']:.3f} deg)"
        print(f"{k:>4s} | {n['MSE']:10.3e} {n['RRMSE']:8.4f} {n['MAE']:10.3e} "
              f"{n['R2']:8.4f} {n['Corr']:8.4f} || {p['MAE']:10.3e} {p['R2']:8.4f} "
              f"{p['Corr']:8.4f}{extra}")
