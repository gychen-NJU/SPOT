"""spot.net.train — training loop: warmup + cosine-annealed LR, AMP, checkpointing, live logs.

Step-based learning-rate schedule:
  - linear warmup from 1e-3 * peak_lr to peak_lr over `warmup_epochs`
  - cosine annealing from peak_lr down to `min_lr_ratio * peak_lr`
    over the remaining steps of `n_epochs`.

The schedule is stepped once per optimizer step (``create_lr_scheduler``).
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import OUTPUT_ROOT, TrainingConfig
from .data import DEPTH_PARAMS, SCALAR_PARAMS
from .loss import compute_loss
from .plots import plot_per_param_curves, plot_training_curves


# ---------------------------------------------------------------- LR scheduler --
def create_lr_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
):
    """LambdaLR: linear warmup, then cosine annealing down to `min_lr_ratio * peak` LR."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return 0.001 + (1.0 - 0.001) * (step / max(warmup_steps, 1))
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ------------------------------------------------------------------- epoch helpers --
def train_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    config: TrainingConfig,
    scheduler=None,
) -> Tuple[float, float]:
    model.train()
    total_loss, n_batches = 0.0, 0
    t0 = time.time()
    for batch in loader:
        stokes = batch["stokes"].to(device, non_blocking=True)
        target = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "stokes"}

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                pred = model(stokes)
                loss, _ = compute_loss(pred, target, config.loss_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            stepped = getattr(optimizer, "_opt_called", False)
        else:
            pred = model(stokes)
            loss, _ = compute_loss(pred, target, config.loss_weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            stepped = True

        # advance the LR schedule only when the optimizer actually stepped (AMP may skip)
        if scheduler is not None and stepped:
            scheduler.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1), time.time() - t0


@torch.no_grad()
def validate(model: nn.Module, loader, device: torch.device, config: TrainingConfig):
    model.eval()
    total_loss, n_batches = 0.0, 0
    per_param: Dict[str, List[float]] = {}
    for batch in loader:
        stokes = batch["stokes"].to(device, non_blocking=True)
        target = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "stokes"}
        pred = model(stokes)
        loss, mses = compute_loss(pred, target, config.loss_weights)
        total_loss += loss.item()
        for k, v in mses.items():
            per_param.setdefault(k, []).append(v)
        n_batches += 1
    avg = {k: float(np.mean(v)) for k, v in per_param.items()}
    return total_loss / max(n_batches, 1), avg


# --------------------------------------------------------------------- training loop --
def run_training(
    config: TrainingConfig,
    data: Dict[str, torch.Tensor],
    normalizer,
    device: torch.device,
    run_dir: str,
    train_loader,
    val_loader,
    verbosity: int = 1,
) -> Tuple[Dict[str, list], float, int]:
    """Run the full training loop. Returns (history, best_val_loss, best_epoch)."""
    from .model import StokesPHNO  # local import keeps module-import cost out of the hot path

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")
    log_file = open(log_path, "w", buffering=1)

    def log(msg: str) -> None:
        print(msg, flush=True)
        print(msg, file=log_file, flush=True)

    normalizer.save(os.path.join(run_dir, "norm_stats.pt"))
    config.save(os.path.join(run_dir, "config.json"))

    torch.manual_seed(config.seed)
    model = StokesPHNO(
        d_model=config.d_model,
        n_fno_layers=config.n_fno_layers,
        n_transformer_layers=config.n_transformer_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        se_reduction=config.se_reduction,
        fno_modes=config.fno_modes,
        out_dims=config.out_dims,
    ).to(device)
    log(f"Model parameters: {model.count_params():,}")
    log(f"Device: {device}  ({torch.cuda.get_device_name(device.index) if device.type == 'cuda' else 'CPU'})")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    steps_per_epoch = len(train_loader)
    warmup_steps = steps_per_epoch * config.warmup_epochs
    total_steps = steps_per_epoch * config.n_epochs
    scheduler = create_lr_scheduler(optimizer, warmup_steps, total_steps, config.min_lr_ratio)
    log(f"steps/epoch={steps_per_epoch}, warmup={warmup_steps} steps, total={total_steps} steps")

    try:  # torch >= 2.3 moved GradScaler under torch.amp; fall back for older versions
        scaler = torch.amp.GradScaler("cuda") if config.amp and device.type == "cuda" else None
    except (AttributeError, TypeError):
        scaler = (torch.cuda.amp.GradScaler(enabled=True) if config.amp and device.type == "cuda" else None)
    log(f"AMP: {'fp16' if scaler else 'off'}")

    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "lr": []}
    best_val = float("inf")
    best_epoch = 0
    early_stop_counter = 0
    total_batches = 0
    t_start = time.time()

    for epoch in range(config.n_epochs):
        tr_loss, elapsed = train_epoch(
            model, train_loader, optimizer, device, scaler, config, scheduler
        )
        val_loss, per_param = validate(model, val_loader, device, config)
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr)
        for k, v in per_param.items():
            history.setdefault(f"val_mse_{k}", []).append(v)

        if val_loss < best_val - config.early_stop_delta:
            best_val, best_epoch = val_loss, epoch + 1
            early_stop_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": tr_loss,
                    "config": config.to_dict(),
                },
                os.path.join(ckpt_dir, "best_model.pt"),
            )
        else:
            early_stop_counter += 1

        if (epoch + 1) % config.save_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": tr_loss,
                    "config": config.to_dict(),
                },
                os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch + 1:04d}.pt"),
            )

        if (epoch + 1) % config.plot_interval == 0:
            plot_training_curves(history, os.path.join(fig_dir, "training_curves.png"), best_epoch)
            plot_per_param_curves(history, os.path.join(fig_dir, "per_param_val_mse.png"))
            with open(os.path.join(run_dir, "history.json"), "w") as f:
                json.dump(history, f)

        log(
            f"Epoch {epoch + 1:3d}/{config.n_epochs} | {elapsed:5.0f}s "
            f"| lr={lr:.3e} | train={tr_loss:.4e} | val={val_loss:.4e}"
        )
        if val_loss < best_val - config.early_stop_delta:
            log(f"  *** Best ({epoch + 1}) val_loss={val_loss:.4e}")
        if early_stop_counter >= config.early_stop_patience:
            log(f"Early stopping at epoch {epoch + 1} (no improvement for "
                f"{config.early_stop_patience} epochs)")
            break

    # final artifacts: last checkpoint, consolidated history, and run metadata
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
            "train_loss": tr_loss,
            "config": config.to_dict(),
        },
        os.path.join(ckpt_dir, "last_model.pt"),
    )
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f)
    meta = {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "n_epochs_run": epoch + 1,
        "wall_seconds": time.time() - t_start,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_batches + steps_per_epoch * (epoch + 1),
    }
    with open(os.path.join(run_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"Done in {(time.time() - t_start) / 3600:.2f} h. Best val loss = {best_val:.4e} @ epoch {best_epoch}")
    log_file.close()
    return history, best_val, best_epoch


def make_run_dir(run_name: Optional[str] = None, root: str = OUTPUT_ROOT) -> str:
    """Create a fresh run directory under `root`: <root>/<run_name>_<YYYYmmdd_HHMMSS>."""
    import datetime
    name = run_name or "run"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(root, f"{name}_{ts}")
    os.makedirs(d, exist_ok=True)
    return d
