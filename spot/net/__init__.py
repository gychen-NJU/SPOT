"""spot.net — the neural-operator subpackage of spot.

FNO + Transformer(SE) + DeepONet-style cross-attention decoder, mapping
Stokes I, Q, U, V profiles (B, 112, 4) to atmospheric parameters:
  depth profiles: t, p, b, g, f, v  (B, 64)  — f may be (B, 64, 2) sin/cos
  scalars: m, M                              (B,)
"""
from .config import TrainingConfig
from .data import (
    ALL_PARAMS,
    DEPTH_PARAMS,
    SCALAR_PARAMS,
    BandHDataset,
    build_split_dataloaders,
    load_bandh,
    split_indices,
)
from .evaluate import run_evaluation
from .inference import PRESETS, StokesInference, find_latest_run, list_presets
from .loss import angular_metrics_deg, compute_loss, regression_metrics
from .model import StokesPHNO
from .normalization import Normalizer
from .train import create_lr_scheduler, run_training

__version__ = "1.0.0"

__all__ = [
    "StokesPHNO",
    "StokesInference",
    "PRESETS",
    "list_presets",
    "Normalizer",
    "TrainingConfig",
    "load_bandh",
    "split_indices",
    "build_split_dataloaders",
    "BandHDataset",
    "compute_loss",
    "regression_metrics",
    "angular_metrics_deg",
    "create_lr_scheduler",
    "run_training",
    "run_evaluation",
    "find_latest_run",
    "ALL_PARAMS",
    "DEPTH_PARAMS",
    "SCALAR_PARAMS",
    "__version__",
]
