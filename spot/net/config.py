"""spot.net.config — training configuration (a dataclass, JSON-serialisable via `to_dict`/`from_dict`)."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .data import DEFAULT_DATA_PATH

OUTPUT_ROOT = "outputs/runs"


@dataclass
class TrainingConfig:
    # dataset / split configuration
    data_path: str = DEFAULT_DATA_PATH
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 42
    num_workers: int = 2
    pin_memory: bool = False

    # model architecture
    d_model: int = 256
    n_fno_layers: int = 2
    n_transformer_layers: int = 4
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    se_reduction: int = 16
    fno_modes: int = 16
    out_dims: Dict[str, int] = field(default_factory=lambda: {"f": 2})

    # optimisation: LR schedule (warmup + cosine), weight decay, AMP, loss weighting
    batch_size: int = 128
    learning_rate: float = 5e-4
    min_lr_ratio: float = 0.01      # cosine-annealing floor = 0.01 * peak LR
    warmup_epochs: int = 15         # number of warmup epochs for the linear LR ramp
    n_epochs: int = 300
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    amp: bool = True
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {"t": 1.5, "p": 1.5, "b": 1.5,
                                 "g": 1.0, "f": 1.0, "v": 1.0,
                                 "m": 2.0, "M": 2.0}
    )

    # run bookkeeping: checkpointing, early stopping, and plotting cadence
    save_interval: int = 10
    early_stop_patience: int = 50
    early_stop_delta: float = 1e-7
    plot_interval: int = 20
    device: str = "auto"            # "auto" selects the first visible CUDA device, else CPU
    run_name: str = "bandh_v1"

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["betas"] = list(self.betas)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        d = dict(d)
        d["betas"] = tuple(d.get("betas", (0.9, 0.95)))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def resolve_device(self) -> torch.device:
        import torch
        if self.device == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
