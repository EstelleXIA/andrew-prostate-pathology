"""Shared training scaffolding.

Provides the device/dtype setup, ``gc=16`` gradient accumulation, the
"save on best val metric / min val loss" checkpoint bookkeeping
(``CheckpointManager``), and seeding (``set_seed``) shared across tasks.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


# All tasks train with batch_size=1 and bf16 autocast on a single GPU.
GRAD_ACCUM_STEPS = 16
DEFAULT_DTYPE = torch.bfloat16


def set_seed(seed: int) -> None:
    """Seed python / numpy / torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class CheckpointManager:
    """"best-val-metric" + "min-val-loss" checkpoint save policy.

    Saves a checkpoint whenever the monitored validation metric improves OR the
    validation loss drops, and always re-saves the latest under the same name.
    ``higher_is_better`` selects the metric direction (AUC/QWK -> True; a
    pure-loss monitor -> False).
    """

    save_dir: str
    higher_is_better: bool = True
    best_metric: float = field(init=False)
    min_loss: float = field(default=float("inf"), init=False)

    def __post_init__(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_metric = -float("inf") if self.higher_is_better else float("inf")

    def _improved(self, metric: float) -> bool:
        return metric > self.best_metric if self.higher_is_better else metric < self.best_metric

    def maybe_save(self, model, optimizer, ckpt_name: str,
                   metric: Optional[float] = None, loss: Optional[float] = None) -> bool:
        """Save if the metric improved or the loss dropped. Returns True if saved."""
        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict()}
        saved = False
        if metric is not None and self._improved(metric):
            self.best_metric = metric
            torch.save(checkpoint, os.path.join(self.save_dir, ckpt_name))
            saved = True
        if loss is not None and loss < self.min_loss:
            self.min_loss = loss
            torch.save(checkpoint, os.path.join(self.save_dir, ckpt_name))
            saved = True
        return saved
