"""Central configuration: filesystem paths and per-encoder feature dimensions.

Usage
-----
    from common.config import get_paths, get_encoder_dims

    paths = get_paths()                     # reads configs/paths.yaml
    patch_dim, wsi_dim = get_encoder_dims("andrew")
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_DIR = os.path.join(_REPO_ROOT, "configs")


@dataclass(frozen=True)
class Paths:
    """Filesystem roots, loaded from ``configs/paths.yaml``."""

    mil_data_root: str
    repgen_data_root: str
    repgen_tumor_root: str
    labels_root: str
    patients_cache_root: str
    sicap_root: str
    ckpt_root: str

    def mil(self, *parts: str) -> str:
        return os.path.join(self.mil_data_root, *parts)

    def repgen(self, *parts: str) -> str:
        return os.path.join(self.repgen_data_root, *parts)

    def repgen_tumor(self, *parts: str) -> str:
        return os.path.join(self.repgen_tumor_root, *parts)

    def labels(self, *parts: str) -> str:
        return os.path.join(self.labels_root, *parts)

    def patients_cache(self, *parts: str) -> str:
        return os.path.join(self.patients_cache_root, *parts)

    def sicap(self, *parts: str) -> str:
        return os.path.join(self.sicap_root, *parts)

    def ckpt(self, *parts: str) -> str:
        return os.path.join(self.ckpt_root, *parts)


def _paths_yaml_path() -> str:
    # Allow an override for CI / alternate machines.
    return os.environ.get("ANDREW_PATHS_YAML", os.path.join(_CONFIG_DIR, "paths.yaml"))


@lru_cache(maxsize=1)
def get_paths() -> Paths:
    """Load ``configs/paths.yaml`` (or ``$ANDREW_PATHS_YAML``)."""
    path = _paths_yaml_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Path config not found at {path}. "
            "Copy configs/paths.example.yaml to configs/paths.yaml and edit it "
            "(or set $ANDREW_PATHS_YAML)."
        )
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Paths(**raw)


@lru_cache(maxsize=1)
def _encoder_table() -> dict:
    with open(os.path.join(_CONFIG_DIR, "encoders.yaml"), "r") as f:
        return yaml.safe_load(f)


def get_encoder_dims(model_type: str) -> Tuple[int, int]:
    """Return ``(patch_dim, wsi_dim)`` for a patch-encoder name.

    Replaces the ``if model_type == ...`` ladder in every train script.
    """
    table = _encoder_table()
    if model_type not in table:
        raise ValueError(
            f"Unknown model_type {model_type!r}; expected one of {sorted(table)}."
        )
    entry = table[model_type]
    return entry["patch_dim"], entry["wsi_dim"]


def check_arch_invariant(model_type: str, arch_type: str) -> None:
    """Enforce ``arch_type != 'only_high' => model_type == 'andrew'``."""
    if arch_type != "only_high" and model_type != "andrew":
        raise AssertionError(
            f"arch_type={arch_type!r} requires model_type='andrew' "
            f"(got {model_type!r}); only the andrew encoder has dual-scale/nuclei/PRISM features."
        )
