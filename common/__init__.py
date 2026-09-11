"""Shared package for the prostate-pathology MIL codebase.

Re-exports the most commonly used building blocks so task modules can do
``from common import build_downstream_model, get_paths, define_loss`` etc.
"""
from __future__ import annotations

from .config import (
    Paths,
    check_arch_invariant,
    get_encoder_dims,
    get_paths,
)
from .engine import (
    DEFAULT_DTYPE,
    GRAD_ACCUM_STEPS,
    CheckpointManager,
    resolve_device,
    set_seed,
)
from .layers import (
    Attention_net_gated,
    GSEncoder,
    HierarchicalEncoder,
    HierarchicalEncoderNuclei,
    RegionEncoder,
    WSIEncoder,
)
from .losses import (
    AsymmetricLoss,
    CosineLoss,
    CoxSurvLoss,
    CrossEntropySurvLoss,
    KLLoss,
    MultiClassFocalLoss,
    NLLSurvLoss,
    OrthogonalLoss,
    define_loss,
)
from .metrics import (
    c_index,
    calculate_metrics,
    quadratic_weighted_kappa,
)
from .models import (
    DualScaleMILModel,
    DualScaleMILModelNuclei,
    build_downstream_model,
)

__all__ = [
    # config
    "Paths", "get_paths", "get_encoder_dims", "check_arch_invariant",
    # engine
    "set_seed", "resolve_device", "CheckpointManager",
    "GRAD_ACCUM_STEPS", "DEFAULT_DTYPE",
    # layers
    "Attention_net_gated", "WSIEncoder", "GSEncoder", "RegionEncoder",
    "HierarchicalEncoder", "HierarchicalEncoderNuclei",
    # models
    "DualScaleMILModel", "DualScaleMILModelNuclei", "build_downstream_model",
    # losses
    "define_loss", "AsymmetricLoss", "MultiClassFocalLoss",
    "NLLSurvLoss", "CrossEntropySurvLoss", "CoxSurvLoss",
    "KLLoss", "CosineLoss", "OrthogonalLoss",
    # metrics
    "calculate_metrics", "quadratic_weighted_kappa", "c_index",
]
