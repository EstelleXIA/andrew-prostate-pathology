"""Unified dual-scale MIL classifier for PathoSubtype / GeneExpression / GeneMolecularSubtype.

The number of output classes and the count of ``wsi_latent`` copies stacked before
the per-class heads are controlled by a single ``num_classes`` parameter.

  * DualScaleMILModelNuclei  (arch: final / no_prism)
  * DualScaleMILModel        (arch: no_nuclei / only_low / only_high)
  * build_downstream_model(arch_type, num_classes, input_dim) -> nn.Module
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import WSIEncoder


class DualScaleMILModelNuclei(nn.Module):
    """Dual-scale (256/512) MIL over nuclei-augmented features, optional PRISM fusion.

    Was ``SubtypeModelNuclei`` / ``GeneModelNuclei``. ``num_classes`` selects the
    number of independent per-class heads and the number of ``wsi_latent`` copies.
    """

    def __init__(self, num_classes=4, with_prism=True, captum=False):
        super().__init__()
        self.feat_dim = 1280
        self.captum = captum
        self.backbone = WSIEncoder(patch_dim=self.feat_dim // 4, class_num=num_classes, captum=captum)
        self.mapping_morph_256 = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim // 4 - 85),
                                               nn.ReLU(),
                                               nn.Dropout(0.1))
        self.mapping_morph_512 = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim // 4 - 85),
                                               nn.ReLU(),
                                               nn.Dropout(0.1))
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.feat_dim // 4, num_heads=4, batch_first=True)

        self.n_classes = num_classes

        self.with_prism = with_prism
        if with_prism:
            self.transform_prism = nn.Linear(self.feat_dim, self.feat_dim // 2)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(p=0.2)
            bag_classifiers = [nn.Linear(self.feat_dim // 4 + self.feat_dim // 2, 1) for _ in range(num_classes)]
        else:
            bag_classifiers = [nn.Linear(self.feat_dim // 4, 1) for _ in range(num_classes)]

        self.classifiers = nn.ModuleList(bag_classifiers)

    def forward(self, x_256, x_512, wsi_latent):
        feature_morph_256 = self.mapping_morph_256(x_256[:, :self.feat_dim])
        feature_morph_512 = self.mapping_morph_512(x_512[:, :self.feat_dim])

        if self.captum:
            features_256, A_256 = self.backbone(torch.concat([feature_morph_256, x_256[:, self.feat_dim:]], dim=1))
            features_512, A_512 = self.backbone(torch.concat([feature_morph_512, x_512[:, self.feat_dim:]], dim=1))
        else:
            features_256 = self.backbone(torch.concat([feature_morph_256, x_256[:, self.feat_dim:]], dim=1))
            features_512 = self.backbone(torch.concat([feature_morph_512, x_512[:, self.feat_dim:]], dim=1))

        features, _ = self.multihead_attn(query=features_256.unsqueeze(1),
                                          key=features_512.unsqueeze(1),
                                          value=features_512.unsqueeze(1))
        if self.with_prism:
            wsi_latent = self.transform_prism(wsi_latent)
            wsi_latent = self.relu(wsi_latent)
            wsi_latent = self.drop(wsi_latent)

            wsi_latent = torch.stack([wsi_latent] * self.n_classes, dim=0)
            features = torch.concat([features, wsi_latent], dim=2)

        logits = torch.empty(1, self.n_classes).to(features.dtype).to(features.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](features[c])

        if self.captum:
            return logits, A_256, A_512
        else:
            return logits


class DualScaleMILModel(nn.Module):
    """Dual-scale MIL without nuclei features (no PRISM).

    Was ``SubtypeModel`` / ``GeneModel``. ``with_low``/``with_high`` select which
    scales feed the shared ``backbone``:
      * with_low and with_high  -> 256 & 512 fused by multi-head attention (no_nuclei)
      * otherwise               -> a single scale (only_low uses x_512, only_high uses x_256)
    """

    def __init__(self, num_classes=4, input_dim=1280, with_low_input=True, with_high_input=True):
        super().__init__()
        self.feat_dim = input_dim
        self.with_low = with_low_input
        self.with_high = with_high_input

        self.backbone = WSIEncoder(patch_dim=self.feat_dim // 4, class_num=num_classes)
        self.mapping_morph_256 = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim // 4),
                                               nn.ReLU(),
                                               nn.Dropout(0.1))
        self.mapping_morph_512 = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim // 4),
                                               nn.ReLU(),
                                               nn.Dropout(0.1))
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.feat_dim // 4, num_heads=4, batch_first=True)

        self.n_classes = num_classes

        bag_classifiers = [nn.Linear(self.feat_dim // 4, 1) for _ in range(num_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)

    def forward(self, x_256, x_512, wsi_latent):
        if self.with_low and self.with_high:
            feature_morph_256 = self.mapping_morph_256(x_256)
            features_256 = self.backbone(feature_morph_256)

            feature_morph_512 = self.mapping_morph_512(x_512)
            features_512 = self.backbone(feature_morph_512)

            features, _ = self.multihead_attn(query=features_256.unsqueeze(1),
                                              key=features_512.unsqueeze(1),
                                              value=features_512.unsqueeze(1))
        else:
            # NOTE: single-scale path always projects x_256 with mapping_morph_256.
            # The dataset feeds the appropriate scale into the x_256 slot
            # (only_low -> 512-px features, only_high -> 256-px features).
            feature_morph_256 = self.mapping_morph_256(x_256)
            features = self.backbone(feature_morph_256)

        logits = torch.empty(1, self.n_classes).to(features.dtype).to(features.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](features[c])
        return logits


def build_downstream_model(arch_type: str, num_classes: int, input_dim: int = 1280,
                           captum: bool = False) -> nn.Module:
    """Factory reproducing the per-``arch_type`` model wiring shared by the three tasks."""
    if arch_type == "final":
        return DualScaleMILModelNuclei(num_classes=num_classes, with_prism=True, captum=captum)
    if arch_type == "no_prism":
        return DualScaleMILModelNuclei(num_classes=num_classes, with_prism=False, captum=captum)
    if arch_type == "no_nuclei":
        return DualScaleMILModel(num_classes=num_classes, input_dim=input_dim,
                                 with_low_input=True, with_high_input=True)
    if arch_type == "only_low":
        return DualScaleMILModel(num_classes=num_classes, input_dim=input_dim,
                                 with_low_input=True, with_high_input=False)
    if arch_type == "only_high":
        return DualScaleMILModel(num_classes=num_classes, input_dim=input_dim,
                                 with_low_input=False, with_high_input=True)
    raise ValueError(f"Unknown arch_type: {arch_type!r}")
