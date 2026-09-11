"""Canonical shared neural-network building blocks.

Blocks:
  * Attention_net_gated  — gated-attention MIL pooling scorer
  * WSIEncoder           — Patho/Gene/GMS dual-scale backbone (mapping_x + wsi_mil)
  * GSEncoder            — GS_ISUP backbone (n_classes=3 gated attention)
  * RegionEncoder        — BCR stage-1 region-level MIL
  * HierarchicalEncoder / HierarchicalEncoderNuclei — BCR stage-2 region->WSI MIL
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention_net_gated(nn.Module):
    """Gated-attention scorer (Ilse et al.); returns per-instance logits ``A`` and ``x``."""

    def __init__(self, input_dim=256, out_dim=256, n_classes=1, dropout=0.):
        super(Attention_net_gated, self).__init__()
        self.attention_a = [nn.LayerNorm(input_dim), nn.Linear(input_dim, out_dim), nn.Tanh()]
        self.attention_b = [nn.LayerNorm(input_dim), nn.Linear(input_dim, out_dim), nn.Sigmoid()]

        if dropout > 0:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(out_dim, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x


class WSIEncoder(nn.Module):
    """Patho/Gene/GMS backbone: linear projection + gated-attention pooling.

    ``captum=True`` additionally returns the attention map for attribution.
    """

    def __init__(self, patch_dim, class_num, captum=False):
        super(WSIEncoder, self).__init__()
        self.mapping_x = nn.Sequential(nn.Linear(patch_dim, patch_dim),
                                       nn.ReLU(),
                                       nn.Dropout(0.1))
        self.wsi_mil = Attention_net_gated(input_dim=patch_dim, out_dim=patch_dim // 2,
                                           n_classes=class_num, dropout=0.1)
        self.captum = captum

    def forward(self, x):
        x = self.mapping_x(x)
        att_embed, _ = self.wsi_mil(x)
        att_embed = torch.transpose(att_embed, 1, 0)  # n_classes,k
        att_embed = F.softmax(att_embed, dim=1)  # n_classes,k
        visual_embedding = torch.mm(att_embed, x)
        if self.captum:
            return visual_embedding, att_embed
        else:
            return visual_embedding


class GSEncoder(nn.Module):
    """GS_ISUP backbone: gated-attention pooling with 3 Gleason-grade heads."""

    def __init__(self, patch_dim):
        super(GSEncoder, self).__init__()
        self.wsi_mil = Attention_net_gated(input_dim=patch_dim, out_dim=patch_dim // 2,
                                           n_classes=3, dropout=0.1)

    def forward(self, x):
        att_embed, _ = self.wsi_mil(x)
        att_embed = torch.transpose(att_embed, 1, 0)  # n_classes,k
        att_embed = F.softmax(att_embed, dim=1)  # n_classes,k
        visual_embedding = torch.mm(att_embed, x)
        return visual_embedding, att_embed, x


class RegionEncoder(nn.Module):
    """BCR stage-1: flat region-level gated-attention MIL -> class logits."""

    def __init__(self, patch_dim, out_class_num):
        super(RegionEncoder, self).__init__()
        self.wsi_mil = Attention_net_gated(input_dim=patch_dim, out_dim=patch_dim // 2,
                                           n_classes=1, dropout=0.1)
        self.classifier = nn.Linear(patch_dim, out_class_num)

    def forward(self, x):
        att_embed, _ = self.wsi_mil(x)
        att_embed = torch.transpose(att_embed, 1, 0)  # n_classes,k
        att_embed = F.softmax(att_embed, dim=1)  # n_classes,k
        visual_embedding = torch.mm(att_embed, x)
        logits = self.classifier(visual_embedding)
        return logits, att_embed


class HierarchicalEncoder(nn.Module):
    """BCR stage-2: two-level (patch->region->WSI) attention, no nuclei features.

    ``x`` is ``[B, num_regions * 256, patch_dim]``; ``patch_per_region == 256``.
    ``with_high_input`` gates the high-res (256) region attention, ``with_low_input``
    fuses the low-res (512) region feature via multi-head attention.
    """

    def __init__(self, patch_dim, wsi_dim, out_class_num, with_low_input=True, with_high_input=True):
        super(HierarchicalEncoder, self).__init__()
        self.region_mil = Attention_net_gated(input_dim=patch_dim, out_dim=patch_dim // 2,
                                               n_classes=1, dropout=0.1)
        self.wsi_mil = Attention_net_gated(input_dim=patch_dim, out_dim=patch_dim // 2,
                                           n_classes=1, dropout=0.1)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=patch_dim, num_heads=4, batch_first=True)
        self.transform_prism = nn.Linear(wsi_dim, wsi_dim // 2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(patch_dim, out_class_num)
        self.with_low = with_low_input
        self.with_high = with_high_input

    def forward(self, x, low_res_feature, wsi_latent):
        B = x.shape[0]
        num_patches = x.shape[1]
        patch_per_region = 256

        num_regions = num_patches // patch_per_region
        x = x.view(B * num_regions, patch_per_region, -1)  # [B*num_regions, 256, patch_dim]

        if self.with_high:
            A_region, _ = self.region_mil(x)  # [B*num_regions, 256, 1]
            weights_region = A_region.transpose(1, 2)  # [B*num_regions, 1, 256]
            weights_region = F.softmax(weights_region, dim=2)

            x_out_region = torch.bmm(weights_region, x)  # [B*num_regions, 1, patch_dim]
            x_out_region = x_out_region.squeeze(1)

            assert x_out_region.shape == low_res_feature.shape

            if self.with_low:
                x_region_fusion, _ = self.multihead_attn(query=x_out_region.unsqueeze(1),
                                                         key=low_res_feature.unsqueeze(1),
                                                         value=low_res_feature.unsqueeze(1))
                x_region_fusion = x_region_fusion.squeeze(1).view(B, num_regions, -1)
            else:
                x_region_fusion = x_out_region.view(B, num_regions, -1)
        else:
            x_region_fusion = low_res_feature.view(B, num_regions, -1)

        A_wsi, _ = self.wsi_mil(x_region_fusion)  # [B, num_regions, 1]
        weights_wsi = A_wsi.transpose(1, 2)  # [B, 1, num_regions]
        weights_wsi = F.softmax(weights_wsi, dim=2)

        visual_embedding = torch.bmm(weights_wsi, x_region_fusion)  # [B, 1, patch_dim]
        visual_embedding = visual_embedding.squeeze(1)

        logits = self.classifier(visual_embedding)
        return logits


class HierarchicalEncoderNuclei(nn.Module):
    """BCR stage-2 with nuclei morphometrics + optional PRISM slide feature.

    ``x`` carries ``patch_dim`` visual dims followed by 85 nuclei dims; the visual
    part is projected down by ``mapping_morph`` (to ``patch_dim - 85``) then
    re-concatenated with the raw nuclei dims to restore ``patch_dim``.
    """

    def __init__(self, patch_dim, wsi_dim, out_class_num, with_prism=True):
        super(HierarchicalEncoderNuclei, self).__init__()
        self.region_mil = Attention_net_gated(input_dim=patch_dim, out_dim=patch_dim // 2,
                                               n_classes=1, dropout=0.1)
        self.wsi_mil = Attention_net_gated(input_dim=patch_dim, out_dim=patch_dim // 2,
                                           n_classes=1, dropout=0.1)
        self.mapping_morph = nn.Sequential(nn.Linear(patch_dim, patch_dim - 85),
                                           nn.ReLU(),
                                           nn.Dropout(0.1))
        self.multihead_attn = nn.MultiheadAttention(embed_dim=patch_dim, num_heads=4, batch_first=True)
        self.transform_prism = nn.Linear(wsi_dim, wsi_dim // 2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)
        if with_prism:
            self.classifier = nn.Linear(wsi_dim // 2 + patch_dim, out_class_num)
        else:
            self.classifier = nn.Linear(patch_dim, out_class_num)
        self.morph_size = patch_dim
        self.with_prism = with_prism

    def forward(self, x, low_res_feature, wsi_latent):
        B = x.shape[0]
        num_patches = x.shape[1]
        patch_per_region = 256

        num_regions = num_patches // patch_per_region
        x = torch.concat([self.mapping_morph(x[0][:, :self.morph_size]), x[0][:, self.morph_size:]], dim=1)
        x = x.view(B * num_regions, patch_per_region, -1)  # [B*num_regions, 256, patch_dim]

        A_region, _ = self.region_mil(x)  # [B*num_regions, 256, 1]
        weights_region = A_region.transpose(1, 2)  # [B*num_regions, 1, 256]
        weights_region = F.softmax(weights_region, dim=2)

        x_out_region = torch.bmm(weights_region, x)  # [B*num_regions, 1, patch_dim]
        x_out_region = x_out_region.squeeze(1)

        assert x_out_region.shape == low_res_feature.shape

        x_region_fusion, _ = self.multihead_attn(query=x_out_region.unsqueeze(1),
                                                 key=low_res_feature.unsqueeze(1),
                                                 value=low_res_feature.unsqueeze(1))
        x_region_fusion = x_region_fusion.squeeze(1).view(B, num_regions, -1)

        A_wsi, _ = self.wsi_mil(x_region_fusion)  # [B, num_regions, 1]
        weights_wsi = A_wsi.transpose(1, 2)  # [B, 1, num_regions]
        weights_wsi = F.softmax(weights_wsi, dim=2)

        visual_embedding = torch.bmm(weights_wsi, x_region_fusion)  # [B, 1, patch_dim]
        visual_embedding = visual_embedding.squeeze(1)

        if self.with_prism:
            wsi_latent = self.transform_prism(wsi_latent)
            wsi_latent = self.relu(wsi_latent)
            wsi_latent = self.drop(wsi_latent)
            visual_embedding = torch.cat([visual_embedding, wsi_latent], dim=1)

        logits = self.classifier(visual_embedding)
        return logits
