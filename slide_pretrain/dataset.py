"""Dataset for PRISM slide-encoder fine-tuning.

Each example is one patient: all region ``.pt`` tile features are concatenated, and
the per-tile [CLS] and average-pool features are concatenated along the feature dim
to form the 2560-dim tile embedding PRISM expects (1280 [CLS] + 1280 avg).

Feature layout (under ``repgen_tumor_root``):
    patient_features_20x_256_andrew_cls/<patient>/<region>.pt   # [n_tiles, 1280]
    patient_features_20x_256_andrew_avg/<patient>/<region>.pt   # [n_tiles, 1280]

``report`` annotations come from a JSON with ``{"train": [...], "val": [...], "test": [...]}``
where each entry is ``{"id": <patient>, "report": <text>}``.
"""
from __future__ import annotations

import glob
import math
import os
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

CLS_DIRNAME = "patient_features_20x_256_andrew_cls"
AVG_DIRNAME = "patient_features_20x_256_andrew_avg"


class ProstateDataset(Dataset):
    """Per-patient tile features + report.

    Args:
        cls_feature_root: dir holding ``<patient>/<region>.pt`` [CLS] features.
        ann_json: path to the report-annotation JSON.
        split: which split to load.
        max_patch_num: cap on tiles per patient.
    """

    def __init__(self, cls_feature_root: str, ann_json: str, split: str = "train",
                 max_patch_num: int = 100 * 256):
        super().__init__()
        self.base_path = cls_feature_root
        self.avg_base_path = cls_feature_root.replace(CLS_DIRNAME, AVG_DIRNAME)

        import json
        with open(ann_json, "r") as f:
            self.ann = json.load(f)

        self.split = split
        self.examples = self.ann[split]
        self.max_patch_num = max_patch_num

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        example = self.examples[item]
        selected_patient = example["id"]

        region_files = sorted(glob.glob(os.path.join(self.base_path, selected_patient, "*.pt")))

        all_features_cls, all_features_avg = [], []
        for region in region_files:
            all_features_cls.append(torch.load(region, weights_only=True))
            all_features_avg.append(torch.load(region.replace(CLS_DIRNAME, AVG_DIRNAME),
                                               weights_only=True))
        images_cls = torch.concat(all_features_cls, dim=0)
        images_avg = torch.concat(all_features_avg, dim=0)
        if images_cls.shape[0] > self.max_patch_num:
            selected_patch_id = random.sample(list(range(images_cls.shape[0])), self.max_patch_num)
            images_cls = images_cls[selected_patch_id]
            images_avg = images_avg[selected_patch_id]

        images = torch.concat([images_cls, images_avg], dim=1)
        report_raw = example['report']

        return selected_patient, images, report_raw


def collate_fn(all_inputs):
    """Pad tile sequences to a multiple of 8 and build the attention mask."""
    patient_names = [single_input[0] for single_input in all_inputs]
    patient_images = pad_sequence([single_input[1] for single_input in all_inputs], batch_first=True)
    seq_lengths = torch.tensor([single_input[1].shape[0] for single_input in all_inputs])

    patient_images = torch.nn.functional.pad(
        patient_images,
        (0, 0, 0, math.ceil(patient_images.size(1) / 8) * 8 - patient_images.size(1)),
        value=0)
    max_length = patient_images.size(1)

    patient_image_masks = torch.arange(max_length).expand(len(seq_lengths), max_length) < seq_lengths.unsqueeze(1)
    patient_reports = [single_input[2] for single_input in all_inputs]
    return patient_names, patient_images, patient_image_masks, patient_reports
