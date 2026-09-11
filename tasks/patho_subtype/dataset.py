"""PathoSubtype dataset.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common import get_paths

from . import config as C


class SubtypeDataset(Dataset):
    """Unified multilabel PathoSubtype dataset (all arch variants)."""

    def __init__(self, model_type: str, arch_type: str, split: str):
        super().__init__()
        assert split in ("train", "val", "test")
        self.model_type = model_type
        self.arch_type = arch_type
        self.split = split
        self.mode = C.dataset_mode(arch_type)

        paths = get_paths()

        self.wsi_feature_path = None
        if self.mode == "nuclei":
            self.patch_feature_path_256 = paths.mil(C.PATCH_DATASET, C.FEATURE_256_MERGED)
            self.patch_feature_path_512 = paths.mil(C.PATCH_DATASET, C.FEATURE_512_MERGED)
            self.wsi_feature_path = paths.repgen(C.WSI_FEATURE_DIR)
            feature_ext = ".npz"
        elif self.mode == "hierarchy":
            self.patch_feature_path_256 = paths.mil(C.PATCH_DATASET, C.FEATURE_256_ANDREW)
            self.patch_feature_path_512 = paths.mil(C.PATCH_DATASET, C.FEATURE_512_ANDREW)
            feature_ext = ".pt"
        else:  # single
            if model_type == "andrew":
                key = f"andrew_{arch_type}"
            else:
                key = model_type
            if key not in C.SINGLE_DIR_BY_MODEL:
                raise ValueError(f"No single-scale feature dir for model_type={model_type!r}, "
                                 f"arch_type={arch_type!r}")
            self.patch_feature_path = paths.mil(C.PATCH_DATASET, C.SINGLE_DIR_BY_MODEL[key])
            feature_ext = ".pt"

        # ---- labels ----------------------------------------------------------------
        label_xlsx = pd.read_excel(paths.labels(C.LABEL_TASK, C.LABEL_XLSX), index_col=0)
        label_patients = label_xlsx.index.tolist()

        # ---- patients present as features ------------------------------------------
        if self.mode in ("nuclei", "hierarchy"):
            feature_patients_1 = [x.replace(feature_ext, "") for x in os.listdir(self.patch_feature_path_256)]
            feature_patients_2 = [x.replace(feature_ext, "") for x in os.listdir(self.patch_feature_path_512)]
            feature_patients = list(set(feature_patients_1).intersection(set(feature_patients_2)))
        else:
            feature_patients = [x.replace(feature_ext, "") for x in os.listdir(self.patch_feature_path)]
        candidate = set(feature_patients).intersection(set(label_patients))
        
        with open(paths.patients_cache(C.LABEL_TASK, C.FINAL_PATIENTS_JSON)) as f:
            final_patients_load = json.load(f)
        candidate = candidate.intersection(set(final_patients_load))

        final_patients = sorted(candidate)
        label_xlsx = label_xlsx.loc[final_patients]

        self.patients = label_xlsx[label_xlsx["split"] == split].index.tolist()
        self.label = label_xlsx.loc[self.patients]
        self.label = self.label[~self.label.index.duplicated(keep="first")]
        self.label = self.label[self.label["split"] == split]
        self.final_patients = self.label.index.tolist()

    def __len__(self):
        return len(self.final_patients)

    def _labels(self, patient_name):
        return torch.from_numpy(
            self.label.loc[patient_name, C.LABEL_COLUMNS].values.astype(int)
        )

    def __getitem__(self, item):
        patient_name = self.final_patients[item]
        labels = self._labels(patient_name)

        if self.mode == "nuclei":
            with np.load(os.path.join(self.patch_feature_path_256, f"{patient_name}.npz"),
                         allow_pickle=True) as loaded:
                patch_feature_256 = torch.from_numpy(loaded["data"])
            with np.load(os.path.join(self.patch_feature_path_512, f"{patient_name}.npz"),
                         allow_pickle=True) as loaded:
                patch_feature_512 = torch.from_numpy(loaded["data"])
            wsi_feature = torch.load(os.path.join(self.wsi_feature_path, f"{patient_name}.pt"),
                                     weights_only=True)[0]
            return patch_feature_256, patch_feature_512, wsi_feature, labels, patient_name

        if self.mode == "hierarchy":
            patch_feature_256 = torch.load(os.path.join(self.patch_feature_path_256, f"{patient_name}.pt"),
                                           weights_only=True)
            patch_feature_512 = torch.load(os.path.join(self.patch_feature_path_512, f"{patient_name}.pt"),
                                           weights_only=True)
            return patch_feature_256, patch_feature_512, torch.tensor([0]), labels, patient_name
            
        patch_feature = torch.load(os.path.join(self.patch_feature_path, f"{patient_name}.pt"),
                                   weights_only=True)
        return patch_feature, torch.tensor([0]), torch.tensor([0]), labels, patient_name
