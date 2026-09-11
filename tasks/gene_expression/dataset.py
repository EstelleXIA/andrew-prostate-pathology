"""Arch-parametrized dataset for the binary gene-mutation task.

One ``Dataset`` class parametrized by ``arch_type`` covers three on-disk feature
layouts:

  * arch: final / no_prism     -> ``.npz`` merged 256/512 features (morphology +
        nuclei), both scales real, plus PRISM wsi.
  * arch: no_nuclei            -> ``.pt`` cls features; 256 is a single tensor, 512
        is a per-patient directory of tensors concatenated; plus PRISM wsi.
  * arch: only_low / only_high -> single-scale ``.pt`` features; the 512-slot and
        wsi-slot are returned as dummy ``torch.tensor([0])`` placeholders (the
        single-scale model ignores them).

Return tuples and dummy placeholders differ per arch (mirrors the
``common.build_downstream_model`` pattern).
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


class GeneBinaryDataset(Dataset):
    """Unified binary gene-mutation dataset.

    Parameters
    ----------
    model_name : one of ``uni``/``virchow``/``gigapath``/``andrew`` (patch encoder).
    split      : ``train`` / ``val`` / ``test``.
    gene_name  : which gene column becomes the binary ``label`` (see config.GENE_NAMES).
    arch_type  : selects the on-disk feature layout / return shape
                 (final, no_prism, no_nuclei, only_low, only_high).
    """

    def __init__(self, model_name: str, split: str, gene_name: str, arch_type: str):
        super().__init__()
        assert split in ("train", "val", "test")
        assert gene_name in C.GENE_NAMES
        self.mode = C.arch_mode(arch_type)

        paths = get_paths()

        # ---- resolve feature directories per mode --------------------------
        if self.mode == "nuclei":
            if model_name != "andrew":
                raise ValueError("nuclei arch (final/no_prism) requires model_name='andrew'")
            self.patch_feature_path_256 = paths.mil(C.MIL_SUBDIR, C.NUCLEI_FEATURE_DIR_256)
            self.patch_feature_path_512 = paths.mil(C.MIL_SUBDIR, C.NUCLEI_FEATURE_DIR_512)
            self.wsi_feature_path = paths.mil(C.MIL_SUBDIR, C.WSI_FEATURE_DIR)
        elif self.mode == "hierarchy":
            if model_name != "andrew":
                raise ValueError("no_nuclei arch requires model_name='andrew'")
            self.patch_feature_path_256 = paths.mil(C.MIL_SUBDIR, C.HIER_FEATURE_DIR_256)
            self.patch_feature_path_512 = paths.mil(C.MIL_SUBDIR, C.HIER_FEATURE_DIR_512)
            self.wsi_feature_path = paths.mil(C.MIL_SUBDIR, C.WSI_FEATURE_DIR)
        else:  # single
            self.patch_feature_path = paths.mil(
                C.MIL_SUBDIR, C.single_scale_feature_dir(model_name, arch_type))

        # ---- labels: rename the gene column to "label" ---------------------
        label_path = paths.labels(C.LABELS_SUBDIR, C.LABEL_XLSX)
        label_xlsx = pd.read_excel(label_path, index_col=0)[[gene_name, "split"]].dropna()
        label_xlsx.columns = ["label", "split"]

        # ---- patient set: features present AND labelled --------------------
        if self.mode == "nuclei":
            # Nuclei mode discovers final_patients from disk (both .npz scales) and does
            # NOT read the final_patients.json cache.
            feature_patients_1 = [x.replace(".npz", "") for x in os.listdir(self.patch_feature_path_256)]
            feature_patients_2 = [x.replace(".npz", "") for x in os.listdir(self.patch_feature_path_512)]
            feature_patients = list(set(feature_patients_1).intersection(set(feature_patients_2)))
            label_patients = label_xlsx.index.tolist()
            final_patients = sorted(list(set(feature_patients).intersection(set(label_patients))))
        elif self.mode == "hierarchy":
            with open(paths.patients_cache(C.PATIENTS_CACHE_SUBDIR, C.PATIENTS_CACHE_JSON)) as f:
                final_patients_load = json.load(f)
            feature_patients_1 = [x.replace(".pt", "") for x in os.listdir(self.patch_feature_path_256)]
            feature_patients_2 = [x.replace(".pt", "") for x in os.listdir(self.patch_feature_path_512)]
            feature_patients = list(set(feature_patients_1).intersection(set(feature_patients_2)))
            label_patients = label_xlsx.index.tolist()
            final_patients = sorted(list(set(feature_patients)
                                         .intersection(set(label_patients))
                                         .intersection(final_patients_load)))
        else:  # single
            with open(paths.patients_cache(C.PATIENTS_CACHE_SUBDIR, C.PATIENTS_CACHE_JSON)) as f:
                final_patients_load = json.load(f)
            feature_patients = [x.replace(".pt", "") for x in os.listdir(self.patch_feature_path)]
            label_patients = label_xlsx.index.tolist()
            final_patients = sorted(list(set(feature_patients)
                                         .intersection(set(label_patients))
                                         .intersection(final_patients_load)))

        label_xlsx = label_xlsx.loc[final_patients]

        self.patients = label_xlsx[label_xlsx["split"] == split].index.tolist()
     
        self.label = label_xlsx

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, item):
        patient_name = self.patients[item]

        if self.mode == "nuclei":
            with np.load(os.path.join(self.patch_feature_path_256, f"{patient_name}.npz"),
                         allow_pickle=True) as loaded:
                patch_feature_256 = torch.from_numpy(loaded["data"])
            with np.load(os.path.join(self.patch_feature_path_512, f"{patient_name}.npz"),
                         allow_pickle=True) as loaded:
                patch_feature_512 = torch.from_numpy(loaded["data"])
            wsi_feature = torch.load(os.path.join(self.wsi_feature_path, f"{patient_name}.pt"),
                                     weights_only=True)[0]
            target_label = self.label.loc[patient_name]["label"]
            return patch_feature_256, patch_feature_512, wsi_feature, target_label, patient_name

        if self.mode == "hierarchy":
            patch_feature_256 = torch.load(os.path.join(self.patch_feature_path_256, f"{patient_name}.pt"),
                                           weights_only=True)
            patch_feature_files_512 = sorted(os.listdir(os.path.join(self.patch_feature_path_512, patient_name)))
            patch_feature_list_512 = []
            for file in patch_feature_files_512:
                patch_feature_list_512.append(
                    torch.load(os.path.join(self.patch_feature_path_512, patient_name, file), weights_only=True))
            patch_feature_512 = torch.concat(patch_feature_list_512, dim=0)
            wsi_feature = torch.load(os.path.join(self.wsi_feature_path, f"{patient_name}.pt"),
                                     weights_only=True)[0]
            target_label = self.label.loc[patient_name]["label"]
            return patch_feature_256, patch_feature_512, wsi_feature, target_label, patient_name

        if "512" in self.patch_feature_path:
            patch_feature_files_256 = sorted(os.listdir(os.path.join(self.patch_feature_path, patient_name)))
            patch_feature_list_256 = []
            for file in patch_feature_files_256:
                patch_feature_list_256.append(
                    torch.load(os.path.join(self.patch_feature_path, patient_name, file), weights_only=True))
            patch_feature_256 = torch.concat(patch_feature_list_256, dim=0)
        else:
            patch_feature_256 = torch.load(os.path.join(self.patch_feature_path, f"{patient_name}.pt"),
                                           weights_only=True)
        target_label = self.label.loc[patient_name]["label"]
        return patch_feature_256, torch.tensor([0]), torch.tensor([0]), target_label, patient_name
