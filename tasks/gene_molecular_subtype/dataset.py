"""Datasets for GeneMolecularSubtype (3-class: basal / luminalA / luminalB).

Three variants mirror the model architectures:

  * ``GeneSubtypeDatasetNuclei``   (arch: final / no_prism) — nuclei-merged npz features.
  * ``GeneSubtypeDatasetHierarchy``(arch: no_nuclei)        — hierarchical [CLS] .pt features.
  * ``GeneSubtypeDataset``         (arch: only_low/only_high + non-andrew baselines).

"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common import get_paths

from . import config


def _load_label_table():
    """Load the shared GeneExpression label sheet -> DataFrame[['label', 'split']]."""
    paths = get_paths()
    label_path = paths.labels(config.LABEL_TASK, config.LABEL_FILE)
    label_xlsx = pd.read_excel(label_path, index_col=0)[[config.LABEL_COLUMN, "split"]].dropna()
    label_xlsx["label"] = label_xlsx[config.LABEL_COLUMN].map(config.LABEL_MAP)
    return label_xlsx[["label", "split"]]


def _load_final_patients():
    paths = get_paths()
    with open(paths.patients_cache(config.PATIENTS_CACHE_TASK, config.FINAL_PATIENTS_FILE)) as f:
        return json.load(f)


class GeneSubtypeDatasetNuclei(Dataset):
    def __init__(self, model_name: str, split: str):
        super().__init__()
        assert split in ("train", "val", "test")
        paths = get_paths()
        if model_name != "andrew":
            raise ValueError
        self.patch_feature_path_256 = paths.mil(config.FEATURE_TASK, config.NUCLEI_FEATURE_DIR_256)
        self.patch_feature_path_512 = paths.mil(config.FEATURE_TASK, config.NUCLEI_FEATURE_DIR_512)
        self.wsi_feature_path = paths.mil(config.FEATURE_TASK, config.WSI_FEATURE_DIR)

        label_xlsx = _load_label_table()

        feature_patients_1 = [x.replace(".npz", "") for x in os.listdir(self.patch_feature_path_256)]
        feature_patients_2 = [x.replace(".npz", "") for x in os.listdir(self.patch_feature_path_512)]
        feature_patients = list(set(feature_patients_1).intersection(set(feature_patients_2)))
        label_patients = label_xlsx.index.tolist()
        final_patients = sorted(list(set(feature_patients).intersection(set(label_patients))))

        label_xlsx = label_xlsx.loc[final_patients]

        self.patients = label_xlsx[label_xlsx["split"] == split].index.tolist()
        # Each split is used exactly as labelled; no cross-split mixing.
        self.label = label_xlsx

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, item):
        patient_name = self.patients[item]
        with np.load(os.path.join(self.patch_feature_path_256, f"{patient_name}.npz"), allow_pickle=True) as loaded:
            patch_feature_256 = loaded["data"]
        patch_feature_256 = torch.from_numpy(patch_feature_256)

        with np.load(os.path.join(self.patch_feature_path_512, f"{patient_name}.npz"), allow_pickle=True) as loaded:
            patch_feature_512 = loaded["data"]
        patch_feature_512 = torch.from_numpy(patch_feature_512)

        wsi_feature = torch.load(os.path.join(self.wsi_feature_path, f"{patient_name}.pt"), weights_only=True)[0]

        labels = self.label.loc[patient_name]
        target_label = labels["label"]

        return patch_feature_256, patch_feature_512, wsi_feature, target_label, patient_name


class GeneSubtypeDatasetHierarchy(Dataset):
    def __init__(self, model_name: str, split: str):
        super().__init__()
        assert split in ("train", "val", "test")
        paths = get_paths()
        if model_name != "andrew":
            raise ValueError
        self.patch_feature_path_256 = paths.mil(config.FEATURE_TASK, config.HIER_FEATURE_DIR_256)
        self.patch_feature_path_512 = paths.mil(config.FEATURE_TASK, config.HIER_FEATURE_DIR_512)
        self.wsi_feature_path = paths.mil(config.FEATURE_TASK, config.WSI_FEATURE_DIR)

        label_xlsx = _load_label_table()
        final_patients_load = _load_final_patients()

        feature_patients_1 = [x.replace(".pt", "") for x in os.listdir(self.patch_feature_path_256)]
        feature_patients_2 = [x.replace(".pt", "") for x in os.listdir(self.patch_feature_path_512)]
        feature_patients = list(set(feature_patients_1).intersection(set(feature_patients_2)))
        label_patients = label_xlsx.index.tolist()
        final_patients = sorted(list(
            set(feature_patients).intersection(set(label_patients)).intersection(final_patients_load)))
        label_xlsx = label_xlsx.loc[final_patients]

        self.patients = label_xlsx[label_xlsx["split"] == split].index.tolist()
        # Each split is used exactly as labelled; no cross-split mixing.
        self.label = label_xlsx

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, item):
        patient_name = self.patients[item]

        patch_feature_256 = torch.load(os.path.join(self.patch_feature_path_256, f"{patient_name}.pt"),
                                       weights_only=True)

        patch_feature_files_512 = sorted(os.listdir(os.path.join(self.patch_feature_path_512, patient_name)))
        patch_feature_list_512 = []
        for file in patch_feature_files_512:
            patch_feature_list_512.append(
                torch.load(os.path.join(self.patch_feature_path_512, patient_name, file), weights_only=True))
        patch_feature_512 = torch.concat(patch_feature_list_512, dim=0)

        wsi_feature = torch.load(os.path.join(self.wsi_feature_path, f"{patient_name}.pt"), weights_only=True)[0]

        labels = self.label.loc[patient_name]
        target_label = labels["label"]

        return patch_feature_256, patch_feature_512, wsi_feature, target_label, patient_name


class GeneSubtypeDataset(Dataset):
    def __init__(self, model_name: str, split: str, arch_name: str):
        super().__init__()
        assert split in ("train", "val", "test")
        paths = get_paths()
        base = lambda sub: paths.mil(config.FEATURE_TASK, sub)
        if model_name == "uni":
            self.patch_feature_path = base(config.SINGLE_FEATURE_DIRS["uni"])
        elif model_name == "virchow":
            self.patch_feature_path = base(config.SINGLE_FEATURE_DIRS["virchow"])
        elif model_name == "andrew":
            if arch_name == "only_high":
                self.patch_feature_path = base(config.SINGLE_FEATURE_DIRS["andrew_only_high"])
            elif arch_name == "only_low":
                self.patch_feature_path = base(config.SINGLE_FEATURE_DIRS["andrew_only_low"])
            else:
                raise ValueError
        elif model_name == "gigapath":
            self.patch_feature_path = base(config.SINGLE_FEATURE_DIRS["gigapath"])
        else:
            raise ValueError

        label_xlsx = _load_label_table()
        final_patients_load = _load_final_patients()

        feature_patients = [x.replace(".pt", "") for x in os.listdir(self.patch_feature_path)]
        label_patients = label_xlsx.index.tolist()
        final_patients = sorted(list(
            set(feature_patients).intersection(set(label_patients)).intersection(final_patients_load)))
        label_xlsx = label_xlsx.loc[final_patients]

        self.patients = label_xlsx[label_xlsx["split"] == split].index.tolist()
        # Each split is used exactly as labelled; no cross-split mixing.
        self.label = label_xlsx

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, item):
        patient_name = self.patients[item]
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

        labels = self.label.loc[patient_name]
        target_label = labels["label"]

        return patch_feature_256, torch.tensor([0]), torch.tensor([0]), target_label, patient_name


def make_dataset(model_type: str, arch_type: str, split: str) -> Dataset:
    """Select the dataset variant for the given architecture."""
    if model_type == "andrew":
        if arch_type in ("final", "no_prism"):
            return GeneSubtypeDatasetNuclei(model_name=model_type, split=split)
        if arch_type == "no_nuclei":
            return GeneSubtypeDatasetHierarchy(model_name=model_type, split=split)
        if arch_type == "only_low":
            return GeneSubtypeDataset(model_name=model_type, split=split, arch_name="only_low")
        if arch_type == "only_high":
            return GeneSubtypeDataset(model_name=model_type, split=split, arch_name="only_high")
        raise ValueError(f"Unknown arch_type: {arch_type!r}")
    # non-andrew baselines are single-scale only_high
    return GeneSubtypeDataset(model_name=model_type, split=split, arch_name="only_high")
