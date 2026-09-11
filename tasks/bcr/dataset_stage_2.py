"""Stage-2 datasets: hierarchical (region -> WSI) survival bags.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common import get_paths

from . import config


def _default_patch_dirs(paths):
    primary = paths.repgen(config.PATCHES_SUBDIR)
    return primary, primary


def _read_regions_and_attn(patient_name, patch_base_primary, patch_base_archive, attn_base):
    """Enumerate a patient's region tiles and load the matching stage-1 attention."""
    primary_patients = os.listdir(patch_base_primary)
    patch_base = patch_base_primary if patient_name in primary_patients else patch_base_archive
    all_files = sorted(glob.glob(os.path.join(patch_base, patient_name, "*", "*.jpg")))

    attn_path = os.path.join(attn_base, f"{patient_name}.npy")
    attn_weights = np.load(attn_path)
    assert len(all_files) == attn_weights.shape[0]
    return all_files, attn_weights


class SurvDataset(Dataset):
    def __init__(self, model_name: str, split: str, infer=False,
                 attn_dir=None, patches_dir=None, patches_archive_dir=None):
        super(SurvDataset, self).__init__()
        paths = get_paths()

        if attn_dir is None:
            attn_dir = paths.mil(config.TASK_DIR, config.REGION_ATTN_SUBDIR,
                                 config.stage2_attn_subdir(model_name))
        self.attn_base = attn_dir

        patch_dir, low_res_dir, wsi_dir = config.STAGE2_FEATURE_DIRS[model_name]
        self.patch_feature_path = paths.repgen(patch_dir)
        self.low_res_feature_path = paths.repgen(low_res_dir) if low_res_dir else None
        self.wsi_feature_path = paths.repgen(wsi_dir)

        _prim, _arch = _default_patch_dirs(paths)
        self.patch_base_primary = patches_dir or _prim
        self.patch_base_archive = patches_archive_dir or _arch

        assert split in ("train", "val", "test", "PLCO", "CPGEA", "TCGA")

        label_path = paths.labels(config.TASK_DIR, config.LABEL_XLSX)
        label_xlsx = pd.read_excel(label_path, index_col=0)
        if infer:
            label_xlsx.loc[label_xlsx["split"] == "CPGEA", "split"] = "test"

        '''remove no tumor patients'''
        feature_patients = [x.replace(".pt", "") for x in os.listdir(self.low_res_feature_path)]
        wsi_patients = [x.replace(".pt", "") for x in os.listdir(self.wsi_feature_path)]
        label_patients = label_xlsx.index.tolist()
        with open(paths.patients_cache(config.TASK_DIR, config.FINAL_PATIENTS_JSON)) as f:
            final_patients_load = json.load(f)

        final_patients = sorted(
            list(set(feature_patients).intersection(set(label_patients)).intersection(final_patients_load)))
        label_xlsx = label_xlsx.loc[final_patients]

        # Each split is used exactly as labelled; no cross-split mixing. Training
        # uses only the native "train" cohort — TCGA / PLCO / CPGEA are never
        # injected into training.
        self.patients = label_xlsx[label_xlsx["split"] == split].index.tolist()

        uncensored_df = label_xlsx[label_xlsx['event'] == 1]

        disc_labels, q_bins = pd.qcut(uncensored_df["time"], q=config.N_TIME_BINS, retbins=True, labels=False)
        q_bins[-1] = label_xlsx["time"].max() + 1e-6
        q_bins[0] = label_xlsx["time"].min() - 1e-6

        disc_labels, q_bins = pd.cut(label_xlsx["time"], bins=q_bins, retbins=True, labels=False, right=False,
                                     include_lowest=True)
        label_xlsx.insert(2, 'time_bins', disc_labels.values.astype(int))
        self.label = label_xlsx
        self.max_region_num = config.MAX_REGION_NUM
        self.model_name = model_name

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, item):
        patient_name = self.patients[item]
        all_files, attn_weights = _read_regions_and_attn(
            patient_name, self.patch_base_primary, self.patch_base_archive, self.attn_base)

        low_res_feature_raw = torch.load(os.path.join(self.low_res_feature_path, f"{patient_name}.pt"),
                                         weights_only=True)

        if len(all_files) > self.max_region_num:
            top_indexes = np.argsort(attn_weights)[-self.max_region_num:]
            selected_regions = [all_files[i] for i in top_indexes]
            low_res_feature = low_res_feature_raw[top_indexes]
        else:
            selected_regions = all_files
            low_res_feature = low_res_feature_raw

        all_features = []
        if self.model_name in config.CLS_SUFFIX_MODELS:
            suffix_id = "_cls"
        else:
            suffix_id = ""
        for region in selected_regions:
            all_features.append(torch.load(
                region.replace(config.PATCHES_SUBDIR, f"features_4096_{self.model_name}{suffix_id}").replace(".jpg", ".pt"),
                weights_only=True))
        patch_feature = torch.concat(all_features, dim=0)

        labels = self.label.loc[patient_name]
        surv_discrete = labels["time_bins"]
        surv_time = labels["time"]
        censor = 1 - labels["event"]

        return patch_feature, low_res_feature, torch.tensor([0]), surv_discrete, surv_time, censor, patient_name


class SurvDatasetNuclei(Dataset):
    def __init__(self, model_name: str, split: str, infer=False,
                 attn_dir=None, patches_dir=None, patches_archive_dir=None):
        super(SurvDatasetNuclei, self).__init__()
        paths = get_paths()

        if attn_dir is None:
            attn_dir = paths.mil(config.TASK_DIR, config.REGION_ATTN_SUBDIR,
                                 config.stage2_attn_subdir(model_name))
        self.attn_base = attn_dir

        if model_name == "andrew":
            patch_dir, low_res_dir, wsi_dir = config.STAGE2_FEATURE_DIRS["andrew"]
            self.patch_feature_path = paths.repgen(patch_dir)
            self.low_res_feature_path = paths.repgen(low_res_dir)
            self.wsi_feature_path = paths.repgen(wsi_dir)
        else:
            raise ValueError

        _prim, _arch = _default_patch_dirs(paths)
        self.patch_base_primary = patches_dir or _prim
        self.patch_base_archive = patches_archive_dir or _arch

        assert split in ("train", "val", "test", "PLCO", "CPGEA", "TCGA")

        label_path = paths.labels(config.TASK_DIR, config.LABEL_XLSX)
        label_xlsx = pd.read_excel(label_path, index_col=0)
        '''remove no tumor patients'''
        feature_patients = [x.replace(".pt", "") for x in os.listdir(self.low_res_feature_path)]

        self.nuclei_path = paths.mil(config.TASK_DIR, config.NUCLEI_SUBDIR)
        nuclei_patients = [x.replace(".npy", "") for x in os.listdir(self.nuclei_path)]
        wsi_patients = [x.replace(".pt", "") for x in os.listdir(self.wsi_feature_path)]
        label_patients = label_xlsx.index.tolist()
        final_patients = sorted(list(
            set(feature_patients).intersection(set(label_patients)).intersection(wsi_patients).intersection(
                set(nuclei_patients))))

        label_xlsx = label_xlsx.loc[final_patients]
        if infer:
            label_xlsx.loc[label_xlsx["split"] == "CPGEA", "split"] = "test"

        # Each split is used exactly as labelled; no cross-split mixing. Training
        # uses only the native "train" cohort — TCGA / PLCO / CPGEA are never
        # injected into training.
        self.patients = label_xlsx[label_xlsx["split"] == split].index.tolist()

        uncensored_df = label_xlsx[label_xlsx['event'] == 1]

        disc_labels, q_bins = pd.qcut(uncensored_df["time"], q=config.N_TIME_BINS, retbins=True, labels=False)
        q_bins[-1] = label_xlsx["time"].max() + 1e-6
        q_bins[0] = label_xlsx["time"].min() - 1e-6

        disc_labels, q_bins = pd.cut(label_xlsx["time"], bins=q_bins, retbins=True, labels=False, right=False,
                                     include_lowest=True)
        label_xlsx.insert(2, 'time_bins', disc_labels.values.astype(int))
        self.label = label_xlsx
        self.max_region_num = config.MAX_REGION_NUM
        self.model_name = model_name

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, item):
        patient_name = self.patients[item]
        all_files, attn_weights = _read_regions_and_attn(
            patient_name, self.patch_base_primary, self.patch_base_archive, self.attn_base)

        low_res_feature_raw = torch.load(os.path.join(self.low_res_feature_path, f"{patient_name}.pt"),
                                         weights_only=True)

        if len(all_files) > self.max_region_num:
            top_indexes = np.argsort(attn_weights)[-self.max_region_num:]
            selected_regions = [all_files[i] for i in top_indexes]
        else:
            selected_regions = all_files
            top_indexes = np.array(list(range(len(all_files))))

        all_features = []
        if self.model_name in config.CLS_SUFFIX_MODELS:
            suffix_id = "_cls"
        else:
            suffix_id = ""

        top_index_again = []
        for region_id, region in enumerate(selected_regions):
            nuclei_feature_path = os.path.join(self.nuclei_path, os.path.basename(region).split("_")[0],
                                               os.path.basename(region).replace(".jpg", ".npy"))
            if os.path.exists(nuclei_feature_path):
                morph_feature = torch.load(
                    region.replace(config.PATCHES_SUBDIR, f"features_4096_{self.model_name}{suffix_id}").replace(".jpg", ".pt"),
                    weights_only=True)
                nuclei_feature = torch.from_numpy(np.load(nuclei_feature_path))
                merge_feature = torch.concat([morph_feature, nuclei_feature], dim=1)
                all_features.append(merge_feature)
                top_index_again.append(top_indexes[region_id])

        low_res_feature = low_res_feature_raw[np.array(top_index_again)]

        patch_feature = torch.concat(all_features, dim=0)
        wsi_feature = torch.load(os.path.join(self.wsi_feature_path, f"{patient_name}.pt"), weights_only=True)[0]

        labels = self.label.loc[patient_name]
        surv_discrete = labels["time_bins"]
        surv_time = labels["time"]
        censor = 1 - labels["event"]

        return patch_feature, low_res_feature, wsi_feature, surv_discrete, surv_time, censor, patient_name
