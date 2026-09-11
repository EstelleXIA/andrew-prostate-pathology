"""GS_ISUP datasets
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

GS_ISUP_codebook = {(3, 3): 1,
                    (3, 4): 2,
                    (4, 3): 3,
                    (4, 4): 4, (5, 3): 4, (3, 5): 4,
                    (4, 5): 5, (5, 4): 5, (5, 5): 5}


def gs_to_isup(gp_1, gp_2):
    return GS_ISUP_codebook[(int(gp_1), int(gp_2))]


class GSDataset(Dataset):
    """Arch-parametrized GS_ISUP dataset.

    ``nuclei=False``: per-encoder ``.pt`` features, ``final_patients.json``
    intersection, no nuclei/PRISM/SICAP.
    ``nuclei=True``: merged ``.npz`` features + nuclei + PRISM WSI features +
    SICAPv2 train-extra; requires ``model_name == "andrew"``.
    """

    def __init__(self, model_name: str, split: str, nuclei: bool = False):
        super(GSDataset, self).__init__()
        self.model_name = model_name
        self.nuclei = nuclei
        paths = get_paths()

        assert split in ("train", "val", "test", "PLCO", "CPGEA", "TCGA")

        if nuclei:
            self._init_nuclei(paths, model_name, split)
        else:
            self._init_plain(paths, model_name, split)

    # ------------------------------------------------------------------ plain
    def _init_plain(self, paths, model_name, split):
        self.patch_feature_path_256 = paths.mil(C.TASK, C.PLAIN_PATCH_256_TPL.format(model_name=model_name))
        self.patch_feature_path_512 = paths.mil(C.TASK, C.PLAIN_PATCH_512_TPL.format(model_name=model_name))

        label_xlsx = pd.read_excel(paths.labels(C.TASK, C.LABEL_XLSX), index_col=0)

        with open(paths.patients_cache(C.PATIENTS_CACHE_TASK, C.FINAL_PATIENTS_JSON)) as f:
            final_patients_load = json.load(f)

        if model_name == "andrew":
            feature_patients = [x.replace(".pt", "") for x in os.listdir(self.patch_feature_path_512)]
        else:
            feature_patients = [x.replace(".pt", "") for x in os.listdir(self.patch_feature_path_256)]

        label_patients = label_xlsx.index.tolist()

        final_patients = sorted(list(set(
            list(set(feature_patients).intersection(set(label_patients)).intersection(final_patients_load)))))
        label_xlsx = label_xlsx.loc[final_patients]
        label_xlsx.loc[label_xlsx["split"] == "CPGEA", "split"] = "test"  # CPGEA -> test remap (eval-only)

        # Each split is used exactly as labelled; no cross-split mixing. Training
        # uses only the native "train" cohort — TCGA / PLCO / CPGEA are never
        # injected into training.
        self.patients = label_xlsx[label_xlsx["split"] == split].index.tolist()

        self.label = label_xlsx
        # SICAP is not used by the plain (non-nuclei) variant.
        self.SICAP_patients = []

    # ----------------------------------------------------------------- nuclei
    def _init_nuclei(self, paths, model_name, split):
        if model_name != "andrew":
            raise ValueError
        self.patch_feature_path_256 = paths.mil(C.TASK, C.NUCLEI_PATCH_256)
        self.patch_feature_path_512 = paths.mil(C.TASK, C.NUCLEI_PATCH_512)
        self.wsi_feature_path = paths.repgen(C.WSI_FEATURE_DIR)

        label_xlsx = pd.read_excel(paths.labels(C.TASK, C.LABEL_XLSX), index_col=0)
        label_xlsx.loc[label_xlsx["split"] == "CPGEA", "split"] = "test"  # CPGEA -> test remap (eval-only)

        feature_patients_1 = [x.replace(".npz", "") for x in os.listdir(self.patch_feature_path_256)]
        feature_patients_2 = [x.replace(".npz", "") for x in os.listdir(self.patch_feature_path_512)]
        feature_patients = list(set(feature_patients_1).intersection(set(feature_patients_2)))
        wsi_patients = [x.replace(".pt", "") for x in os.listdir(self.wsi_feature_path)]
        label_patients = label_xlsx.index.tolist()
        final_patients = sorted(list(set(feature_patients).intersection(set(label_patients)).intersection(wsi_patients)))
        label_xlsx = label_xlsx.loc[final_patients]

        # Each split is used exactly as labelled; no cross-split mixing. Training
        # uses only the native "train" cohort — TCGA / PLCO / CPGEA are never
        # injected into training.
        self.patients = label_xlsx[label_xlsx["split"] == split].index.tolist()

        self.label = label_xlsx

        ''' add SICAPv2 '''
        self.SICAP_feature_base_256 = paths.mil(C.TASK, C.SICAP_PATCH_256)
        self.SICAP_feature_base_512 = paths.mil(C.TASK, C.SICAP_PATCH_512)
        SICAP_patients_1 = [x.replace(".npz", "") for x in os.listdir(self.SICAP_feature_base_256)]
        SICAP_patients_2 = [x.replace(".npz", "") for x in os.listdir(self.SICAP_feature_base_512)]
        self.SICAP_patients = list(set(SICAP_patients_1).intersection(set(SICAP_patients_2)))

        self.SICAP_instance_label_base_256 = paths.mil(C.TASK, C.SICAP_INSTANCE_LABEL_256)
        self.SICAP_instance_label_base_512 = paths.mil(C.TASK, C.SICAP_INSTANCE_LABEL_512)
        self.SICAP_wsi_feature_path = paths.mil(C.TASK, C.SICAP_WSI_FEATURE)
        if split == "train":
            # SICAPv2 is external training data (weak WSI-level GS labels).
            self.patients = self.patients + self.SICAP_patients
            wsi_label_pd = pd.read_excel(paths.sicap(C.SICAP_WSI_LABELS_XLSX), index_col=0)[
                ["Gleason_primary", "Gleason_secondary"]]
            wsi_label_pd = wsi_label_pd[wsi_label_pd["Gleason_primary"] != 0]
            wsi_label_pd.columns = ["GP_primary", "GP_secondary"]
            wsi_label_pd["ISUP"] = wsi_label_pd.apply(lambda x: gs_to_isup(x["GP_primary"], x["GP_secondary"]), axis=1)

            self.label = pd.concat([self.label, wsi_label_pd], axis=0)

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, item):
        patient_name = self.patients[item]

        if not self.nuclei:
            return self._getitem_plain(patient_name)
        return self._getitem_nuclei(patient_name)

    # ------------------------------------------------------------------ plain
    def _getitem_plain(self, patient_name):
        patch_feature_256 = torch.load(os.path.join(self.patch_feature_path_256, f"{patient_name}.pt"), weights_only=True)
        if self.model_name == "andrew":
            patch_feature_512 = torch.load(os.path.join(self.patch_feature_path_512, f"{patient_name}.pt"), weights_only=True)
        else:
            patch_feature_512 = torch.tensor([0])
        instance_label_256 = 999
        instance_label_512 = 999

        labels = self.label.loc[patient_name]
        gs_primary = labels[C.COL_GP_PRIMARY] - C.GP_OFFSET
        gs_secondary = labels[C.COL_GP_SECONDARY] - C.GP_OFFSET
        isup_score = labels[C.COL_ISUP] - C.ISUP_OFFSET

        return patch_feature_256, patch_feature_512, instance_label_256, instance_label_512, torch.tensor([0]), gs_primary, gs_secondary, isup_score, patient_name

    # ----------------------------------------------------------------- nuclei
    def _getitem_nuclei(self, patient_name):
        if patient_name in self.SICAP_patients:
            with np.load(os.path.join(self.SICAP_feature_base_256, f"{patient_name}.npz"), allow_pickle=True) as loaded:
                patch_feature_256 = loaded["data"]
            with np.load(os.path.join(self.SICAP_feature_base_256, f"{patient_name}.npz"), allow_pickle=True) as loaded:
                patch_index_256 = loaded["index"]
            patch_feature_256 = torch.from_numpy(patch_feature_256)

            with np.load(os.path.join(self.SICAP_feature_base_512, f"{patient_name}.npz"), allow_pickle=True) as loaded:
                patch_feature_512 = loaded["data"]
            with np.load(os.path.join(self.SICAP_feature_base_512, f"{patient_name}.npz"), allow_pickle=True) as loaded:
                patch_index_512 = loaded["index"]
            patch_feature_512 = torch.from_numpy(patch_feature_512)

            instance_label_256 = torch.from_numpy(pd.read_csv(os.path.join(self.SICAP_instance_label_base_256, f"{patient_name}.csv"), index_col=0).loc[patch_index_256].values).squeeze(1)
            instance_label_512 = torch.from_numpy(pd.read_csv(os.path.join(self.SICAP_instance_label_base_512, f"{patient_name}.csv"), index_col=0).loc[patch_index_512].values).squeeze(1)
            wsi_feature = torch.load(os.path.join(self.SICAP_wsi_feature_path, f"{patient_name}.pt"), weights_only=True)[0]

        else:
            with np.load(os.path.join(self.patch_feature_path_256, f"{patient_name}.npz"), allow_pickle=True) as loaded:
                patch_feature_256 = loaded["data"]
            patch_feature_256 = torch.from_numpy(patch_feature_256)

            with np.load(os.path.join(self.patch_feature_path_512, f"{patient_name}.npz"), allow_pickle=True) as loaded:
                patch_feature_512 = loaded["data"]
            patch_feature_512 = torch.from_numpy(patch_feature_512)

            instance_label_256 = 999
            instance_label_512 = 999
            wsi_feature = torch.load(os.path.join(self.wsi_feature_path, f"{patient_name}.pt"), weights_only=True)[0]

        labels = self.label.loc[patient_name]
        gs_primary = labels[C.COL_GP_PRIMARY] - C.GP_OFFSET
        gs_secondary = labels[C.COL_GP_SECONDARY] - C.GP_OFFSET
        isup_score = labels[C.COL_ISUP] - C.ISUP_OFFSET

        return patch_feature_256, patch_feature_512, instance_label_256, instance_label_512, wsi_feature, gs_primary, gs_secondary, isup_score, patient_name


def build_dataset(model_name: str, split: str, arch_type: str) -> GSDataset:
    """Construct the correct (clean) dataset for a given architecture."""
    return GSDataset(model_name=model_name, split=split, nuclei=C.uses_nuclei(arch_type))
