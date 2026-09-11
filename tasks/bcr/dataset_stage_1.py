"""Stage-1 dataset: flat region-level survival bags.

Each item is the set of low-res (4096px region) features for one patient, plus
the discrete-time survival label. 
"""
from __future__ import annotations

import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from common import get_paths

from . import config


class SurvDataset(Dataset):
    def __init__(self, model_name: str, split: str):
        super(SurvDataset, self).__init__()
        paths = get_paths()

        patch_dir, wsi_dir = config.STAGE1_FEATURE_DIRS[model_name]
        self.patch_feature_path = paths.repgen(patch_dir)
        self.wsi_feature_path = paths.repgen(wsi_dir)

        assert split in ("train", "val", "test", "PLCO")

        label_path = paths.labels(config.TASK_DIR, config.LABEL_XLSX)
        label_xlsx = pd.read_excel(label_path, index_col=0)

        feature_patients = [x.replace(".pt", "") for x in os.listdir(self.patch_feature_path)]
        wsi_patients = [x.replace(".pt", "") for x in os.listdir(self.wsi_feature_path)]
        label_patients = label_xlsx.index.tolist()
        final_patients = sorted(
            list(set(feature_patients).intersection(set(label_patients)).intersection(wsi_patients)))
        label_xlsx = label_xlsx.loc[final_patients]

        self.patients = label_xlsx[label_xlsx["split"] == split].index.tolist()

        uncensored_df = label_xlsx[label_xlsx['event'] == 1]

        disc_labels, q_bins = pd.qcut(uncensored_df["time"], q=config.N_TIME_BINS, retbins=True, labels=False)
        q_bins[-1] = label_xlsx["time"].max() + 1e-6
        q_bins[0] = label_xlsx["time"].min() - 1e-6

        disc_labels, q_bins = pd.cut(label_xlsx["time"], bins=q_bins, retbins=True, labels=False, right=False,
                                     include_lowest=True)
        label_xlsx.insert(2, 'time_bins', disc_labels.values.astype(int))
        self.label = label_xlsx

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, item):
        patient_name = self.patients[item]
        patch_feature = torch.load(os.path.join(self.patch_feature_path, f"{patient_name}.pt"), weights_only=True)

        labels = self.label.loc[patient_name]
        surv_discrete = labels["time_bins"]
        surv_time = labels["time"]
        censor = 1 - labels["event"]

        return patch_feature, surv_discrete, surv_time, censor, patient_name
