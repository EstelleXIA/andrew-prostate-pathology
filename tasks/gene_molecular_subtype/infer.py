"""Inference / evaluation for the 3-class molecular-subtype model.

Loads a trained checkpoint, runs val + test, writes per-patient probability CSVs and
ROC figures, and reports per-class AUC/Acc/F1 + Macro_AUC_rare (rare defaults to all
three classes for this task).

Usage
-----
    python -m tasks.gene_molecular_subtype.infer --model_type andrew --arch_type no_prism \
        --ckpt_name <checkpoint>.pth
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from monai.data import decollate_batch
from monai.transforms import Activations, AsDiscrete, Compose
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    DEFAULT_DTYPE,
    build_downstream_model,
    calculate_metrics,
    check_arch_invariant,
    get_encoder_dims,
    get_paths,
    resolve_device,
    set_seed,
)
from common.attribution import plot_multiclass_auc

from . import config
from .dataset import make_dataset


@torch.no_grad()
def infer_once(loader, model, dtype, device, save_fig_base, split):
    """Port of utils.infer_once: writes prob CSV + ROC figure, returns metrics dict."""
    patient_name_list = []
    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=config.NUM_CLASSES)])

    y_pred = torch.tensor([], dtype=torch.float32, device=device)
    y = torch.tensor([], dtype=torch.long, device=device)

    for batch_idx, data in tqdm(enumerate(loader)):
        patch_feature_256, patch_feature_512, wsi_feature, labels, patient_name = data
        patient_name_list.append(patient_name[0])

        patch_feature_256 = patch_feature_256.type(dtype).to(device)[0]
        patch_feature_512 = patch_feature_512.type(dtype).to(device)[0]
        wsi_feature = wsi_feature.type(dtype).to(device)
        labels = labels.type(torch.LongTensor).to(device)

        logits = model(patch_feature_256, patch_feature_512, wsi_feature)
        y_pred = torch.cat([y_pred, logits], dim=0)
        y = torch.cat([y, labels], dim=0)

    y_true = [post_label(i).cpu() for i in decollate_batch(y, detach=False)]
    y_pred_prob = [post_pred(i).cpu() for i in decollate_batch(y_pred)]

    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)

    col_names = config.FIGURE_CLASS_NAMES
    y_true_pd = pd.DataFrame(y_true, columns=["gt_" + x for x in col_names], index=patient_name_list)
    y_pred_pd = pd.DataFrame(y_pred_prob, columns=["pred_" + x for x in col_names], index=patient_name_list)
    final_save_pd = pd.concat([y_true_pd, y_pred_pd], axis=1)
    save_pd_base = save_fig_base.replace("figures", "save_pd")
    os.makedirs(save_pd_base, exist_ok=True)
    final_save_pd.to_csv(os.path.join(save_pd_base, f"{split}.csv"))

    metrics = calculate_metrics(y_true, y_pred_prob, class_names=config.METRIC_CLASS_NAMES)

    plot_multiclass_auc(y_true, y_pred_prob, save_fig_base, split,
                        class_names=config.FIGURE_CLASS_NAMES, plot_micro=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Arguments for model inference.")
    parser.add_argument("--model_type", type=str, default="andrew", choices=config.MODEL_TYPES)
    parser.add_argument("--arch_type", type=str, default="no_prism", choices=config.ARCH_TYPES)
    parser.add_argument("--ckpt_name", type=str, default=None,
                        help="checkpoint filename; defaults to the newest *.pth in the ckpt dir")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    set_seed(args.seed)
    check_arch_invariant(args.model_type, args.arch_type)

    patch_dim, wsi_dim = get_encoder_dims(args.model_type)
    dtype = DEFAULT_DTYPE
    device = resolve_device()

    model = build_downstream_model(args.arch_type, num_classes=config.NUM_CLASSES,
                                   input_dim=patch_dim, captum=False)
    model = model.to(dtype).to(device)

    val_dataset = make_dataset(args.model_type, args.arch_type, split="val")
    test_dataset = make_dataset(args.model_type, args.arch_type, split="test")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print(f"Validation on {len(val_dataset)} samples.")
    print(f"Test on {len(test_dataset)} samples.")

    save_ckpt_base = get_paths().mil(config.CKPT_TASK, "ckpt", f"{args.model_type}_{args.arch_type}")
    ckpt_name = args.ckpt_name
    if ckpt_name is None:
        candidates = glob.glob(os.path.join(save_ckpt_base, "*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No *.pth checkpoints found under {save_ckpt_base}")
        ckpt_name = os.path.basename(max(candidates, key=os.path.getmtime))
    print(f"Loading checkpoint: {ckpt_name}")
    checkpoint = torch.load(os.path.join(save_ckpt_base, ckpt_name), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    figure_base = get_paths().patients_cache(config.CKPT_TASK, "figures", f"{args.model_type}_{args.arch_type}")
    os.makedirs(figure_base, exist_ok=True)

    metrics_val = infer_once(val_loader, model, dtype, device, figure_base, "val")
    metrics_test = infer_once(test_loader, model, dtype, device, figure_base, "test")
    print("[Validation] metrics:", metrics_val)
    print("[Test] metrics:", metrics_test)


if __name__ == "__main__":
    main()
