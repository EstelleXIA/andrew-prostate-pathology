"""Inference / evaluation for the binary gene-mutation classifier.

Loads a trained checkpoint, runs val + test, writes per-patient softmax
probabilities to CSV and a positive-class ROC (via ``common.attribution.plot_binary_auc``).

    python -m tasks.gene_expression.infer --model_type andrew --gene_name ETS \
        --arch_type final --ckpt_name <checkpoint>.pth
"""
from __future__ import annotations

import argparse
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
    check_arch_invariant,
    get_encoder_dims,
    get_paths,
    resolve_device,
    set_seed,
)
from common.attribution import plot_binary_auc

from . import config as C
from .dataset import GeneBinaryDataset


def infer_once(loader, model, dtype, device, save_fig_base, split):
    """Run inference, dump per-patient probabilities to CSV, plot the ROC."""
    patient_name_list = []
    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=C.NUM_CLASSES)])
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

    y_true = np.array([post_label(i).cpu() for i in decollate_batch(y, detach=False)])
    y_pred_prob = np.array([post_pred(i).cpu() for i in decollate_batch(y_pred)])

    col_names = C.CLASS_NAMES  # ['Wild', 'Mutation']
    y_true_pd = pd.DataFrame(y_true, columns=["gt_" + x for x in col_names], index=patient_name_list)
    y_pred_pd = pd.DataFrame(y_pred_prob, columns=["pred_" + x for x in col_names], index=patient_name_list)
    final_save_pd = pd.concat([y_true_pd, y_pred_pd], axis=1)
    save_pd_base = save_fig_base.replace("figures", "save_pd")
    os.makedirs(save_pd_base, exist_ok=True)
    final_save_pd.to_csv(os.path.join(save_pd_base, f"{split}.csv"))

    plot_binary_auc(y_true, y_pred_prob, save_fig_base, split, pos_label=C.POS_LABEL)
    return None


def main():
    parser = argparse.ArgumentParser(description="Arguments for model inference.")
    parser.add_argument("--model_type", type=str, default="andrew", choices=list(C.MODEL_TYPES))
    parser.add_argument("--arch_type", type=str, default="only_high", choices=list(C.ARCH_TYPES))
    parser.add_argument("--gene_name", type=str, default="ETS", choices=list(C.GENE_NAMES))
    parser.add_argument("--ckpt_name", type=str, required=True,
                        help="checkpoint filename under the task's ckpt dir")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    check_arch_invariant(args.model_type, args.arch_type)

    patch_dim, wsi_dim = get_encoder_dims(args.model_type)
    dtype = DEFAULT_DTYPE
    gene_name = args.gene_name
    device = resolve_device(args.device)

    model = build_downstream_model(args.arch_type, num_classes=C.NUM_CLASSES, input_dim=patch_dim)
    model = model.to(dtype).to(device)

    val_dataset = GeneBinaryDataset(model_name=args.model_type, split="val",
                                    gene_name=gene_name, arch_type=args.arch_type)
    test_dataset = GeneBinaryDataset(model_name=args.model_type, split="test",
                                     gene_name=gene_name, arch_type=args.arch_type)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print(f"Validation on {len(val_dataset)} samples.")
    print(f"Test on {len(test_dataset)} samples.")

    paths = get_paths()
    save_ckpt_base = paths.ckpt(C.MIL_SUBDIR, args.gene_name, f"{args.model_type}_{args.arch_type}")
    checkpoint = torch.load(os.path.join(save_ckpt_base, args.ckpt_name),
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():
        model.eval()
        # "figures" segment is load-bearing: infer_once derives the CSV dir via
        # save_fig_base.replace("figures", "save_pd").
        figure_base = paths.ckpt(C.MIL_SUBDIR, "figures", args.gene_name,
                                 f"{args.model_type}_{args.arch_type}")
        os.makedirs(figure_base, exist_ok=True)

        infer_once(val_loader, model, dtype, device, figure_base, "val")
        infer_once(test_loader, model, dtype, device, figure_base, "test")


if __name__ == "__main__":
    main()
