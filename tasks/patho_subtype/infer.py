"""Inference / evaluation for the multilabel PathoSubtype classifier.

Usage
-----
    python -m tasks.patho_subtype.infer --model_type andrew --arch_type final \
        [--ckpt_name <checkpoint>.pth]

"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd
import torch
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

from . import config as C
from .dataset import SubtypeDataset


def infer_once(loader, model, dtype, device, save_pd_base, save_fig_base, split):
    all_preds, all_labels, patient_name_list = [], [], []

    for data in tqdm(loader):
        patch_feature_256, patch_feature_512, wsi_feature, labels, patient_name = data
        patient_name_list.append(patient_name[0])

        patch_feature_256 = patch_feature_256.type(dtype).to(device)[0]
        patch_feature_512 = patch_feature_512.type(dtype).to(device)[0]
        wsi_feature = wsi_feature.type(dtype).to(device)
        labels = labels.type(dtype).to(device)

        logits = model(patch_feature_256, patch_feature_512, wsi_feature)
        all_preds.append(torch.sigmoid(logits).cpu())
        all_labels.append(labels.cpu())

    y_true = torch.cat(all_labels).detach().to(torch.float).numpy()
    y_pred_prob = torch.cat(all_preds).detach().to(torch.float).numpy()

    y_true_pd = pd.DataFrame(y_true, columns=["gt_" + x for x in C.CLASS_NAMES], index=patient_name_list)
    y_pred_pd = pd.DataFrame(y_pred_prob, columns=["pred_" + x for x in C.CLASS_NAMES], index=patient_name_list)
    os.makedirs(save_pd_base, exist_ok=True)
    pd.concat([y_true_pd, y_pred_pd], axis=1).to_csv(os.path.join(save_pd_base, f"{split}.csv"))

    metrics = calculate_metrics(y_true, y_pred_prob,
                                class_names=C.CLASS_NAMES, rare_class_names=C.RARE_CLASS_NAMES)
    plot_multiclass_auc(y_true, y_pred_prob, save_fig_base, split, class_names=C.CLASS_NAMES)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Arguments for PathoSubtype inference.")
    parser.add_argument("--model_type", type=str, default="andrew",
                        choices=["uni", "virchow", "gigapath", "andrew"])
    parser.add_argument("--arch_type", type=str, default="final",
                        choices=["final", "no_nuclei", "no_prism", "only_low", "only_high"])
    parser.add_argument("--ckpt_name", type=str, default=None,
                        help="checkpoint filename; defaults to the newest *.pth in the ckpt dir")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    set_seed(args.seed)
    check_arch_invariant(args.model_type, args.arch_type)

    patch_dim, _wsi_dim = get_encoder_dims(args.model_type)
    dtype = DEFAULT_DTYPE
    device = resolve_device()

    model = build_downstream_model(args.arch_type, num_classes=C.NUM_CLASSES, input_dim=patch_dim)

    val_dataset = SubtypeDataset(model_type=args.model_type, arch_type=args.arch_type, split="val")
    test_dataset = SubtypeDataset(model_type=args.model_type, arch_type=args.arch_type, split="test")

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print(f"Validation on {len(val_dataset)} samples.")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print(f"Test on {len(test_dataset)} samples.")

    model = model.to(dtype).to(device)

    paths = get_paths()
    save_ckpt_base = paths.ckpt(C.LABEL_TASK, f"{args.model_type}_{args.arch_type}")
    ckpt_name = args.ckpt_name
    if ckpt_name is None:
        candidates = glob.glob(os.path.join(save_ckpt_base, "*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoints found in {save_ckpt_base}")
        ckpt_name = os.path.basename(max(candidates, key=os.path.getmtime))
    print(f"Loading checkpoint: {ckpt_name}")

    checkpoint = torch.load(os.path.join(save_ckpt_base, ckpt_name),
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    save_fig_base = os.path.join(save_ckpt_base, "figures")
    save_pd_base = os.path.join(save_ckpt_base, "save_pd")

    with torch.autocast(device_type="cuda", dtype=dtype):
        with torch.no_grad():
            model.eval()
            metrics_val = infer_once(val_loader, model, dtype, device, save_pd_base, save_fig_base, "val")
            metrics_test = infer_once(test_loader, model, dtype, device, save_pd_base, save_fig_base, "test")
            print("[Validation]", metrics_val)
            print("[Test]", metrics_test)


if __name__ == "__main__":
    main()
