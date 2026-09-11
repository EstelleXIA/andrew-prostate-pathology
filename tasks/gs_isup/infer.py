"""Inference / evaluation for the ordinal Gleason/ISUP grading model.

Usage
-----
    python -m tasks.gs_isup.infer --model_type andrew --arch_type final \
        --ckpt_name <checkpoint>.pth

Writes per-split ISUP probability CSVs + ROC figures, and prints QWK / accuracy.
ROC plotting is delegated to ``common.attribution.plot_multiclass_auc``.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

from coral_pytorch.dataset import proba_to_label

from common import (
    DEFAULT_DTYPE,
    get_paths,
    quadratic_weighted_kappa,
    resolve_device,
    set_seed,
)
from common.attribution import plot_multiclass_auc

from . import config as C
from .dataset import build_dataset
from .model import ProstateGradingModel, ProstateGradingModelNuclei


def probability_to_logits(p):
    # avoid numerical overflow when p is near 0 or 1
    p = torch.clamp(p, 1e-7, 1 - 1e-7)
    return torch.log(p / (1 - p))


def build_model(model_type, arch_type):
    if model_type == "andrew":
        if arch_type == "final":
            return ProstateGradingModelNuclei(with_prism=True)
        elif arch_type == "no_prism":
            return ProstateGradingModelNuclei(with_prism=False)
        elif arch_type == "no_nuclei":
            return ProstateGradingModel(input_dim=1280, with_high_input=True, with_low_input=True)
        elif arch_type == "only_high":
            return ProstateGradingModel(input_dim=1280, with_high_input=True, with_low_input=False)
        elif arch_type == "only_low":
            return ProstateGradingModel(input_dim=1280, with_high_input=False, with_low_input=True)
        else:
            raise ValueError
    else:
        from common import get_encoder_dims
        patch_dim, _ = get_encoder_dims(model_type)
        return ProstateGradingModel(input_dim=patch_dim, with_high_input=True, with_low_input=False)


def infer_once(loader, model, dtype, device, save_fig_base, save_pd_base, split):
    all_pred_scores = np.zeros((len(loader)))
    all_labels = np.zeros((len(loader)))
    patient_list = []
    probas_list = []
    for batch_idx, data in tqdm(enumerate(loader)):
        patch_feature_256, patch_feature_512, instance_label_256, instance_label_512, wsi_latent, gp_1, gp_2, isup_score, patient_name = data

        patient_list.append(patient_name[0])

        patch_feature_256 = patch_feature_256.type(dtype).to(device)[0]
        patch_feature_512 = patch_feature_512.type(dtype).to(device)[0]

        instance_label_256 = instance_label_256.type(dtype).to(device)
        instance_label_512 = instance_label_512.type(dtype).to(device)

        wsi_latent = wsi_latent.type(dtype).to(device)

        isup_score = isup_score.type(torch.LongTensor).to(device)

        primary_proba_ordinal, secondary_proba_ordinal, isup_proba_ordinal, isup_proba, inst_loss = model(patch_feature_256,
                                                                                                          patch_feature_512,
                                                                                                          instance_label_256,
                                                                                                          instance_label_512,
                                                                                                          wsi_latent)

        isup_logits = probability_to_logits(isup_proba_ordinal)

        probas = torch.sigmoid(isup_logits)
        predicted_labels = proba_to_label(probas).float()

        all_pred_scores[batch_idx] = predicted_labels.item()
        all_labels[batch_idx] = isup_score.item()
        probas_list.append(isup_proba.detach().cpu().float().numpy()[0])

    probas_pd = pd.DataFrame(np.stack(probas_list), index=patient_list, columns=[f"ISUP_{i}" for i in range(1, 6)])
    pred_pd = pd.DataFrame(all_pred_scores, index=patient_list, columns=["pred"])
    label_pd = pd.DataFrame(all_labels, index=patient_list, columns=["label"])

    save_pd = pd.concat([probas_pd, pred_pd, label_pd], axis=1)

    weighted_kappa_val = quadratic_weighted_kappa(all_pred_scores, all_labels)
    acc_val = (all_pred_scores == all_labels).sum() / all_pred_scores.shape[0]

    encoder = OneHotEncoder(sparse_output=False)
    all_labels_onehot = encoder.fit_transform(all_labels.reshape(-1, 1))
    os.makedirs(save_fig_base, exist_ok=True)

    final_save_pd = pd.concat([label_pd, probas_pd], axis=1)
    os.makedirs(save_pd_base, exist_ok=True)
    final_save_pd.to_csv(os.path.join(save_pd_base, f"{split}.csv"))

    plot_multiclass_auc(all_labels_onehot, np.stack(probas_list), save_fig_base, split,
                        class_names=C.ISUP_CLASS_NAMES, colors=C.AUC_COLORS, plot_micro=True)

    return weighted_kappa_val, acc_val, save_pd


def main():
    parser = argparse.ArgumentParser(description="Arguments for model inference.")
    parser.add_argument("--model_type", type=str, default="andrew", choices=C.MODEL_CHOICES)
    parser.add_argument("--arch_type", type=str, default="final", choices=C.ARCH_CHOICES)
    parser.add_argument("--ckpt_name", type=str, default=None,
                        help="checkpoint filename under ckpt(<model>_<arch>); "
                             "defaults to the newest *.pth in that dir if omitted")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    set_seed(args.seed)
    from common import check_arch_invariant
    check_arch_invariant(args.model_type, args.arch_type)

    device = resolve_device()
    dtype = DEFAULT_DTYPE if torch.cuda.is_available() else torch.float32

    model = build_model(args.model_type, args.arch_type)

    val_dataset = build_dataset(model_name=args.model_type, split="val", arch_type=args.arch_type)
    test_dataset = build_dataset(model_name=args.model_type, split="test", arch_type=args.arch_type)
    PLCO_dataset = build_dataset(model_name=args.model_type, split="PLCO", arch_type=args.arch_type)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print(f"Validation {len(val_dataset)} samples.")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print(f"Test {len(test_dataset)} samples.")
    PLCO_loader = DataLoader(PLCO_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print(f"PLCO {len(PLCO_dataset)} samples.")

    paths = get_paths()
    ckpt_base = paths.ckpt(f"{args.model_type}_{args.arch_type}")
    ckpt_name = args.ckpt_name
    if ckpt_name is None:
        candidates = glob.glob(os.path.join(ckpt_base, "*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No *.pth checkpoints found under {ckpt_base}")
        ckpt_name = os.path.basename(max(candidates, key=os.path.getmtime))
    print(f"Loading checkpoint: {ckpt_name}")

    model = model.to(dtype).to(device)
    ckpt_path = os.path.join(ckpt_base, ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"]
    model.load_state_dict(ckpt, strict=False)

    figure_base = os.path.join(ckpt_base, "figures")
    save_pd_base = os.path.join(ckpt_base, "save_pd")

    with torch.autocast(device_type="cuda", dtype=dtype):
        with torch.no_grad():
            model.eval()

            kappa_val, acc_val, val_pd = infer_once(val_loader, model, dtype, device, figure_base, save_pd_base, "val")
            kappa_test, acc_test, test_pd = infer_once(test_loader, model, dtype, device, figure_base, save_pd_base, "test")
            kappa_PLCO, acc_PLCO, PLCO_pd = infer_once(PLCO_loader, model, dtype, device, figure_base, save_pd_base, "PLCO")

            print(f"[Validation] kappa {kappa_val:.4f}, acc {acc_val:.4f}.")
            print(f"[Test] kappa {kappa_test:.4f}, acc {acc_test:.4f}.")
            print(f"[PLCO] kappa {kappa_PLCO:.4f}, acc {acc_PLCO:.4f}.")


if __name__ == "__main__":
    main()
