"""Stage-2 inference: score the trained model and dump per-patient risk CSVs.

Run as::

    python -m tasks.bcr.infer_stage_2 --model_type andrew --arch_type final \
        --ckpt <checkpoint>.pth

"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (DEFAULT_DTYPE, HierarchicalEncoder, HierarchicalEncoderNuclei,
                    c_index, check_arch_invariant, get_encoder_dims, get_paths,
                    resolve_device, set_seed)
from . import config
from .dataset_stage_2 import SurvDataset, SurvDatasetNuclei


def eval_stage_2_once(loader, model, dtype, device):
    """Evaluate a loader; returns (c_index, per-patient risk DataFrame)."""
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    all_patient_names = []
    for batch_idx, data in tqdm(enumerate(loader)):
        patch_feature, low_res_feature, wsi_feature, surv_discrete, surv_time, censor, patient_name = data
        patch_feature = patch_feature.type(dtype).to(device)
        low_res_feature = low_res_feature.type(dtype).to(device)
        wsi_feature = wsi_feature.type(dtype).to(device)
        censor = censor.type(dtype).to(device)

        logits = model(patch_feature, low_res_feature[0], wsi_feature)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = censor.item()
        all_event_times[batch_idx] = surv_time.item()
        all_patient_names.append(patient_name[0])

    c_index_val = c_index(all_censorships, all_event_times, all_risk_scores)
    save_pd = pd.DataFrame(
        {"time": all_event_times, "event": 1 - all_censorships, "risk": all_risk_scores},
        index=all_patient_names)
    return c_index_val, save_pd


def build_model(args, patch_dim, wsi_dim):
    if args.model_type == "andrew":
        if args.arch_type == "final":
            model = HierarchicalEncoderNuclei(patch_dim=patch_dim, wsi_dim=wsi_dim,
                                              out_class_num=config.OUT_CLASS_NUM)
            dataset_cls = SurvDatasetNuclei
        elif args.arch_type == "no_prism":
            model = HierarchicalEncoderNuclei(patch_dim=patch_dim, wsi_dim=wsi_dim,
                                              out_class_num=config.OUT_CLASS_NUM, with_prism=False)
            dataset_cls = SurvDatasetNuclei
        elif args.arch_type == "no_nuclei":
            model = HierarchicalEncoder(patch_dim=patch_dim, wsi_dim=wsi_dim, out_class_num=config.OUT_CLASS_NUM,
                                        with_high_input=True, with_low_input=True)
            dataset_cls = SurvDataset
        elif args.arch_type == "only_high":
            model = HierarchicalEncoder(patch_dim=patch_dim, wsi_dim=wsi_dim, out_class_num=config.OUT_CLASS_NUM,
                                        with_high_input=True, with_low_input=False)
            dataset_cls = SurvDataset
        elif args.arch_type == "only_low":
            model = HierarchicalEncoder(patch_dim=patch_dim, wsi_dim=wsi_dim, out_class_num=config.OUT_CLASS_NUM,
                                        with_high_input=False, with_low_input=True)
            dataset_cls = SurvDataset
        else:
            raise ValueError
    else:
        model = HierarchicalEncoder(patch_dim=patch_dim, wsi_dim=wsi_dim, out_class_num=config.OUT_CLASS_NUM,
                                    with_high_input=True, with_low_input=False)
        dataset_cls = SurvDataset
    return model, dataset_cls


def main():
    parser = argparse.ArgumentParser(description="Arguments for BCR stage-2 inference.")
    parser.add_argument("--model_type", type=str, default="andrew", choices=config.MODEL_CHOICES)
    parser.add_argument("--arch_type", type=str, default="final", choices=config.ARCH_CHOICES)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="stage-2 checkpoint: absolute path, or filename under "
                             "mil/<TASK_DIR>/ckpt_stage_2/<model_type>_<arch_type>/.")
    parser.add_argument("--out_csv", type=str, default=None,
                        help="output risk CSV path; defaults to "
                             "patients_cache/<TASK_DIR>/risk_pd/save_risk_pd_<model>_<arch>.csv.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    check_arch_invariant(args.model_type, args.arch_type)
    set_seed(args.seed)

    patch_dim, wsi_dim = get_encoder_dims(args.model_type)
    dtype = DEFAULT_DTYPE if torch.cuda.is_available() else torch.float32
    device = resolve_device()
    paths = get_paths()

    model, dataset_cls = build_model(args, patch_dim, wsi_dim)

    val_dataset = dataset_cls(model_name=args.model_type, split="val")
    test_dataset = dataset_cls(model_name=args.model_type, split="test", infer=True)
    PLCO_dataset = dataset_cls(model_name=args.model_type, split="PLCO")

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    PLCO_loader = DataLoader(PLCO_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    ckpt_base = paths.mil(config.TASK_DIR, config.CKPT_STAGE2_SUBDIR,
                          f"{args.model_type}_{args.arch_type}")
    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) or os.path.exists(args.ckpt) \
        else os.path.join(ckpt_base, args.ckpt)

    model = model.to(dtype).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"]
    model.load_state_dict(ckpt)

    with torch.autocast(device_type="cuda", dtype=dtype), torch.no_grad():
        model.eval()

        c_index_val, save_pd_val = eval_stage_2_once(val_loader, model, dtype, device)
        c_index_test, save_pd_test = eval_stage_2_once(test_loader, model, dtype, device)
        c_index_PLCO, save_pd_PLCO = eval_stage_2_once(PLCO_loader, model, dtype, device)

        save_pd_val["split"] = "val"
        save_pd_test["split"] = "test"
        save_pd_PLCO["split"] = "PLCO"
        save_pd = pd.concat([save_pd_val, save_pd_test, save_pd_PLCO], axis=0)

        out_csv = args.out_csv or paths.patients_cache(
            config.TASK_DIR, "risk_pd", f"save_risk_pd_{args.model_type}_{args.arch_type}.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        save_pd.to_csv(out_csv)

        print(f"[Validation] c-index {c_index_val:.4f}.")
        print(f"[Test] c-index {c_index_test:.4f}.")
        print(f"[PLCO] c-index {c_index_PLCO:.4f}.")
        print(f"Saved risk table -> {out_csv}")


if __name__ == "__main__":
    main()
