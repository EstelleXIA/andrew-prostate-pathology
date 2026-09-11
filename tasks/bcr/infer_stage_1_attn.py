"""Stage-1 inference: dump per-region attention for every cohort.

Run as::

    python -m tasks.bcr.infer_stage_1_attn --model_type andrew \
        --ckpt <checkpoint>.pth

"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (DEFAULT_DTYPE, RegionEncoder, c_index, get_encoder_dims,
                    get_paths, resolve_device, set_seed)
from . import config
from .dataset_stage_1 import SurvDataset


def save_attn_loader(loader, dtype, device, model, save_attn_base):
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    for batch_idx, data in tqdm(enumerate(loader)):
        patch_feature, _, surv_time, censor, patient_name = data
        patch_feature = patch_feature.type(dtype).to(device)
        censor = censor.type(dtype).to(device)

        logits, attn_value = model(patch_feature[0])

        np.save(os.path.join(save_attn_base, f"{patient_name[0]}.npy"),
                attn_value.detach().cpu().numpy()[0])

        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = censor.item()
        all_event_times[batch_idx] = surv_time.item()

    return c_index(all_censorships, all_event_times, all_risk_scores)


def main():
    parser = argparse.ArgumentParser(description="Arguments for BCR stage-1 attention export.")
    parser.add_argument("--model_type", type=str, default="andrew", choices=config.MODEL_CHOICES)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="stage-1 checkpoint: absolute path, or filename under "
                             "mil/<TASK_DIR>/ckpt_stage_1/<model_type>/.")
    parser.add_argument("--save_attn_dir", type=str, default=None,
                        help="where to write <patient>.npy attention; defaults to "
                             "mil/<TASK_DIR>/region_attn/<model_type>/.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    set_seed(args.seed)

    patch_dim, wsi_dim = get_encoder_dims(args.model_type)
    dtype = DEFAULT_DTYPE if torch.cuda.is_available() else torch.float32
    device = resolve_device()
    paths = get_paths()

    loaders = {}
    for split in ("train", "val", "test", "PLCO"):
        loaders[split] = DataLoader(SurvDataset(model_name=args.model_type, split=split),
                                    batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = RegionEncoder(patch_dim=patch_dim, out_class_num=config.OUT_CLASS_NUM)
    model = model.to(dtype).to(device)

    ckpt_base = paths.mil(config.TASK_DIR, config.CKPT_STAGE1_SUBDIR, args.model_type)
    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) or os.path.exists(args.ckpt) \
        else os.path.join(ckpt_base, args.ckpt)
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    save_attn_base = args.save_attn_dir or paths.mil(
        config.TASK_DIR, config.REGION_ATTN_SUBDIR, args.model_type)
    os.makedirs(save_attn_base, exist_ok=True)

    with torch.autocast(device_type="cuda", dtype=dtype), torch.inference_mode(), torch.no_grad():
        model.eval()
        results = {split: save_attn_loader(loaders[split], dtype, device, model, save_attn_base)
                   for split in ("train", "val", "test", "PLCO")}

    print(f"[Train] c-index {results['train']:.4f}.")
    print(f"[Validation] c-index {results['val']:.4f}.")
    print(f"[Test] c-index {results['test']:.4f}.")
    print(f"[PLCO] c-index {results['PLCO']:.4f}.")


if __name__ == "__main__":
    main()
