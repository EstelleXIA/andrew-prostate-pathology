"""Stage-1 training: flat region-level ``RegionEncoder`` (NLL survival).

Run as::

    python -m tasks.bcr.train_stage_1 --model_type andrew --seed 2026

Step 1 of 4 in the BCR pipeline (see ``tasks/bcr/__init__``).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (DEFAULT_DTYPE, GRAD_ACCUM_STEPS, RegionEncoder, c_index,
                    define_loss, get_encoder_dims, get_paths, resolve_device, set_seed)
from . import config
from .dataset_stage_1 import SurvDataset


def eval_once(loader, model, dtype, device, criterion):
    val_loss = 0
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, data in tqdm(enumerate(loader)):
        patch_feature, surv_discrete, surv_time, censor, patient_name = data
        patch_feature = patch_feature.type(dtype).to(device)
        surv_discrete = surv_discrete.type(torch.LongTensor).to(device)
        censor = censor.type(dtype).to(device)

        logits, _ = model(patch_feature[0])
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        loss = criterion(hazards=hazards, S=S, Y=surv_discrete, c=censor)

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = censor.item()
        all_event_times[batch_idx] = surv_time.item()

        val_loss += loss.item()

    c_index_val = c_index(all_censorships, all_event_times, all_risk_scores)
    return c_index_val, val_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Arguments for BCR stage-1 model training.")
    parser.add_argument("--epochs", type=int, default=500, help="training epochs")
    parser.add_argument("--model_type", type=str, default="andrew", choices=config.MODEL_CHOICES)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    set_seed(args.seed)

    patch_dim, wsi_dim = get_encoder_dims(args.model_type)

    max_epochs = args.epochs
    gc = GRAD_ACCUM_STEPS
    best_c_index_val = 0
    min_loss_val = 999

    dtype = DEFAULT_DTYPE if torch.cuda.is_available() else torch.float32
    device = resolve_device()

    train_dataset = SurvDataset(model_name=args.model_type, split="train")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)

    val_loader = DataLoader(SurvDataset(model_name=args.model_type, split="val"),
                            batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = RegionEncoder(patch_dim=patch_dim, out_class_num=config.OUT_CLASS_NUM)
    model = model.to(dtype).to(device)
    optimizer = optim.Adam(model.parameters(), 5e-4, weight_decay=5e-4)

    params_with_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable Parameter Number:", params_with_grad)

    criterion = define_loss("nll_surv")
    paths = get_paths()
    save_ckpt_base = paths.mil(config.TASK_DIR, config.CKPT_STAGE1_SUBDIR, args.model_type)
    os.makedirs(save_ckpt_base, exist_ok=True)

    for epoch in range(max_epochs):
        with torch.autocast(device_type="cuda", dtype=dtype):
            train_loss = 0.
            model.train()

            all_risk_scores = np.zeros((len(train_loader)))
            all_censorships = np.zeros((len(train_loader)))
            all_event_times = np.zeros((len(train_loader)))

            for batch_idx, data in tqdm(enumerate(train_loader)):
                patch_feature, surv_discrete, surv_time, censor, patient_name = data
                patch_feature = patch_feature.type(dtype).to(device)
                surv_discrete = surv_discrete.type(torch.LongTensor).to(device)
                censor = censor.type(dtype).to(device)

                logits, _ = model(patch_feature[0])
                hazards = torch.sigmoid(logits)
                S = torch.cumprod(1 - hazards, dim=1)

                loss = criterion(hazards=hazards, S=S, Y=surv_discrete, c=censor)
                risk = -torch.sum(S, dim=1).detach().cpu().numpy()
                all_risk_scores[batch_idx] = risk
                all_censorships[batch_idx] = censor.item()
                all_event_times[batch_idx] = surv_time.item()

                train_loss += loss.item()

                loss = loss / gc
                loss.backward()
                if (batch_idx + 1) % gc == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            c_index_train = c_index(all_censorships, all_event_times, all_risk_scores)
            epoch_train_loss = train_loss / len(train_loader)
            print(f"[Train] Epoch: {epoch}, overall loss {epoch_train_loss:.4f}, c-index {c_index_train:.4f}.")

            with torch.no_grad():
                model.eval()

                c_index_val, loss_val = eval_once(val_loader, model, dtype, device, criterion)

                ckpt_name = f"epoch_{epoch}_{c_index_val:.4f}.pth"
                if c_index_val > best_c_index_val:
                    best_c_index_val = c_index_val
                    checkpoint = {"model_state_dict": model.state_dict(),
                                  "optimizer_state_dict": optimizer.state_dict()}
                    torch.save(checkpoint, os.path.join(save_ckpt_base, ckpt_name))
                    print("Save the best checkpoint!")

                if loss_val < min_loss_val:
                    min_loss_val = loss_val
                    checkpoint = {"model_state_dict": model.state_dict(),
                                  "optimizer_state_dict": optimizer.state_dict()}
                    torch.save(checkpoint, os.path.join(save_ckpt_base, ckpt_name))
                    print("Save the best checkpoint by loss!")

                print(f"[Validation] Epoch: {epoch}, overall loss {loss_val:.4f}, c-index {c_index_val:.4f}, "
                      f"best c-index {best_c_index_val:.4f}.")


if __name__ == "__main__":
    main()
