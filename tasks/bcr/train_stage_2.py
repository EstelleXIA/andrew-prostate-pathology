"""Stage-2 training: hierarchical (region -> WSI) encoder (NLL survival).

Run as::

    python -m tasks.bcr.train_stage_2 --model_type andrew --arch_type final --seed 2026

Step 3 of 4 in the BCR pipeline (requires stage-1 attention on disk; see
``tasks/bcr/__init__``).

"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (DEFAULT_DTYPE, GRAD_ACCUM_STEPS, HierarchicalEncoder,
                    HierarchicalEncoderNuclei, c_index, check_arch_invariant,
                    define_loss, get_encoder_dims, get_paths, resolve_device, set_seed)
from . import config
from .dataset_stage_2 import SurvDataset, SurvDatasetNuclei


def eval_stage_2_once(loader, model, dtype, device, criterion):
    val_loss = 0
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, data in tqdm(enumerate(loader)):
        patch_feature, low_res_feature, wsi_feature, surv_discrete, surv_time, censor, patient_name = data
        patch_feature = patch_feature.type(dtype).to(device)
        low_res_feature = low_res_feature.type(dtype).to(device)
        wsi_feature = wsi_feature.type(dtype).to(device)
        surv_discrete = surv_discrete.type(torch.LongTensor).to(device)
        censor = censor.type(dtype).to(device)

        logits = model(patch_feature, low_res_feature[0], wsi_feature)
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


def build_model(args, patch_dim, wsi_dim):
    """Return (model, dataset_cls) for the requested (model_type, arch_type)."""
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
        # Non-andrew encoders are single-scale baselines; only arch_type=="only_high"
        # is admissible (enforced by check_arch_invariant above).
        model = HierarchicalEncoder(patch_dim=patch_dim, wsi_dim=wsi_dim, out_class_num=config.OUT_CLASS_NUM,
                                    with_high_input=True, with_low_input=False)
        dataset_cls = SurvDataset
    return model, dataset_cls


def maybe_warmstart(model, args, device):
    """`final`-arch warm-start from two partial checkpoints (both strict=False).

    Loads a nuclei/no-prism stage-2 checkpoint and a PRISM-only checkpoint, each
    with the classifier head popped, layering the second over the first. Paths are
    supplied via CLI args (``--warmstart_partial`` / ``--warmstart_prism``).
    """
    if args.arch_type != "final":
        return
    if not (args.warmstart_partial and args.warmstart_prism):
        print("[warmstart] --warmstart_partial / --warmstart_prism not both set; "
              "training `final` from scratch.")
        return

    checkpoint_partial_weights = torch.load(args.warmstart_partial, map_location=device,
                                            weights_only=False)["model_state_dict"]
    checkpoint_partial_weights.pop("classifier.weight")
    checkpoint_partial_weights.pop("classifier.bias")

    pretrained_prism_weights = torch.load(args.warmstart_prism, map_location=device,
                                          weights_only=False)["model_state_dict"]
    pretrained_prism_weights.pop("classifier.weight")
    pretrained_prism_weights.pop("classifier.bias")

    model.load_state_dict(checkpoint_partial_weights, strict=False)
    model.load_state_dict(pretrained_prism_weights, strict=False)
    print("[warmstart] loaded partial + PRISM-only weights (strict=False, classifier head reset).")


def main():
    parser = argparse.ArgumentParser(description="Arguments for BCR stage-2 model training.")
    parser.add_argument("--epochs", type=int, default=500, help="training epochs")
    parser.add_argument("--model_type", type=str, default="andrew", choices=config.MODEL_CHOICES)
    parser.add_argument("--arch_type", type=str, default="final", choices=config.ARCH_CHOICES)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--warmstart_partial", type=str, default=None,
                        help="(final only) partial stage-2 checkpoint, e.g. the andrew_no_prism best ckpt.")
    parser.add_argument("--warmstart_prism", type=str, default=None,
                        help="(final only) PRISM-only pretrained checkpoint.")
    args = parser.parse_args()

    check_arch_invariant(args.model_type, args.arch_type)
    set_seed(args.seed)

    patch_dim, wsi_dim = get_encoder_dims(args.model_type)

    max_epochs = args.epochs
    gc = GRAD_ACCUM_STEPS
    best_c_index_val = 0
    min_loss_val = 999

    dtype = DEFAULT_DTYPE if torch.cuda.is_available() else torch.float32
    device = resolve_device()

    model, dataset_cls = build_model(args, patch_dim, wsi_dim)

    # Datasets are built once; the train split is used exactly as labelled.
    train_dataset = dataset_cls(model_name=args.model_type, split="train")
    val_dataset = dataset_cls(model_name=args.model_type, split="val")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = model.to(dtype).to(device)
    maybe_warmstart(model, args, device)

    optimizer = optim.Adam(model.parameters(), 5e-4, weight_decay=5e-4)

    params_with_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable Parameter Number:", params_with_grad)

    criterion = define_loss("nll_surv")
    paths = get_paths()
    save_ckpt_base = paths.mil(config.TASK_DIR, config.CKPT_STAGE2_SUBDIR,
                               f"{args.model_type}_{args.arch_type}")
    os.makedirs(save_ckpt_base, exist_ok=True)

    for epoch in range(max_epochs):
        with torch.autocast(device_type="cuda", dtype=dtype):
            train_loss = 0.
            model.train()

            all_risk_scores = np.zeros((len(train_loader)))
            all_censorships = np.zeros((len(train_loader)))
            all_event_times = np.zeros((len(train_loader)))

            for batch_idx, data in tqdm(enumerate(train_loader)):
                patch_feature, low_res_feature, wsi_feature, surv_discrete, surv_time, censor, patient_name = data
                patch_feature = patch_feature.type(dtype).to(device)
                low_res_feature = low_res_feature.type(dtype).to(device)
                wsi_feature = wsi_feature.type(dtype).to(device)
                surv_discrete = surv_discrete.type(torch.LongTensor).to(device)
                censor = censor.type(dtype).to(device)

                logits = model(patch_feature, low_res_feature[0], wsi_feature)
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

                c_index_val, loss_val = eval_stage_2_once(val_loader, model, dtype, device, criterion)

                ckpt_name = f"epoch_{epoch}_{c_index_val:.4f}.pth"
                checkpoint = {"model_state_dict": model.state_dict(),
                              "optimizer_state_dict": optimizer.state_dict()}
                if c_index_val > best_c_index_val:
                    best_c_index_val = c_index_val
                    torch.save(checkpoint, os.path.join(save_ckpt_base, ckpt_name))
                    print("Save the best checkpoint!")

                if loss_val < min_loss_val:
                    min_loss_val = loss_val
                    torch.save(checkpoint, os.path.join(save_ckpt_base, ckpt_name))
                    print("Save the best checkpoint by loss!")

                # Also re-save the latest epoch under the same name.
                torch.save(checkpoint, os.path.join(save_ckpt_base, ckpt_name))

                print(f"[Validation] Epoch: {epoch}, overall loss {loss_val:.4f}, c-index {c_index_val:.4f}, "
                      f"best c-index {best_c_index_val:.4f}.")


if __name__ == "__main__":
    main()
