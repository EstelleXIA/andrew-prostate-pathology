"""Train the multilabel PathoSubtype classifier (PAA / PDA / IDC-P / NEPC).

Usage
-----
    python -m tasks.patho_subtype.train --model_type andrew \
        --arch_type {final,no_nuclei,no_prism,only_low,only_high}
    python -m tasks.patho_subtype.train --model_type {uni,virchow,andrew} \
        --arch_type only_high

Loss is ``common.AsymmetricLoss`` (multilabel, *10 scale, m=0.05). The monitored
metric is ``Macro_AUC_rare`` (mean AUC over PDA / IDC-P / NEPC, higher is better).
Rare-class balance comes from the WeightedRandomSampler below.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from common import (
    AsymmetricLoss,
    CheckpointManager,
    DEFAULT_DTYPE,
    GRAD_ACCUM_STEPS,
    build_downstream_model,
    calculate_metrics,
    check_arch_invariant,
    get_encoder_dims,
    get_paths,
    resolve_device,
    set_seed,
)

from . import config as C
from .dataset import SubtypeDataset


def eval_once(loader, model, dtype, device, loss_function):
    """Multilabel eval loop: sigmoid the logits before collecting probabilities."""
    val_loss = 0.0
    all_preds, all_labels = [], []
    for data in tqdm(loader):
        patch_feature_256, patch_feature_512, wsi_feature, labels, _ = data

        patch_feature_256 = patch_feature_256.type(dtype).to(device)[0]
        patch_feature_512 = patch_feature_512.type(dtype).to(device)[0]
        wsi_feature = wsi_feature.type(dtype).to(device)
        labels = labels.type(dtype).to(device)

        logits = model(patch_feature_256, patch_feature_512, wsi_feature)
        loss = loss_function(logits, labels)
        val_loss += loss.item()

        all_preds.append(torch.sigmoid(logits).cpu())
        all_labels.append(labels.cpu())

    metrics = calculate_metrics(
        torch.cat(all_labels).detach().to(torch.float).numpy(),
        torch.cat(all_preds).detach().to(torch.float).numpy(),
        class_names=C.CLASS_NAMES,
        rare_class_names=C.RARE_CLASS_NAMES,
    )
    return metrics, val_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Arguments for PathoSubtype training.")
    parser.add_argument("--epochs", type=int, default=1000, help="training epochs")
    parser.add_argument("--model_type", type=str, default="andrew",
                        choices=["uni", "virchow", "gigapath", "andrew"])
    parser.add_argument("--arch_type", type=str, default="final",
                        choices=["final", "no_nuclei", "no_prism", "only_low", "only_high"])
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    set_seed(args.seed)
    check_arch_invariant(args.model_type, args.arch_type)

    patch_dim, _wsi_dim = get_encoder_dims(args.model_type)
    dtype = DEFAULT_DTYPE
    gc = GRAD_ACCUM_STEPS
    device = resolve_device()

    model = build_downstream_model(args.arch_type, num_classes=C.NUM_CLASSES, input_dim=patch_dim)

    train_dataset = SubtypeDataset(model_type=args.model_type, arch_type=args.arch_type, split="train")
    val_dataset = SubtypeDataset(model_type=args.model_type, arch_type=args.arch_type, split="val")

    # ---- WeightedRandomSampler for rare-class balance -----------------------
    labels = train_dataset.label
    train_labels = labels.loc[labels["split"] == "train", C.LABEL_COLUMNS]
    class_counts = np.sum(train_labels, axis=0)
    class_weights = len(train_dataset) / (len(class_counts) * class_counts)
    sample_weights = [class_weights[np.where(labels)[0]].sum() for labels in train_labels.values]

    train_sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=args.num_workers,
                              sampler=train_sampler)
    print(f"Train on {len(train_dataset)} samples.")

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print(f"Validation on {len(val_dataset)} samples.")

    model = model.to(dtype).to(device)

    loss_function = AsymmetricLoss()
    optimizer = optim.Adam(model.parameters(), 5e-4, weight_decay=5e-4)

    params_with_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable Parameter Number:", params_with_grad)

    save_ckpt_base = get_paths().ckpt(C.LABEL_TASK, f"{args.model_type}_{args.arch_type}")
    ckpt_manager = CheckpointManager(save_ckpt_base, higher_is_better=True)

    for epoch in range(args.epochs):
        with torch.autocast(device_type="cuda", dtype=dtype):
            train_loss = 0.0
            model.train()
            all_preds, all_labels = [], []

            for batch_idx, data in tqdm(enumerate(train_loader)):
                patch_feature_256, patch_feature_512, wsi_feature, batch_labels, _ = data

                patch_feature_256 = patch_feature_256.type(dtype).to(device)[0]
                patch_feature_512 = patch_feature_512.type(dtype).to(device)[0]
                wsi_feature = wsi_feature.type(dtype).to(device)
                batch_labels = batch_labels.type(dtype).to(device)

                logits = model(patch_feature_256, patch_feature_512, wsi_feature)
                loss = loss_function(logits, batch_labels)
                train_loss += loss.item()

                all_preds.append(torch.sigmoid(logits).cpu())
                all_labels.append(batch_labels.cpu())

                loss = loss / gc
                loss.backward()
                if (batch_idx + 1) % gc == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            metrics = calculate_metrics(
                torch.cat(all_labels).detach().to(torch.float).numpy(),
                torch.cat(all_preds).detach().to(torch.float).numpy(),
                class_names=C.CLASS_NAMES,
                rare_class_names=C.RARE_CLASS_NAMES,
            )
            epoch_train_loss = train_loss / len(train_loader)
            print(f"[Train] Epoch: {epoch}, overall loss {epoch_train_loss:.4f}, "
                  f"auc {metrics['Macro_AUC_rare']:.4f}.")
            print(metrics)

            # ---- validation --------------------------------------------------------
            with torch.no_grad():
                model.eval()
                metrics_val, loss_val = eval_once(val_loader, model, dtype, device, loss_function)

                ckpt_name = f"epoch_{epoch}_{metrics_val['Macro_AUC_rare']:.4f}.pth"
                if ckpt_manager.maybe_save(model, optimizer, ckpt_name,
                                           metric=metrics_val['Macro_AUC_rare'], loss=loss_val):
                    print("Saved checkpoint (best val AUC and/or min val loss).")

                print(f"[Validation] Epoch: {epoch}, overall loss {loss_val:.4f}, "
                      f"auc {metrics_val['Macro_AUC_rare']:.4f}.")
                print(metrics_val)


if __name__ == "__main__":
    main()
