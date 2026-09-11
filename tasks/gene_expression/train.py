"""Train the binary gene-mutation classifier (2 classes: Wild / Mutation).

CrossEntropy over the 2-class logits, monai ``ROCAUCMetric`` with softmax post-pred
+ one-hot(2) post-label, accuracy via argmax, Adam(5e-4, wd 5e-4), gc=16 gradient
accumulation, bf16 autocast, batch_size=1. Monitors validation AUC (higher is better).

Usage
-----
    python -m tasks.gene_expression.train --model_type andrew --gene_name ETS   --arch_type final
    python -m tasks.gene_expression.train --model_type andrew --gene_name FOXA1 --arch_type no_nuclei
    python -m tasks.gene_expression.train --model_type uni    --gene_name ETS   --arch_type only_high
"""
from __future__ import annotations

import argparse

import torch
import torch.optim as optim
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
from monai.transforms import Activations, AsDiscrete, Compose
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    GRAD_ACCUM_STEPS,
    DEFAULT_DTYPE,
    CheckpointManager,
    build_downstream_model,
    check_arch_invariant,
    get_encoder_dims,
    get_paths,
    resolve_device,
    set_seed,
)

from . import config as C
from .dataset import GeneBinaryDataset


def eval_once(loader, model, dtype, device, class_num, loss_function):
    """Run the model over ``loader`` and return (acc, auc, mean_loss)."""
    val_loss = 0.0
    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=class_num)])
    y_pred = torch.tensor([], dtype=torch.float32, device=device)
    y = torch.tensor([], dtype=torch.long, device=device)
    auc_metric = ROCAUCMetric()

    for batch_idx, data in tqdm(enumerate(loader)):
        patch_feature_256, patch_feature_512, wsi_feature, labels, patient_name = data
        patch_feature_256 = patch_feature_256.type(dtype).to(device)[0]
        patch_feature_512 = patch_feature_512.type(dtype).to(device)[0]
        wsi_feature = wsi_feature.type(dtype).to(device)
        labels = labels.type(torch.LongTensor).to(device)

        logits = model(patch_feature_256, patch_feature_512, wsi_feature)
        loss = loss_function(logits, labels)

        y_pred = torch.cat([y_pred, logits], dim=0)
        y = torch.cat([y, labels], dim=0)
        val_loss += loss.item()

    y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
    y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]

    acc_value = torch.eq(y_pred.argmax(dim=1), y)
    acc = acc_value.sum().item() / len(acc_value)
    auc_metric(y_pred_act, y_onehot)
    auc = auc_metric.aggregate()
    auc_metric.reset()
    return acc, auc, val_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Arguments for model training.")
    parser.add_argument("--epochs", type=int, default=500, help="training epochs")
    parser.add_argument("--model_type", type=str, default="andrew", choices=list(C.MODEL_TYPES))
    parser.add_argument("--arch_type", type=str, default="final", choices=list(C.ARCH_TYPES))
    parser.add_argument("--gene_name", type=str, default="HRR_ANY", choices=list(C.GENE_NAMES))
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    check_arch_invariant(args.model_type, args.arch_type)

    patch_dim, wsi_dim = get_encoder_dims(args.model_type)

    max_epochs = args.epochs
    class_num = C.NUM_CLASSES
    gc = GRAD_ACCUM_STEPS
    patience = 10
    min_delta = 0.001
    wait = 0
    dtype = DEFAULT_DTYPE
    gene_name = args.gene_name
    device = resolve_device(args.device)

    model = build_downstream_model(args.arch_type, num_classes=class_num, input_dim=patch_dim)
    model = model.to(dtype).to(device)

    # Datasets are built once; no per-epoch rebuild.
    train_dataset = GeneBinaryDataset(model_name=args.model_type, split="train",
                                      gene_name=gene_name, arch_type=args.arch_type)
    val_dataset = GeneBinaryDataset(model_name=args.model_type, split="val",
                                    gene_name=gene_name, arch_type=args.arch_type)

    # Train loader uses neither a sampler nor shuffle (default shuffle=False).
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print(f"Train on {len(train_dataset)} samples.")
    print(f"Validation on {len(val_dataset)} samples.")

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 5e-4, weight_decay=5e-4)

    params_with_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable Parameter Number:", params_with_grad)

    save_ckpt_base = get_paths().ckpt(C.MIL_SUBDIR, args.gene_name,
                                      f"{args.model_type}_{args.arch_type}")
    ckpt_mgr = CheckpointManager(save_ckpt_base, higher_is_better=True)
    min_loss_val = float("inf")

    for epoch in range(max_epochs):
        with torch.autocast(device_type="cuda", dtype=dtype):
            train_loss = 0.0
            model.train()

            post_pred = Compose([Activations(softmax=True)])
            post_label = Compose([AsDiscrete(to_onehot=class_num)])
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            auc_metric = ROCAUCMetric()

            for batch_idx, data in tqdm(enumerate(train_loader)):
                patch_feature_256, patch_feature_512, wsi_feature, labels, patient_name = data
                patch_feature_256 = patch_feature_256.type(dtype).to(device)[0]
                patch_feature_512 = patch_feature_512.type(dtype).to(device)[0]
                wsi_feature = wsi_feature.type(dtype).to(device)
                labels = labels.type(torch.LongTensor).to(device)

                logits = model(patch_feature_256, patch_feature_512, wsi_feature)
                loss = loss_function(logits, labels)

                train_loss += loss.item()
                y_pred = torch.cat([y_pred, logits], dim=0)
                y = torch.cat([y, labels], dim=0)

                loss = loss / gc
                loss.backward()
                if (batch_idx + 1) % gc == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
            y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]

            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_train = acc_value.sum().item() / len(acc_value)
            auc_metric(y_pred_act, y_onehot)
            auc_train = auc_metric.aggregate()
            auc_metric.reset()

            epoch_train_loss = train_loss / len(train_loader)
            print(f"[Train] Epoch: {epoch}, overall loss {epoch_train_loss:.4f}, "
                  f"acc {acc_train:.4f}, auc {auc_train:.4f}.")

            with torch.no_grad():
                model.eval()
                acc_val, auc_val, loss_val = eval_once(val_loader, model, dtype, device,
                                                       class_num, loss_function)

                ckpt_name = f"epoch_{epoch}_{auc_val:.4f}.pth"
                if ckpt_mgr.maybe_save(model, optimizer, ckpt_name, metric=auc_val, loss=loss_val):
                    print("Save the best checkpoint (val AUC improved or val loss dropped)!")

                print(f"[Validation] Epoch: {epoch}, overall loss {loss_val:.4f}, "
                      f"acc {acc_val:.4f}, auc {auc_val:.4f}.")


if __name__ == "__main__":
    main()
