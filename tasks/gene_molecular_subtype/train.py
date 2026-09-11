"""Train the 3-class molecular-subtype MIL model (basal / luminalA / luminalB).

Usage
-----
    python -m tasks.gene_molecular_subtype.train --model_type andrew --arch_type final
    python -m tasks.gene_molecular_subtype.train --model_type andrew \
        --arch_type ["final","no_nuclei","no_prism","only_low","only_high"]
    python -m tasks.gene_molecular_subtype.train --model_type ["uni","virchow","andrew"] \
        --arch_type only_high
"""
from __future__ import annotations

import argparse
import os

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
    MultiClassFocalLoss,
    build_downstream_model,
    check_arch_invariant,
    get_encoder_dims,
    get_paths,
    resolve_device,
    set_seed,
)

from . import config
from .dataset import make_dataset

# Raw cross-entropy over one-hot targets, used for loss_2 / loss_3 (and loss_1 at eval).
loss_function = lambda x_, y_: -(x_.log() * y_).sum()


def _hier_probs(logits):
    prob_1 = torch.softmax(logits, dim=1)
    prob_2 = torch.concat([prob_1[:, [0]], prob_1[:, [1]] + prob_1[:, [2]]], dim=1)
    prob_3 = torch.softmax(logits[:, 1:], dim=1)
    return prob_1, prob_2, prob_3


@torch.no_grad()
def evaluate(loader, model, dtype, device, class_num):
    """Faithful port of utils.eval_once: (acc, auc, mean_loss). Uses the lambda for loss_1."""
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
        prob_1, prob_2, prob_3 = _hier_probs(logits)

        labels_onehot = (labels == torch.tensor([0, 1, 2]).to(labels.device)).unsqueeze(0).type(dtype)

        loss_1 = loss_function(prob_1, labels_onehot)
        loss_2 = loss_function(prob_2, torch.concat([labels_onehot[:, [0]], 1 - labels_onehot[:, [0]]], dim=1))
        loss_3 = loss_function(prob_3, labels_onehot[:, 1:])

        if labels[0] == 0:
            loss = loss_1 + config.LOSS_W_BASAL_L2 * loss_2
        else:
            loss = loss_1 + loss_2 + config.LOSS_W_LUM_L3 * loss_3

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
    parser.add_argument("--epochs", type=int, default=200, help="training epochs")
    parser.add_argument("--model_type", type=str, default="andrew", choices=config.MODEL_TYPES)
    parser.add_argument("--arch_type", type=str, default="final", choices=config.ARCH_TYPES)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    set_seed(args.seed)
    check_arch_invariant(args.model_type, args.arch_type)

    patch_dim, wsi_dim = get_encoder_dims(args.model_type)

    max_epochs = args.epochs
    class_num = config.NUM_CLASSES
    gc = GRAD_ACCUM_STEPS
    patience = 10
    min_delta = 0.001
    wait = 0
    dtype = DEFAULT_DTYPE
    device = resolve_device()

    model = build_downstream_model(args.arch_type, num_classes=class_num, input_dim=patch_dim, captum=False)
    model = model.to(dtype).to(device)

    train_dataset = make_dataset(args.model_type, args.arch_type, split="train")
    val_dataset = make_dataset(args.model_type, args.arch_type, split="val")

    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print(f"Train on {len(train_dataset)} samples.")
    print(f"Validation on {len(val_dataset)} samples.")

    loss_function_focal = MultiClassFocalLoss(
        alpha=config.focal_alpha_tensor(device), gamma=config.FOCAL_GAMMA, reduction="mean")

    optimizer = optim.Adam(model.parameters(), config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    params_with_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable Parameter Number：", params_with_grad)

    save_ckpt_base = get_paths().mil(config.CKPT_TASK, "ckpt", f"{args.model_type}_{args.arch_type}")
    ckpt_manager = CheckpointManager(save_ckpt_base, higher_is_better=True)
    min_loss_val = float("inf")

    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=class_num)])

    for epoch in range(max_epochs):
        train_loss, val_loss = 0.0, 0.0
        model.train()

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
            prob_1, prob_2, prob_3 = _hier_probs(logits)

            labels_onehot = (labels == torch.tensor([0, 1, 2]).to(labels.device)).unsqueeze(0).type(dtype)

            # TRAIN: loss_1 uses the focal loss over class indices; loss_2/3 use the lambda.
            loss_1 = loss_function_focal(prob_1, torch.argmax(labels_onehot, dim=1))
            loss_2 = loss_function(prob_2, torch.concat([labels_onehot[:, [0]], 1 - labels_onehot[:, [0]]], dim=1))
            loss_3 = loss_function(prob_3, labels_onehot[:, 1:])

            if labels[0] == 0:
                loss = loss_1 + config.LOSS_W_BASAL_L2 * loss_2
            else:
                loss = loss_1 + loss_2 + config.LOSS_W_LUM_L3 * loss_3

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
        print(f"[Train] Epoch: {epoch}, overall loss {epoch_train_loss:.4f}, acc {acc_train:.4f}, "
              f"auc {auc_train:.4f}.")

        # validation stage
        model.eval()
        acc_val, auc_val, loss_val = evaluate(val_loader, model, dtype, device, class_num)

        ckpt_name = f"epoch_{epoch}_{auc_val:.4f}.pth"
        # Save on best val AUC OR min val loss; always write this epoch's
        # uniquely-named checkpoint too.
        ckpt_manager.maybe_save(model, optimizer, ckpt_name, metric=auc_val, loss=loss_val)
        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict()}
        torch.save(checkpoint, os.path.join(save_ckpt_base, ckpt_name))
        if loss_val < min_loss_val:
            min_loss_val = loss_val

        print(f"[Validation] Epoch: {epoch}, overall loss {loss_val:.4f}, "
              f"acc {acc_val:.4f}, auc {auc_val:.4f}.")

        if epoch > 1000:
            if val_loss + min_delta < min_loss_val:
                min_loss_val = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping triggered.")
                    break


if __name__ == "__main__":
    main()
