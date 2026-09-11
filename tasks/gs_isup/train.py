"""Train the ordinal Gleason/ISUP grading model (CORAL).

Usage
-----
    python -m tasks.gs_isup.train --model_type andrew --arch_type final
    python -m tasks.gs_isup.train --model_type andrew \
        --arch_type [final|no_nuclei|no_prism|only_low|only_high]
    python -m tasks.gs_isup.train --model_type [uni|virchow|andrew] --arch_type only_high
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from coral_pytorch.dataset import levels_from_labelbatch, proba_to_label
from coral_pytorch.losses import coral_loss

from common import (
    CheckpointManager,
    DEFAULT_DTYPE,
    GRAD_ACCUM_STEPS,
    check_arch_invariant,
    get_encoder_dims,
    get_paths,
    quadratic_weighted_kappa,
    resolve_device,
    set_seed,
)

from . import config as C
from .dataset import build_dataset
from .model import ProstateGradingModel, ProstateGradingModelNuclei


def probability_to_logits(p):
    # avoid numerical overflow when p is near 0 or 1
    p = torch.clamp(p, 1e-7, 1 - 1e-7)
    return torch.log(p / (1 - p))


def eval_once(loader, model, dtype, device, class_num):
    """Validation pass. Eval loss uses the same scaling but WITHOUT the
    trailing ``* 0.5`` that the training total loss applies."""
    val_loss = 0
    all_pred_scores = np.zeros((len(loader)))
    all_labels = np.zeros((len(loader)))

    for batch_idx, data in tqdm(enumerate(loader)):
        patch_feature_256, patch_feature_512, instance_label_256, instance_label_512, wsi_latent, gp_1, gp_2, isup_score, patient_name = data

        patch_feature_256 = patch_feature_256.type(dtype).to(device)[0]
        patch_feature_512 = patch_feature_512.type(dtype).to(device)[0]

        instance_label_256 = instance_label_256.type(dtype).to(device)
        instance_label_512 = instance_label_512.type(dtype).to(device)

        wsi_latent = wsi_latent.type(dtype).to(device)

        gp_1 = gp_1.type(torch.LongTensor).to(device)
        gp_2 = gp_2.type(torch.LongTensor).to(device)
        isup_score = isup_score.type(torch.LongTensor).to(device)

        gp_1_levels = levels_from_labelbatch(gp_1, num_classes=C.GLEASON_NUM_CLASSES).to(device)
        gp_2_levels = levels_from_labelbatch(gp_2, num_classes=C.GLEASON_NUM_CLASSES).to(device)
        isup_levels = levels_from_labelbatch(isup_score, num_classes=class_num).to(device)

        primary_proba_ordinal, secondary_proba_ordinal, isup_proba_ordinal, _, inst_loss = model(patch_feature_256,
                                                                                                 patch_feature_512,
                                                                                                 instance_label_256,
                                                                                                 instance_label_512,
                                                                                                 wsi_latent)

        gp_1_logits = probability_to_logits(primary_proba_ordinal)
        gp_2_logits = probability_to_logits(secondary_proba_ordinal)
        isup_logits = probability_to_logits(isup_proba_ordinal)

        probas = torch.sigmoid(isup_logits)
        predicted_labels = proba_to_label(probas).float()

        all_pred_scores[batch_idx] = predicted_labels.item()
        all_labels[batch_idx] = isup_score.item()

        gp_1_loss = coral_loss(gp_1_logits, gp_1_levels)
        gp_2_loss = coral_loss(gp_2_logits, gp_2_levels)
        isup_loss = coral_loss(isup_logits, isup_levels)

        loss = 0.5 * gp_1_loss + 0.5 * gp_2_loss + isup_loss + inst_loss

        loss_value = loss.item()
        val_loss += loss_value

    weighted_kappa_val = quadratic_weighted_kappa(all_pred_scores, all_labels)
    acc_val = (all_pred_scores == all_labels).sum() / all_pred_scores.shape[0]
    epoch_val_loss = val_loss / len(loader)

    return weighted_kappa_val, acc_val, epoch_val_loss


def build_model(model_type, arch_type):
    """Instantiate the model matching (model_type, arch_type)."""
    patch_dim, _ = get_encoder_dims(model_type)
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
        return ProstateGradingModel(input_dim=patch_dim, with_high_input=True, with_low_input=False)


def main():
    parser = argparse.ArgumentParser(description="Arguments for model training.")
    parser.add_argument("--epochs", type=int, default=1000, help="training epochs")
    parser.add_argument("--few_shot_prop", type=float, default=1.0, help="use prop to train")
    parser.add_argument("--model_type", type=str, default="andrew", choices=C.MODEL_CHOICES)
    parser.add_argument("--arch_type", type=str, default="only_high", choices=C.ARCH_CHOICES)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    set_seed(args.seed)
    check_arch_invariant(args.model_type, args.arch_type)

    class_num = C.CLASS_NUM
    max_epochs = args.epochs
    gc = GRAD_ACCUM_STEPS

    device = resolve_device()
    dtype = DEFAULT_DTYPE if torch.cuda.is_available() else torch.float32

    model = build_model(args.model_type, args.arch_type)

    train_dataset = build_dataset(model_name=args.model_type, split="train", arch_type=args.arch_type)
    val_dataset = build_dataset(model_name=args.model_type, split="val", arch_type=args.arch_type)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    print(f"Train {len(train_dataset)} samples.")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print(f"Validation {len(val_dataset)} samples.")

    model = model.to(dtype).to(device)

    optimizer = optim.Adam(model.parameters(), C.LR, weight_decay=C.WEIGHT_DECAY)

    params_with_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable Parameter Number：", params_with_grad)

    save_ckpt_base = get_paths().ckpt(f"{args.model_type}_{args.arch_type}")
    # Monitor QWK (higher is better); CheckpointManager also re-saves on min val loss.
    ckpt_mgr = CheckpointManager(save_ckpt_base, higher_is_better=True)

    for epoch in range(max_epochs):
        with torch.autocast(device_type="cuda", dtype=dtype):
            train_loss = 0.
            model.train()

            all_pred_scores = np.zeros((len(train_loader)))
            all_labels = np.zeros((len(train_loader)))

            for batch_idx, data in tqdm(enumerate(train_loader)):
                patch_feature_256, patch_feature_512, instance_label_256, instance_label_512, wsi_latent, gp_1, gp_2, isup_score, patient_name = data

                if patch_feature_256.max() > 8:
                    continue
                if patch_feature_512.max() > 8:
                    continue
                patch_feature_256 = patch_feature_256.type(dtype).to(device)[0]
                patch_feature_512 = patch_feature_512.type(dtype).to(device)[0]

                instance_label_256 = instance_label_256.type(dtype).to(device)
                instance_label_512 = instance_label_512.type(dtype).to(device)

                wsi_latent = wsi_latent.type(dtype).to(device)

                gp_1 = gp_1.type(torch.LongTensor).to(device)
                gp_2 = gp_2.type(torch.LongTensor).to(device)
                isup_score = isup_score.type(torch.LongTensor).to(device)

                gp_1_levels = levels_from_labelbatch(gp_1, num_classes=C.GLEASON_NUM_CLASSES).to(device)
                gp_2_levels = levels_from_labelbatch(gp_2, num_classes=C.GLEASON_NUM_CLASSES).to(device)
                isup_levels = levels_from_labelbatch(isup_score, num_classes=class_num).to(device)

                primary_proba_ordinal, secondary_proba_ordinal, isup_proba_ordinal, _, inst_loss = model(patch_feature_256,
                                                                                                         patch_feature_512,
                                                                                                         instance_label_256,
                                                                                                         instance_label_512,
                                                                                                         wsi_latent)

                gp_1_logits = probability_to_logits(primary_proba_ordinal)
                gp_2_logits = probability_to_logits(secondary_proba_ordinal)
                isup_logits = probability_to_logits(isup_proba_ordinal)

                gp_1_loss = coral_loss(gp_1_logits, gp_1_levels)
                gp_2_loss = coral_loss(gp_2_logits, gp_2_levels)
                isup_loss = coral_loss(isup_logits, isup_levels)

                loss = (0.5 * gp_1_loss + 0.5 * gp_2_loss + isup_loss + inst_loss) * 0.5

                probas = torch.sigmoid(isup_logits)
                predicted_labels = proba_to_label(probas).float()

                all_pred_scores[batch_idx] = predicted_labels.item()
                all_labels[batch_idx] = isup_score.item()

                loss_value = loss.item()
                train_loss += loss_value
                loss = loss / gc
                loss.backward()
                if (batch_idx + 1) % gc == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            kappa_train = quadratic_weighted_kappa(all_pred_scores, all_labels)
            acc_train = (all_pred_scores == all_labels).sum() / all_pred_scores.shape[0]

            epoch_train_loss = train_loss / len(train_loader)
            print(f"[Train] Epoch: {epoch}, overall loss {epoch_train_loss:.4f}, kappa {kappa_train:.4f}, "
                  f"acc {acc_train:.4f}.")

            # validation stage
            with torch.no_grad():
                model.eval()

                kappa_val, acc_val, loss_val = eval_once(val_loader, model, dtype, device, class_num)

                ckpt_name = f"epoch_{epoch}_{kappa_val:.4f}.pth"
                # Checkpoint selection monitors only the validation QWK / validation loss.
                if ckpt_mgr.maybe_save(model, optimizer, ckpt_name, metric=kappa_val, loss=loss_val):
                    print("Save the best checkpoint!")

                print(f"[Validation] Epoch: {epoch}, overall loss {loss_val:.4f}, "
                      f"kappa {kappa_val:.4f}, acc {acc_val:.4f}.")


if __name__ == "__main__":
    main()
