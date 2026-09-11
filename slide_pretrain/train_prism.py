"""Fine-tune the Paige PRISM slide encoder on prostate WSIs + reports.

PRISM is a CoCa-style slide encoder: the training objective is the sum of a
caption cross-entropy loss and a symmetric image/text contrastive loss.
Tile features are the 2560-dim ([CLS] || avg) embeddings produced by a frozen
Virchow patch encoder (see slide_pretrain/dataset.py and README section 2).

Run:
    export HF_TOKEN=...            # read from the environment
    python -m slide_pretrain.train_prism \
        --cls-feature-root /path/to/patient_features_20x_256_andrew_cls \
        --ann-json /path/to/RepGen_annot_update_tumor.json
"""
from __future__ import annotations

import argparse
import os

# hf-mirror endpoint (per project network policy); override with $HF_ENDPOINT.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

from common import get_paths, set_seed
from slide_pretrain.dataset import CLS_DIRNAME, ProstateDataset, collate_fn


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune PRISM on prostate WSIs.")
    p.add_argument("--cls-feature-root", type=str, default=None,
                   help=f"Dir of per-patient [CLS] tile features. "
                        f"Default: <repgen_tumor_root>/{CLS_DIRNAME}")
    p.add_argument("--ann-json", type=str, default=None,
                   help="Report-annotation JSON. Default: <repgen_tumor_root>/RepGen_annot_update_tumor.json")
    p.add_argument("--prism-model", type=str, default="paige-ai/Prism")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--ckpt-dir", type=str, default=None,
                   help="Where to save checkpoints. Default: <ckpt_root>/slide_prism")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    paths = get_paths()

    cls_root = args.cls_feature_root or paths.repgen_tumor(CLS_DIRNAME)
    ann_json = args.ann_json or paths.repgen_tumor("RepGen_annot_update_tumor.json")
    ckpt_dir = args.ckpt_dir or paths.ckpt("slide_prism")
    os.makedirs(ckpt_dir, exist_ok=True)

    if os.environ.get("HF_TOKEN"):
        from huggingface_hub import login
        login(os.environ["HF_TOKEN"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    model = AutoModel.from_pretrained(args.prism_model, trust_remote_code=True)
    model = model.to(dtype).to(device)
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    gc = args.grad_accum

    train_dataset = ProstateDataset(cls_feature_root=cls_root, ann_json=ann_json, split="train")
    print(f"train on {len(train_dataset)} patients (train split).")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)

    for epoch in range(args.epochs):
        train_loss = 0.
        model.train()

        for batch_idx, data in tqdm(enumerate(train_loader)):
            names, tile_features, tile_masks, reports = data

            tile_features = tile_features.type(dtype).to(device)
            tile_masks = tile_masks.type(torch.bool).to(device)
            report_ids = model.tokenize(reports).to(device)

            output = model(input_ids=report_ids, tile_embeddings=tile_features, tile_mask=tile_masks)

            # caption loss (cross entropy)
            ce = F.cross_entropy
            logits = output['logits']
            logits = rearrange(logits, 'b n c -> b c n')
            caption_loss = ce(logits[:, :, :-1], report_ids[:, 1:], ignore_index=1)

            # contrastive loss (symmetric image/text)
            contrastive_labels = torch.arange(report_ids.shape[0], device=device)
            sim = output['sim']
            contrastive_loss = (ce(sim, contrastive_labels) + ce(sim.t(), contrastive_labels)) * 0.5
            loss = caption_loss + contrastive_loss

            print(f"caption loss: {caption_loss}, contrastive loss: {contrastive_loss}")

            loss_value = loss.item()
            train_loss += loss_value
            loss = loss / gc
            loss.backward()
            if (batch_idx + 1) % gc == 0:
                optimizer.step()
                optimizer.zero_grad()

        epoch_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch} / {args.epochs}: {epoch_loss}")

        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict()}
        torch.save(checkpoint, os.path.join(ckpt_dir, f"ckpt_{epoch}_{epoch_loss:.6f}.pth"))


if __name__ == "__main__":
    main()
