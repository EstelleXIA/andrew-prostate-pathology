"""Inference with a fine-tuned PRISM slide encoder: caption generation + zero-shot.

Run:
    export HF_TOKEN=...
    python -m slide_pretrain.infer --ckpt /path/to/checkpoint.pth --split val
"""
from __future__ import annotations

import argparse
import glob
import json
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from tqdm import tqdm
from transformers import AutoModel

from common import get_paths
from slide_pretrain.dataset import AVG_DIRNAME, CLS_DIRNAME


def parse_args():
    p = argparse.ArgumentParser(description="PRISM slide-encoder inference.")
    p.add_argument("--ckpt", type=str, required=True, help="Fine-tuned PRISM checkpoint (.pth).")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--cls-feature-root", type=str, default=None)
    p.add_argument("--ann-json", type=str, default=None)
    p.add_argument("--prism-model", type=str, default="paige-ai/Prism")
    return p.parse_args()


def main():
    args = parse_args()
    paths = get_paths()
    feature_base = args.cls_feature_root or paths.repgen_tumor(CLS_DIRNAME)
    ann_path = args.ann_json or paths.repgen_tumor("RepGen_annot_update_tumor.json")

    if os.environ.get("HF_TOKEN"):
        from huggingface_hub import login
        login(os.environ["HF_TOKEN"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(args.prism_model, trust_remote_code=True)
    model = model.to(device)
    model.eval()  # inference must be in eval mode

    checkpoint = torch.load(args.ckpt, map_location=device)["model_state_dict"]
    model.load_state_dict(checkpoint)

    with open(ann_path, "r") as f:
        ann = json.load(f)[args.split]
    patients = [x["id"] for x in ann]

    for patient in tqdm(patients):
        with torch.autocast('cuda', torch.float16), torch.inference_mode():
            region_files = glob.glob(os.path.join(feature_base, patient, "*.pt"))

            all_features_cls, all_features_avg = [], []
            for region in region_files:
                all_features_cls.append(torch.load(region, weights_only=True))
                all_features_avg.append(torch.load(region.replace(CLS_DIRNAME, AVG_DIRNAME),
                                                   weights_only=True))
            images_cls = torch.concat(all_features_cls, dim=0)
            images_avg = torch.concat(all_features_avg, dim=0)

            tile_embeddings = torch.concat([images_cls, images_avg], dim=1).unsqueeze(0).to(device)

            reprs = model.slide_representations(tile_embeddings)

            genned_ids = model.generate(
                key_value_states=reprs['image_latents'],
                do_sample=False,
                num_beams=5,
                num_beam_groups=1,
                use_cache=False,
            )
            genned_caption = model.untokenize(genned_ids)
            print(patient, genned_caption)

            scores = model.zero_shot(
                reprs['image_embedding'],
                neg_prompts=['The patient presents with lung cancer. Gleason score of 5 + 5 = 10.'],
                pos_prompts=['The patient presents with acinar adenocarcinoma. Gleason score of 4 + 3 = 7.'],
            )
            print(scores)


if __name__ == "__main__":
    main()
