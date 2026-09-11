"""Convert the pathology encoder weights to release-ready formats.

Inputs:
  andrew-patch.timm.pth   a plain, unwrapped timm Virchow ViT-H/14 state_dict
                          (631M params, SwiGLUPacked MLP + LayerScale). The tile encoder.
  andrew-slide.pth        the PRISM slide-encoder checkpoint (a dict with
                          `model_state_dict` + `optimizer_state_dict`). NOT Virchow.

This tool:
  1. Rebuilds the Virchow architecture locally (arch reconstructed from explicit args
     matching `hf-hub:paige-ai/Virchow`), loads `andrew-patch.timm.pth` with
     strict=True, and re-saves it in HuggingFace/timm format (`model.safetensors` +
     `config.json`) under `--out-dir`. Load it the same way as paige-ai/Virchow:

        import timm, torch
        from timm.layers import SwiGLUPacked
        model = timm.create_model(
            "hf-hub:<out-dir>/andrew-tile", pretrained=True,
            mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,
        )

     (`mlp_layer`/`act_layer` are not JSON-serialisable, so — like the Virchow card —
     they are passed at call time, not stored in config.json.)
  2. Verifies the round-trip: reloads the saved safetensors into a fresh model and
     asserts the forward output matches the source model on a random 224x224 input.
  3. Strips `andrew-slide.pth` down to a bare `model_state_dict`.

Run:
    python -m tools.convert_andrew_weights \
        --timm-ckpt  /path/to/andrew-patch.timm.pth \
        --slide-ckpt /path/to/andrew-slide.pth \
        --out-dir    weights
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load_file

import timm
from timm.layers import SwiGLUPacked
from timm.models._hub import save_for_hf

# Architecture of `hf-hub:paige-ai/Virchow` (Virchow ViT-H/14). These are the exact
# args the in-house code passed to timm.create_model when producing the checkpoint.
BASE_MODEL = "vit_huge_patch14_224"
# JSON-serialisable args -> embedded in config.json's `model_args`, so a plain
# `create_model("hf-hub:<dir>")` reconstructs the same graph.
MODEL_ARGS = dict(
    img_size=224,
    patch_size=14,
    embed_dim=1280,
    depth=32,
    num_heads=16,
    mlp_ratio=5.3375,
    global_pool="",
    num_classes=0,
    reg_tokens=0,
    init_values=1e-5,
)
# Non-serialisable args -> must be passed by the consumer at create_model time,
# exactly as with the real paige-ai/Virchow.
NONSERIALIZABLE_ARGS = dict(mlp_layer=SwiGLUPacked, act_layer=nn.SiLU)


def build_virchow() -> nn.Module:
    return timm.create_model(
        BASE_MODEL, pretrained=False, **MODEL_ARGS, **NONSERIALIZABLE_ARGS
    )


def _unwrap_state_dict(obj) -> dict:
    """Return a bare parameter dict from either a plain state_dict or a wrapped one."""
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def convert_patch_encoder(timm_ckpt: str, out_dir: str) -> str:
    print(f"[patch] loading timm state_dict: {timm_ckpt}")
    state_dict = _unwrap_state_dict(torch.load(timm_ckpt, map_location="cpu"))

    model = build_virchow()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch — missing={missing[:5]}... "
            f"unexpected={unexpected[:5]}... (arch args are wrong)"
        )
    print(f"[patch] loaded strict; {sum(p.numel() for p in model.parameters()):,} params")

    model.eval()
    torch.manual_seed(0)
    dummy = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        ref_out = model(dummy)  # [1, 257, 1280] (tokens, global_pool="")
    print(f"[patch] reference forward output shape: {tuple(ref_out.shape)}")

    hf_dir = os.path.join(out_dir, "andrew-tile")
    os.makedirs(hf_dir, exist_ok=True)
    save_for_hf(model, hf_dir, model_args=MODEL_ARGS, safe_serialization=True)
    print(f"[patch] saved HF format -> {hf_dir}")
    print(f"[patch]   files: {sorted(os.listdir(hf_dir))}")

    # --- round-trip verification: reload safetensors into a fresh model ---
    reloaded = build_virchow()
    reloaded_sd = safe_load_file(os.path.join(hf_dir, "model.safetensors"))
    r_missing, r_unexpected = reloaded.load_state_dict(reloaded_sd, strict=False)
    if r_missing or r_unexpected:
        raise RuntimeError(f"reload mismatch: missing={r_missing} unexpected={r_unexpected}")
    reloaded.eval()
    with torch.inference_mode():
        new_out = reloaded(dummy)
    max_abs = (new_out - ref_out).abs().max().item()
    assert torch.equal(new_out, ref_out), f"round-trip output differs (max_abs={max_abs})"
    print(f"[patch] round-trip verified: outputs identical (max_abs_diff={max_abs:.2e})")

    # sanity: confirm config.json is well-formed and carries the model_args
    with open(os.path.join(hf_dir, "config.json")) as f:
        cfg = json.load(f)
    assert cfg.get("architecture") == BASE_MODEL, cfg.get("architecture")
    assert cfg.get("model_args", {}).get("mlp_ratio") == 5.3375, cfg.get("model_args")
    print(f"[patch] config.json OK (architecture={cfg['architecture']}, "
          f"num_features={cfg.get('num_features')})")
    return hf_dir


def convert_slide_encoder(slide_ckpt: str, out_dir: str) -> str:
    print(f"[slide] loading PRISM checkpoint: {slide_ckpt}")
    ckpt = torch.load(slide_ckpt, map_location="cpu")
    model_sd = _unwrap_state_dict(ckpt)
    n = sum(v.numel() for v in model_sd.values() if hasattr(v, "numel"))
    print(f"[slide] stripping to bare model_state_dict ({len(model_sd)} tensors, {n:,} params)")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "andrew-slide.model.pth")
    torch.save({"model_state_dict": model_sd}, out_path)
    print(f"[slide] saved -> {out_path} (PRISM arch, NOT Virchow; load via "
          f"AutoModel.from_pretrained('paige-ai/Prism').load_state_dict(...))")
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description="Convert andrew encoder weights for release.")
    p.add_argument("--timm-ckpt", type=str, required=True,
                   help="Path to andrew-patch.timm.pth (the timm Virchow patch encoder).")
    p.add_argument("--slide-ckpt", type=str, default=None,
                   help="Path to andrew-slide.pth (the PRISM slide encoder). Optional.")
    p.add_argument("--out-dir", type=str, default="weights",
                   help="Output directory (gitignored). Default: ./weights")
    return p.parse_args()


def main():
    args = parse_args()
    convert_patch_encoder(args.timm_ckpt, args.out_dir)
    if args.slide_ckpt:
        convert_slide_encoder(args.slide_ckpt, args.out_dir)
    print("done.")


if __name__ == "__main__":
    main()
