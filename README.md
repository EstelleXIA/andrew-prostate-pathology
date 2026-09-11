# andrew-pathology

Prostate pathology whole-slide-image (WSI) modelling with multiple-instance learning (MIL). The repo contains tile-encoder / slide-encoder and five downstream MIL tasks: biochemical-recurrence survival, gene-mutation classification, molecular subtype, pathological subtype, and Gleason/ISUP grading.

## Repository layout

```
common/                    shared package: layers / models / losses / metrics / engine / attribution / config
slide_pretrain/            PRISM slide-encoder fine-tuning (dataset / train / infer)
tasks/
  bcr/                     biochemical-recurrence survival (two-stage hierarchical MIL)
  gene_expression/         binary gene-mutation classification (Wild / Mutation)
  gene_molecular_subtype/  3-class molecular subtype (Basal / LumA / LumB)
  patho_subtype/           4-class pathological subtype (PAA / PDA / IDC-P / NEPC, multilabel)
  gs_isup/                 Gleason-pattern + ISUP grading (ordinal CORAL)
tools/                     convert_andrew_weights.py (encoder weight conversion)
```

## Setup

```bash
pip install -e .
```

## Models

Two encoders produce the pre-extracted features the tasks consume:

- **andrew-tile** — the patch/tile encoder, a ViT-H/14 trained with the
  [DINOv2](https://github.com/facebookresearch/dinov2) self-supervised framework on
  prostate pathology tiles. Each tile is turned into a `concat([CLS, mean(patch tokens)])`
  embedding.
- **andrew-slide** — the slide encoder ([PRISM](https://huggingface.co/paige-ai/Prism),
  CoCa-style), fine-tuned on prostate WSIs + reports (see `slide_pretrain/`), which
  aggregates the tile embeddings into one slide-level representation.

 Pretrained weights can be accessed on the Hugging Face Hub 🤗: [**EstelleXIA/andrew**](https://huggingface.co/EstelleXIA/andrew) (`andrew-tile/` + `andrew-slide/`).

`tools/convert_andrew_weights.py` converts the raw encoder checkpoints to release format.

## Running the tasks

Shared downstream arguments: `--model_type ∈ {uni, virchow, gigapath, andrew}`,
`--arch_type ∈ {final, no_nuclei, no_prism, only_low, only_high}` (any `arch_type`
other than `only_high` requires `model_type=andrew`), `--seed` (default 2026),
`--num_workers` (default 16). Each `train.py` docstring documents the task's loss and
monitored metric.

**Slide-encoder pretraining**

```bash
python -m slide_pretrain.train_prism --epochs 100 --batch-size 16 --lr 5e-4 --grad-accum 4
python -m slide_pretrain.infer --ckpt <checkpoint>.pth --split val
```

**patho_subtype** (multilabel)

```bash
python -m tasks.patho_subtype.train --model_type andrew --arch_type final --epochs 1000
python -m tasks.patho_subtype.infer --model_type andrew --arch_type final --ckpt_name <checkpoint>.pth
```

**gene_expression** (binary; `--gene_name ∈ {FOXA1, ETS, HRR_ANY, HRR_3genes}`)

```bash
python -m tasks.gene_expression.train --model_type andrew --gene_name ETS --arch_type final --epochs 500
python -m tasks.gene_expression.infer --model_type andrew --gene_name ETS --arch_type final --ckpt_name <checkpoint>.pth
```

**gene_molecular_subtype** (3-class)

```bash
python -m tasks.gene_molecular_subtype.train --model_type andrew --arch_type final --epochs 200
python -m tasks.gene_molecular_subtype.infer --model_type andrew --arch_type final --ckpt_name <checkpoint>.pth
```

**gs_isup** (ordinal CORAL)

```bash
python -m tasks.gs_isup.train --model_type andrew --arch_type final --epochs 1000
python -m tasks.gs_isup.infer --model_type andrew --arch_type final --ckpt_name <checkpoint>.pth
```

**bcr** (survival) — two stages, run in order:

```bash
# 1. train stage 1 (region-level encoder)
python -m tasks.bcr.train_stage_1 --model_type andrew --epochs 500
# 2. export stage-1 attention to disk (stage 2 reads it)
python -m tasks.bcr.infer_stage_1_attn --model_type andrew --ckpt <stage1_checkpoint>.pth --save_attn_dir <attn_out_dir>
# 3. train stage 2 (region -> WSI hierarchical encoder)
python -m tasks.bcr.train_stage_2 --model_type andrew --arch_type final --epochs 500
# 4. evaluate stage 2 (C-index) and export a risk-score CSV
python -m tasks.bcr.infer_stage_2 --model_type andrew --arch_type final --ckpt <stage2_checkpoint>.pth --out_csv <risk>.csv
```

## License

Released under the [Apache License 2.0](LICENSE).
