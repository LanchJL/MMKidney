# MMKidney (Multimodal, src-only)

This repository now uses a single code path under `src/` for multimodal kidney pathology modeling.

- WSI-only student (Stage A)
- Tri-modal teacher (Stage B)
- Distillation teacher -> student (Stage C)

Detailed documentation: see [README_multimodal.md](README_multimodal.md).

## 1) Environment

Current runtime requires at least:

- Python 3.10+
- `torch`
- `h5py`
- `tqdm`
- `scikit-learn`

Minimal install example:

```bash
python3 -m pip install torch h5py tqdm scikit-learn
```

Optional (for parquet export):

```bash
python3 -m pip install pandas pyarrow
```

Or directly:

```bash
bash scripts/install_deps.sh
```

### TRIDENT environment (recommended for end-to-end WSI)

For `Section 7` (TRIDENT + CONCHv1.5), use a dedicated conda env:

```bash
conda env create -f envs/trident_environment.yml
conda activate trident
pip install -e external/TRIDENT
```

If the environment already exists, update it with:

```bash
conda env update -n trident -f envs/trident_environment.yml --prune
conda activate trident
pip install -e external/TRIDENT
```

To re-export current env after changes:

```bash
conda env export -n trident | sed '/^prefix:/d' > envs/trident_environment.yml
```

## Quick Start (One Command Per Stage)

```bash
bash scripts/run_pipeline.sh prepare
bash scripts/run_pipeline.sh stage_a
bash scripts/run_pipeline.sh stage_b
bash scripts/run_pipeline.sh stage_c
bash scripts/run_pipeline.sh infer
```

All-in-one:

```bash
bash scripts/run_pipeline.sh all
```

## 2) Build manifests and features

```bash
bash scripts/build_all_manifests.sh
```

This creates processed assets in `data/processed/`, including:

- `stain_vocab.json`
- `cohort_wsi.csv`
- `cohort_tabular.csv`
- `cohort_trimodal.csv`
- `tabular_features.csv`
- `lab_features.csv`
- `manifests/*.jsonl`

## 3) Train WSI-only student (Stage A)

```bash
bash scripts/train_wsi.sh \
  --train-manifest data/processed/manifests/train_manifest.jsonl \
  --val-manifest data/processed/manifests/val_manifest.jsonl \
  --test-manifest data/processed/manifests/test_manifest.jsonl \
  --stain-vocab data/processed/stain_vocab.json \
  --out-dir outputs/wsi
```

## 4) Train tri-modal teacher (Stage B)

```bash
bash scripts/train_teacher.sh \
  --trimodal-manifest data/processed/manifests/trimodal_manifest.jsonl \
  --stain-vocab data/processed/stain_vocab.json \
  --tabular-features data/processed/tabular_features.csv \
  --lab-features data/processed/lab_features.csv \
  --init-wsi-ckpt outputs/wsi/best_wsi.pt \
  --out-dir outputs/teacher
```

## 5) Distill to WSI student (Stage C)

```bash
bash scripts/train_distill.sh \
  --teacher-ckpt outputs/teacher/best_teacher.pt \
  --student-init-ckpt outputs/wsi/best_wsi.pt \
  --train-manifest data/processed/manifests/train_manifest.jsonl \
  --val-manifest data/processed/manifests/val_manifest.jsonl \
  --test-manifest data/processed/manifests/test_manifest.jsonl \
  --trimodal-manifest data/processed/manifests/trimodal_manifest.jsonl \
  --stain-vocab data/processed/stain_vocab.json \
  --tabular-features data/processed/tabular_features.csv \
  --lab-features data/processed/lab_features.csv \
  --out-dir outputs/distill
```

## 6) Inference + explanations

WSI student:

```bash
bash scripts/infer.sh \
  --model-type wsi \
  --manifest data/processed/manifests/test_manifest.jsonl \
  --stain-vocab data/processed/stain_vocab.json \
  --ckpt outputs/distill/best_distilled_student.pt \
  --out-dir outputs/explanations
```

Teacher:

```bash
bash scripts/infer.sh \
  --model-type teacher \
  --manifest data/processed/manifests/trimodal_manifest.jsonl \
  --stain-vocab data/processed/stain_vocab.json \
  --tabular-features data/processed/tabular_features.csv \
  --lab-features data/processed/lab_features.csv \
  --ckpt outputs/teacher/best_teacher.pt \
  --out-dir outputs/explanations_teacher
```

## 7) End-to-end single WSI (TRIDENT + CONCHv1.5 + MMKidney)

This repo now includes a single-slide end-to-end inference entry:
- tissue segmentation
- patch coordinate extraction
- CONCHv1.5 patch feature extraction
- MMKidney WSI model prediction

One command:

```bash
bash scripts/infer_single_wsi_end2end.sh \
  --slide-path /abs/path/to/your_HE_slide.svs \
  --stain-name HE \
  --ckpt outputs/distill/best_distilled_student.pt \
  --stain-vocab data/processed/stain_vocab.json \
  --trident-dir external/TRIDENT \
  --trident-job-dir outputs/trident_single_wsi \
  --out-dir outputs/end2end_single_wsi \
  --gpu 0 \
  --segmenter hest \
  --patch-encoder conch_v15 \
  --mag 20 \
  --patch-size 512 \
  --overlap 0
```

Notes:
- TRIDENT source is placed under `external/TRIDENT` and used as-is.
- You can directly use the exported env file at `envs/trident_environment.yml`.
- To force TRIDENT part to run in that env, pass `--trident-python /home/a6000/anaconda3/envs/trident/bin/python`.
- If features are already extracted, add `--skip-trident` to only run MMKidney inference from existing `--trident-job-dir`.
