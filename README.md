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

## 3) Banff-first multitask path

The Banff-first workflow builds WSI-level Banff labels from the censored pathology
spreadsheet, reports which targets are currently trainable, and trains separate
task heads with stain-aware supervision.

```bash
bash scripts/build_banff_first_manifests.sh
```

This writes:

- `data/processed_banff/manifests/*_banff_manifest.jsonl`
- `data/processed_banff/manifests/banff_feasibility_report.json`

Core training defaults to `ci,ct,c4d,pvl,cg,g`. Sparse 0-3 targets with poor
class support are trained as binary-first (`0` vs `>0`) by default. For `ci`
and `ct`, the default input mode is `MASSON+HE` so good-quality HE can support
cases where Masson is faded or missing:

```bash
python -m src.training.train_banff \
  --train-manifest data/processed_banff/manifests/train_banff_manifest.jsonl \
  --val-manifest data/processed_banff/manifests/val_banff_manifest.jsonl \
  --test-manifest data/processed_banff/manifests/test_banff_manifest.jsonl \
  --stain-vocab data/processed/stain_vocab.json \
  --out-dir outputs/banff_first \
  --tasks ci,ct,c4d,pvl,cg,g \
  --binary-tasks auto \
  --ci-ct-stain-mode masson_he
```

Run `ci/ct` ablations to verify whether HE is helping:

```bash
python -m src.training.train_banff \
  --tasks ci,ct \
  --binary-tasks none \
  --ci-ct-stain-mode masson \
  --out-dir outputs/banff_first_ci_ct_masson

python -m src.training.train_banff \
  --tasks ci,ct \
  --binary-tasks none \
  --ci-ct-stain-mode he \
  --out-dir outputs/banff_first_ci_ct_he

python -m src.training.train_banff \
  --tasks ci,ct \
  --binary-tasks none \
  --ci-ct-stain-mode masson_he \
  --out-dir outputs/banff_first_ci_ct_masson_he
```

Recommended first-pass chronic-injury validation uses binary label modes before
returning to 0/1/2/3 ordinal scoring:

```bash
python -m src.training.train_banff \
  --tasks ci \
  --binary-tasks none \
  --label-mode zero_vs_positive \
  --ci-ct-stain-mode he \
  --out-dir outputs/banff_first_ci_zero_vs_positive_he

python -m src.training.train_banff \
  --tasks ci \
  --binary-tasks none \
  --label-mode low_vs_high \
  --ci-ct-stain-mode he \
  --out-dir outputs/banff_first_ci_low_vs_high_he

python -m src.training.train_banff \
  --tasks ct \
  --binary-tasks none \
  --label-mode low_vs_high \
  --ci-ct-stain-mode he \
  --out-dir outputs/banff_first_ct_low_vs_high_he

python -m src.training.train_banff \
  --tasks ifta \
  --binary-tasks none \
  --label-mode low_vs_high \
  --ci-ct-stain-mode he \
  --out-dir outputs/banff_first_ifta_low_vs_high_he
```

Feasibility tiers in the report:

- `ready_slide_mil`: enough WSI-aligned labels to start slide-level MIL.
- `binary_first`: train `0` vs `>0` first; multiclass needs more positives or
  stronger annotation.
- `needs_extra_annotation`: WSI labels alone are not reliable enough for the
  requested lesion score; add region/cell/tissue-compartment annotations before
  making it a primary model target.
- `exclude_now`: target is intentionally removed from the Banff-first setup.

## 4) Train WSI-only student (Stage A)

```bash
bash scripts/train_wsi.sh \
  --train-manifest data/processed/manifests/train_manifest.jsonl \
  --val-manifest data/processed/manifests/val_manifest.jsonl \
  --test-manifest data/processed/manifests/test_manifest.jsonl \
  --stain-vocab data/processed/stain_vocab.json \
  --out-dir outputs/wsi
```

## 5) Train tri-modal teacher (Stage B)

```bash
bash scripts/train_teacher.sh \
  --trimodal-manifest data/processed/manifests/trimodal_manifest.jsonl \
  --stain-vocab data/processed/stain_vocab.json \
  --tabular-features data/processed/tabular_features.csv \
  --lab-features data/processed/lab_features.csv \
  --init-wsi-ckpt outputs/wsi/best_wsi.pt \
  --out-dir outputs/teacher
```

## 6) Distill to WSI student (Stage C)

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

## 7) Inference + explanations

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

## 8) End-to-end single WSI (TRIDENT + CONCHv1.5 + MMKidney)

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
