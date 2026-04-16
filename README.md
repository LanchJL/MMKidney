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


bash scripts/cluster_hierarchical_steps.sh --step fit    --manifest "$MAN" --stain-name HE --output-dir "$OUT" --n-prototypes 2048 --prototype-patches-per-sample 3000
bash scripts/cluster_hierarchical_steps.sh --step assign --manifest "$MAN" --stain-name HE --output-dir "$OUT"
bash scripts/cluster_hierarchical_steps.sh --step l1     --output-dir "$OUT" --l1-method leiden --l1-resolution 0.45
bash scripts/cluster_hierarchical_steps.sh --step l2     --output-dir "$OUT" --l1-method leiden --refine-resolution 0.60 --min-prototypes-to-refine 40 --min-l2-silhouette 0.05 --stability-n-seeds 3 --min-stability-ari 0.70 --enable-l3
bash scripts/cluster_hierarchical_steps.sh --step final  --output-dir "$OUT" --spatial-smooth-k 7 --spatial-smooth-iter 1
bash scripts/cluster_hierarchical_steps.sh --step check  --output-dir "$OUT"
