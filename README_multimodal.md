# MMKidney Multimodal Pipeline

This implementation adds a three-stage multimodal system:

1. Stage A: WSI-only student (`train_wsi.py`), supports multi-stain missingness with HE anchor.
2. Stage B: Tri-modal teacher (`train_teacher.py`) using WSI + tabular + pre-anchor lab summary features.
3. Stage C: Distillation (`train_distill.py`) from teacher back to WSI-only student while still training on all WSI samples.

## Key guarantees

- HE-only samples can run end-to-end.
- Missing non-HE stains do not crash forward.
- Lab features are pre-anchor only by default.
- Teacher uses bridge cohort; student can train on full WSI cohort.

## Prepare manifests and features

```bash
python -m src.datasets.build_cohort
python -m src.datasets.build_tabular_features
python -m src.datasets.build_lab_features
```

Outputs go to `data/processed/`:

- `stain_vocab.json`
- `cohort_wsi.csv`
- `cohort_tabular.csv`
- `cohort_trimodal.csv`
- `tabular_features.csv` (+ optional parquet)
- `lab_features.csv` (+ optional parquet)
- manifests in `data/processed/manifests/*.jsonl`

## Train WSI-only

```bash
python -m src.training.train_wsi \
  --train-manifest data/processed/manifests/train_manifest.jsonl \
  --val-manifest data/processed/manifests/val_manifest.jsonl \
  --test-manifest data/processed/manifests/test_manifest.jsonl \
  --stain-vocab data/processed/stain_vocab.json \
  --out-dir outputs/wsi
```

## Train teacher

```bash
python -m src.training.train_teacher \
  --trimodal-manifest data/processed/manifests/trimodal_manifest.jsonl \
  --stain-vocab data/processed/stain_vocab.json \
  --tabular-features data/processed/tabular_features.csv \
  --lab-features data/processed/lab_features.csv \
  --init-wsi-ckpt outputs/wsi/best_wsi.pt \
  --out-dir outputs/teacher
```

## Distill teacher -> student

```bash
python -m src.training.train_distill \
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

## Inference and explanations

WSI student inference:

```bash
python -m src.training.infer \
  --model-type wsi \
  --manifest data/processed/manifests/test_manifest.jsonl \
  --stain-vocab data/processed/stain_vocab.json \
  --ckpt outputs/distill/best_distilled_student.pt \
  --out-dir outputs/explanations
```

Teacher inference with modality gates:

```bash
python -m src.training.infer \
  --model-type teacher \
  --manifest data/processed/manifests/trimodal_manifest.jsonl \
  --stain-vocab data/processed/stain_vocab.json \
  --tabular-features data/processed/tabular_features.csv \
  --lab-features data/processed/lab_features.csv \
  --ckpt outputs/teacher/best_teacher.pt \
  --out-dir outputs/explanations_teacher
```

Each sample produces `outputs/explanations/{sample_id}.json` containing:

- `stain_patch_attention_topk`
- `stain_fusion_weights`
- `modality_gates` (teacher)
- hierarchical probabilities

## Shell wrappers

- `scripts/build_all_manifests.sh`
- `scripts/train_wsi.sh`
- `scripts/train_teacher.sh`
- `scripts/train_distill.sh`
- `scripts/infer.sh`
