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
python3 -m src.datasets.build_cohort
python3 -m src.datasets.build_tabular_features
python3 -m src.datasets.build_lab_features
```

Or use one command:

```bash
bash scripts/run_pipeline.sh prepare
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
python3 -m src.training.train_wsi \
  --train-manifest data/processed/manifests/train_manifest.jsonl \
  --val-manifest data/processed/manifests/val_manifest.jsonl \
  --test-manifest data/processed/manifests/test_manifest.jsonl \
  --stain-vocab data/processed/stain_vocab.json \
  --out-dir outputs/wsi
```

## Train teacher

```bash
python3 -m src.training.train_teacher \
  --trimodal-manifest data/processed/manifests/trimodal_manifest.jsonl \
  --stain-vocab data/processed/stain_vocab.json \
  --tabular-features data/processed/tabular_features.csv \
  --lab-features data/processed/lab_features.csv \
  --init-wsi-ckpt outputs/wsi/best_wsi.pt \
  --out-dir outputs/teacher
```

## Distill teacher -> student

```bash
python3 -m src.training.train_distill \
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
python3 -m src.training.infer \
  --model-type wsi \
  --manifest data/processed/manifests/test_manifest.jsonl \
  --stain-vocab data/processed/stain_vocab.json \
  --ckpt outputs/distill/best_distilled_student.pt \
  --out-dir outputs/explanations
```

Teacher inference with modality gates:

```bash
python3 -m src.training.infer \
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

- `scripts/install_deps.sh`
- `scripts/run_pipeline.sh` (`prepare|stage_a|stage_b|stage_c|infer|all`)
- `scripts/build_pseudo_patch_fusion.sh` (HE anchor + optional multi-stain pseudo patch fusion)
- `scripts/overlay_hier_clusters_wsi.sh` (map final hierarchical cluster labels back onto WSI thumbnails)
- `scripts/build_all_manifests.sh`
- `scripts/train_wsi.sh`
- `scripts/train_teacher.sh`
- `scripts/train_distill.sh`
- `scripts/infer.sh`

## Optional: HE vs Multi-stain Pseudo Patch Fusion (for clustering comparison)

Build HE-anchored pseudo fused patch features (minimal-intrusion, no model retraining required):

```bash
bash scripts/build_pseudo_patch_fusion.sh \
  --manifest data/processed/manifests/trimodal_manifest.jsonl \
  --output-dir data/processed/pseudo_patch_fusion \
  --k 3 \
  --alpha 0.5 \
  --max-radius -1
```

Outputs:

- `data/processed/pseudo_patch_fusion/h5/{sample_id}_fused.h5`
- `data/processed/pseudo_patch_fusion/fused_manifest.jsonl`
- `data/processed/pseudo_patch_fusion/fused_summary.json`

You can then run the same clustering pipeline on:

1. HE-only patch features
2. pseudo fused patch features

to compare clustering behavior.

## Optional: Hierarchical Clustering (L1 -> L2 refinement)

This script follows a CLUSTER-like two-level workflow:

1. fit prototypes
2. L1 clustering on prototypes
3. refine L2 inside each L1 cluster
4. fallback `NA` L2 to L1 as final cluster

Wrapper:

- `scripts/cluster_hierarchical_patches.sh`

Example A (fused features):

```bash
bash scripts/cluster_hierarchical_patches.sh \
  --manifest data/processed/pseudo_patch_fusion/fused_manifest.jsonl \
  --output-dir data/processed/hier_cluster_fused \
  --l1-method leiden \
  --n-prototypes 2048 \
  --l1-resolution 0.45 \
  --refine-resolution 0.60
```

Example B (HE-only features; if you prepare an HE-only manifest with `h5` field):

```bash
bash scripts/cluster_hierarchical_patches.sh \
  --manifest data/processed/he_only_manifest.jsonl \
  --output-dir data/processed/hier_cluster_he \
  --l1-method leiden \
  --n-prototypes 2048
```

Main outputs:

- `prototype_centers.npy`
- `prototype_clusters_L1L2.(parquet|csv)`
- `patch_final_clusters.(parquet|csv)`
- `slide_final_cluster_hist.(parquet|csv)`
- `cluster_meta.json`

## Optional: Map Cluster Labels Back to WSI (CLUSTER-style overlay)

Use final patch-level labels and draw them on WSI thumbnails:

```bash
bash scripts/overlay_hier_clusters_wsi.sh \
  --patch-final data/processed/hier_cluster_fused/patch_final_clusters.parquet \
  --output-dir data/processed/hier_cluster_fused_overlay \
  --downsample 16 \
  --alpha 0.40 \
  --legend
```

By default, `--wsi-dir` is auto-resolved from `CLUSTER/*.py` (`WSI_DIR=...`), so manual input is not required.
If needed, you can still override it with `--wsi-dir`, or set env `MMKIDNEY_WSI_DIR`.

For HE-only result, change `--patch-final` to the HE clustering output, e.g.:

```bash
bash scripts/overlay_hier_clusters_wsi.sh \
  --patch-final data/processed/hier_cluster_he/patch_final_clusters.parquet \
  --output-dir data/processed/hier_cluster_he_overlay \
  --downsample 16 \
  --alpha 0.40 \
  --legend
```

Useful outputs:

- `all_clusters/*_overlay_all.png`
- `filtered_clusters/*_overlay_filtered.png`
- `final_cluster_colors.csv`
- `overlay_meta.json`
