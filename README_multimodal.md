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

## Optional: Hierarchical Clustering (L1 -> L2, optional L3 refinement)

This script follows a CLUSTER-like hierarchical workflow:

1. fit prototypes
2. L1 clustering on prototypes
3. refine L2 inside each L1 cluster
4. optional refine L3 inside selected L2 clusters
5. fallback chain for final labels: `L3 -> L2 -> L1`

Wrapper:

- `scripts/cluster_hierarchical_patches.sh`
- `scripts/cluster_hierarchical_steps.sh` (`fit|assign|l1|l2|final|check`, CLUSTER-like stepwise debug)

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

You can also pass existing `train/val/test_manifest.jsonl` directly (with `h5s` dict).
The clustering script now auto-selects `h5s["HE"]` by default (`--stain-name HE`).

Example C (HE-only + adaptive L3):

```bash
bash scripts/cluster_hierarchical_patches.sh \
  --manifest data/processed/he_only_manifest.jsonl \
  --output-dir data/processed/hier_cluster_he_l3 \
  --l1-method leiden \
  --n-prototypes 2048 \
  --l1-resolution 0.45 \
  --refine-resolution 0.60 \
  --enable-l3 \
  --l3-resolution 0.75 \
  --min-l2-to-refine-l3 35 \
  --min-l3-prototypes 10 \
  --min-l3-slides 3 \
  --min-l3-patch-frac 0.003 \
  --min-l3-silhouette 0.06
```

Recommended robust HE setting (with stability gate + balanced prototype sampling + spatial smoothing):

```bash
bash scripts/cluster_hierarchical_patches.sh \
  --manifest data/processed/he_only_manifest.jsonl \
  --output-dir data/processed/hier_cluster_he_robust \
  --l1-method leiden \
  --n-prototypes 2048 \
  --prototype-patches-per-sample 3000 \
  --min-prototypes-to-refine 40 \
  --min-l2-silhouette 0.05 \
  --stability-n-seeds 3 \
  --min-stability-ari 0.70 \
  --min-l2-slides 3 \
  --min-l2-patch-frac 0.005 \
  --enable-l3 \
  --min-l2-to-refine-l3 35 \
  --min-l3-silhouette 0.06 \
  --spatial-smooth-k 7 \
  --spatial-smooth-iter 1
```

Main outputs:

- `prototype_centers.npy`
- `prototype_clusters_L1L2.(parquet|csv)`
- `patch_final_clusters.(parquet|csv)`
- `slide_final_cluster_hist.(parquet|csv)`
- `cluster_meta.json`
- `refine_decisions.json` (records per-parent refine/skip reasons for L2/L3)

### Step-wise Run (strict CLUSTER-like debugging)

Use this when results look wrong and you want to inspect every stage:

```bash
OUT=data/processed/hier_cluster_he_steps
MAN=data/processed/manifests/all_manifest.jsonl
```

```bash
bash scripts/cluster_hierarchical_steps.sh --step fit    --manifest "$MAN" --stain-name HE --output-dir "$OUT" --n-prototypes 2048 --prototype-patches-per-sample 3000
bash scripts/cluster_hierarchical_steps.sh --step assign --manifest "$MAN" --stain-name HE --output-dir "$OUT"
bash scripts/cluster_hierarchical_steps.sh --step l1     --output-dir "$OUT" --l1-method leiden --l1-resolution 0.45
bash scripts/cluster_hierarchical_steps.sh --step l2     --output-dir "$OUT" --l1-method leiden --refine-resolution 0.60 --min-prototypes-to-refine 40 --min-l2-silhouette 0.05 --stability-n-seeds 3 --min-stability-ari 0.70 --enable-l3
bash scripts/cluster_hierarchical_steps.sh --step final  --output-dir "$OUT" --spatial-smooth-k 7 --spatial-smooth-iter 1
bash scripts/cluster_hierarchical_steps.sh --step check  --output-dir "$OUT"
```

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
- `precheck_missing_wsi.csv` (generated when some slide ids cannot be matched to WSI files)

Matching policy:

- default is strict normalized-id match (safer, prevents cross-slide mapping)
- add `--allow-fuzzy-match` only if your filenames are inconsistent and strict mode misses too many slides
- pass explicit `--wsi-dir` to override auto-resolved CLUSTER path
