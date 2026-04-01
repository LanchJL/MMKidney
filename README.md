# Clean WSI MIL (Kidney) - Minimal Core

This folder contains a clean, minimal implementation of the core deep learning pipeline:

- Hierarchical multi-label heads (L1/L2/L3)
- MIL for patch features
- Multi-stain late fusion (per-stain MIL + stain attention)
- Optional extra multimodal vector fused by concatenation

## Structure

- `models.py` - model definitions
- `losses.py` - hierarchical loss + attention diversity loss
- `hierarchy.py` - label hierarchy mapping
- `data.py` - datasets + collate + dataloaders
- `utils.py` - seed, metrics, config
- `train.py` - training entrypoint

## Data Format

All inputs are JSON lists. Each item is a sample.

### 1. Vector mode (`--mode vector`)

Slide-level feature vectors.

```json
{
  "id": "SID_00001",
  "h5": "/path/to/slide_features.h5",
  "labels": {"L1": [0,1,0,0], "L2": [1,0,0,0,0], "L3": [0,0,1,0,0,0,0,0]}
}
```

`h5` must contain `features`.

### 2. MIL mode (`--mode mil`)

Patch feature sets per slide.

```json
{
  "id": "SID_00001",
  "h5": "/path/to/patch_features.h5",
  "labels": {"L1": [0,1,0,0], "L2": [1,0,0,0,0], "L3": [0,0,1,0,0,0,0,0]}
}
```

`h5` must contain one of: `patches`, `features`, `feats`, `x`.

### 3. Multi-stain MIL (`--mode mil_ms`)

```json
{
  "id": "PATIENT_001",
  "h5s": {
    "HE": "/path/to/HE.h5",
    "PAS": "/path/to/PAS.h5"
  },
  "labels": {"L1": [0,1,0,0], "L2": [1,0,0,0,0], "L3": [0,0,1,0,0,0,0,0]}
}
```

Missing stains are skipped per sample (the model fuses available stains).

### Optional multimodal vector

If you have extra modalities (labs, clinical features, etc.), add an `extra` vector per sample and pass `--extra-dim`.

```json
{
  "id": "SID_00001",
  "h5": "/path/to/slide_features.h5",
  "labels": {"L1": [0,1,0,0], "L2": [1,0,0,0,0], "L3": [0,0,1,0,0,0,0,0]},
  "extra": [0.12, 5.7, 1.0, 0.0]
}
```

## Train

```bash
python code/train.py \
  --mode mil_ms \
  --train-json /path/train_ms.json \
  --val-json /path/val_ms.json \
  --test-json /path/test_ms.json \
  --out-dir /path/out \
  --batch-size 8 \
  --epochs 50 \
  --extra-dim 4
```

## Notes

- Labels are multi-hot vectors in the "Normal removed" index space.
- `hierarchy.py` defines the hierarchy used by the consistency loss.
- This clean folder is isolated from the rest of the repo and intended as the minimal core.
