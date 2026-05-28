# Banff-First Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a parallel Banff-first multitask WSI pipeline for lesion score prediction and rule-based diagnostic derivation.

**Architecture:** Add focused modules for Banff schema/rules, manifest building, dataset collation, multitask model heads, losses, metrics, and training. Reuse existing H5 readers and stain bag encoders so feature extraction does not change.

**Tech Stack:** Python, pandas, PyTorch, existing MMKidney H5 feature manifests.

---

### Task 1: Banff Schema And Rules

**Files:**
- Create: `src/banff/schema.py`
- Create: `src/banff/rules.py`
- Test: `tests/test_banff_rules.py`

- [x] Add tests for score parsing, IFTA grade derivation, PVN class derivation, and feasibility tier assignment.
- [x] Implement schema constants and rules with deterministic functions.
- [x] Run Banff rule tests.

### Task 2: Banff Manifest Builder

**Files:**
- Create: `src/datasets/build_banff_first.py`
- Test: `tests/test_banff_manifest_builder.py`

- [x] Add tests using a tiny synthetic Excel-like dataframe and synthetic WSI records.
- [x] Implement relaxed pathology-id matching, Banff label normalization, feasibility summaries, and JSONL/JSON writing.
- [x] Run Banff manifest builder tests.

### Task 3: Dataset And Collate

**Files:**
- Create: `src/datasets/banff_dataset.py`
- Modify: `src/datasets/collate.py`
- Test: `tests/test_banff_dataset_model.py`

- [x] Add synthetic dataset/collate tests that do not require real H5 files.
- [x] Implement Banff-specific sample parsing and batch collation.
- [x] Run Banff dataset/model tests.

### Task 4: Model, Loss, Metrics

**Files:**
- Create: `src/models/banff_model.py`
- Create: `src/losses/banff_loss.py`
- Create: `src/training/banff_metrics.py`
- Test: `tests/test_banff_dataset_model.py`

- [x] Add synthetic forward/loss tests.
- [x] Implement stain-specific task heads, masked cross entropy, and per-task metrics.
- [x] Run Banff dataset/model tests.

### Task 5: Training And Rule Export

**Files:**
- Create: `src/training/train_banff.py`
- Create: `scripts/build_banff_first_manifests.sh`
- Modify: `README.md`

- [x] Add CLI for training Banff-first model.
- [x] Add shell entrypoint for manifest creation.
- [x] Document commands and feasibility report interpretation.
- [x] Run manifest build against the real Excel and current manifests.
