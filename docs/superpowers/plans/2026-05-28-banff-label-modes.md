# Banff Label Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add clinically useful binary label modes for `ci`, `ct`, and derived `ifta` validation before returning to full 0/1/2/3 ordinal scoring.

**Architecture:** Keep raw Banff labels in the manifest, then transform them at dataset load time according to a per-task label mode. Add `ifta` as a derived WSI task from `max(ci, ct)` and configure model head class counts from the active label mode.

**Tech Stack:** Python, pandas, PyTorch, existing Banff-first dataset/model/training modules.

---

### Task 1: Label Mode Tests

**Files:**
- Modify: `tests/test_banff_dataset_model.py`
- Modify: `tests/test_banff_manifest_builder.py`

- [x] Add tests for `zero_vs_positive` and `low_vs_high` label transforms.
- [x] Add tests for per-task label mode overrides in `banff_task_config`.
- [x] Add tests that the manifest builder emits derived `ifta` labels and masks.
- [x] Run targeted tests and verify they fail before implementation.

### Task 2: Dataset And Training Implementation

**Files:**
- Modify: `src/banff/schema.py`
- Modify: `src/datasets/build_banff_first.py`
- Modify: `src/datasets/banff_dataset.py`
- Modify: `src/training/train_banff.py`

- [x] Add `ifta` to `BANFF_TASKS` using Masson+HE stains.
- [x] Add label mode parsing and per-task label mode configuration.
- [x] Transform labels at dataset load time without changing raw manifest labels.
- [x] Configure model heads as 2-class for binary modes and 4-class for ordinal mode.

### Task 3: Metrics, Docs, Verification

**Files:**
- Modify: `src/training/banff_metrics.py`
- Modify: `README.md`

- [x] Add predicted class counts and confusion matrix to per-task validation metrics.
- [x] Document recommended commands for `ci0 vs ci>0`, `ci0-1 vs ci2-3`, `ct0-1 vs ct2-3`, and `IFTA 0-I vs II-III`.
- [x] Rebuild Banff manifests.
- [x] Run Banff unit tests.
- [x] Run a small smoke train.
