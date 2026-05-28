# CI/CT HE Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow `ci` and `ct` Banff score experiments to compare Masson-only, HE-only, and Masson+HE inputs.

**Architecture:** Keep one Banff manifest with broad `ci/ct` masks that accept either Masson or HE, then let training choose the actual `ci/ct` stain mode through task-head configuration. The feasibility report records per-mode aligned counts so each ablation has a clear data denominator.

**Tech Stack:** Python, pandas, PyTorch, existing Banff-first manifest/model/training modules.

---

### Task 1: Red Tests For CI/CT Stain Modes

**Files:**
- Modify: `tests/test_banff_manifest_builder.py`
- Modify: `tests/test_banff_dataset_model.py`

- [x] Add a manifest-builder test where one case has only Masson and another has only HE; expected `ci/ct` aligned count is 2 and per-mode counts show Masson=1, HE=1, Masson+HE=2.
- [x] Add a train-config test that calls `banff_task_config(..., ci_ct_stain_mode="he")` and expects `ci/ct` heads to use `["HE"]`.
- [x] Run the targeted tests and verify they fail against the existing Masson-only implementation.

### Task 2: Implement Shared CI/CT Stain Modes

**Files:**
- Modify: `src/banff/schema.py`
- Modify: `src/datasets/build_banff_first.py`
- Modify: `src/training/train_banff.py`

- [x] Define `CI_CT_STAIN_MODES` and `resolve_ci_ct_stains`.
- [x] Change default `ci/ct` task stains to `["MASSON", "HE"]`.
- [x] Add `stain_mode_aligned_counts` for `ci/ct` in `banff_feasibility_report.json`.
- [x] Add CLI option `--ci-ct-stain-mode {masson,he,masson_he}` to training.

### Task 3: Documentation And Verification

**Files:**
- Modify: `README.md`
- Regenerate: `data/processed_banff/manifests/banff_feasibility_report.json`

- [x] Document the three ablation commands.
- [x] Rebuild Banff manifests from the real Excel.
- [x] Run Banff tests and a small train smoke test.
