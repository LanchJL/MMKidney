# Banff-First Multitask Design

## Goal

Add a parallel Banff-first training path that predicts lesion scores from stain-specific WSI features, then derives IFTA, BKVN/PVN, and AMR-related diagnoses with deterministic rules.

## Scope

The existing L1/L2/L3 diagnosis model remains untouched. The new path creates Banff manifests, feasibility reports, Banff-specific datasets, a multitask WSI model, masked losses, metrics, and rule utilities.

## Tasks

The first implementation covers:

- `ci`, `ct` from Masson, HE, or Masson+HE with derived IFTA grade. HE is allowed as a fallback/ablation input because available HE slides are higher quality than some faded Masson slides.
- `C4d.1` from C4d.
- `pvl` from SV40 with pvl+ci BKVN/PVN class rules.
- `cg` from BM/PASM as `cg0` vs `cg>0`, preserving raw categories.
- `g` from HE/PAS.
- Optional lower-confidence heads for `i`, `t`, `v`, and `ptc_mononuclear`.
- Explicitly excluded targets: limited-field `t`, PMN-only `ptc`, false-positive C4d remark, and `aah`.

## Feasibility Policy

Each Banff target is assigned a feasibility tier using available labels, available stain-aligned slides, class balance, and whether the morphology can be learned from slide-level labels.

- `ready_slide_mil`: enough slide-level labels and matching stains for weakly supervised MIL.
- `binary_first`: ordinal labels exist, but rare classes make binary learning the first stable target.
- `needs_extra_annotation`: slide-level labels are too scarce, too imbalanced, or the target requires local structures not reliably learnable without region/cell/tubule/glomerulus annotation.
- `exclude_now`: empty, contradictory, or intentionally removed targets.

## Architecture

The Banff model reuses the existing stain bag encoder. It keeps stain-token outputs and creates task-specific logits from the relevant stain representations, falling back to fused patient representation only for rule-level diagnosis experiments. Losses are masked per sample-task, so a sample can train C4d without Masson or SV40.

## Data Flow

1. Read the Excel sheet `移植肾汇总-筛选后`.
2. Join rows to existing WSI manifests by standardized pathology id, with relaxed matching for common suffix/prefix variants.
3. Normalize Banff labels into class indices and per-task masks.
4. Emit train/val/test Banff manifests and `banff_feasibility_report.json`.
5. Train stain-specific Banff heads with masked class-balanced cross entropy.
6. Derive diagnostic fields from predicted scores with rule utilities.

## Rule Choices

IFTA grade uses `max(ci, ct)`.

BKVN/PVN class uses the pvl+ci table provided by the user:

- pvl0: no BKVN by pvl.
- pvl1 with ci0/ci1: class 1.
- pvl1 with ci2/ci3, any pvl2, or pvl3 with ci0/ci1: class 2.
- pvl3 with ci2/ci3: class 3.

AMR rules will be implemented as an explicit post-processor with priority ordering. Where user rules conflict, `C4d>0 and PRA+` is treated as `Probable AMR`, otherwise `C4d>0` confirms `Active AMR`.

## Testing

Unit tests cover label normalization, IFTA and PVN derivation, feasibility tiering, masked loss, and a small forward pass with synthetic bags.
