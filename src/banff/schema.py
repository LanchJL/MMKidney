from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional


@dataclass(frozen=True)
class BanffTaskSpec:
    name: str
    source_col: str
    num_classes: int
    stains: List[str]
    requires_extra_annotation: bool = False
    excluded: bool = False
    reason: str = ""


@dataclass(frozen=True)
class Feasibility:
    tier: str
    recommendation: str
    reason: str


CI_CT_STAIN_MODES: Dict[str, List[str]] = {
    "masson": ["MASSON"],
    "he": ["HE"],
    "masson_he": ["MASSON", "HE"],
}


def resolve_ci_ct_stains(mode: str) -> List[str]:
    key = str(mode or "").strip().lower().replace("+", "_").replace("-", "_")
    if key not in CI_CT_STAIN_MODES:
        raise ValueError(f"Unknown ci/ct stain mode: {mode}. Expected one of {sorted(CI_CT_STAIN_MODES)}")
    return list(CI_CT_STAIN_MODES[key])


BANFF_TASKS: Dict[str, BanffTaskSpec] = {
    "ci": BanffTaskSpec("ci", "ci", 4, list(CI_CT_STAIN_MODES["masson_he"])),
    "ct": BanffTaskSpec("ct", "ct", 4, list(CI_CT_STAIN_MODES["masson_he"])),
    "c4d": BanffTaskSpec("c4d", "C4d.1", 4, ["C4d"]),
    "pvl": BanffTaskSpec("pvl", "pvl", 4, ["SV40"]),
    "cg": BanffTaskSpec("cg", "cg", 4, ["BM", "PASM"]),
    "g": BanffTaskSpec("g", "g", 4, ["HE", "PAS"]),
    "i": BanffTaskSpec("i", "i", 4, ["HE", "PAS", "CD3"], requires_extra_annotation=True),
    "t": BanffTaskSpec("t", "t", 4, ["HE", "PAS", "CD3"], requires_extra_annotation=True),
    "v": BanffTaskSpec("v", "v", 4, ["HE", "PAS", "CD3"], requires_extra_annotation=True),
    "ptc_mononuclear": BanffTaskSpec(
        "ptc_mononuclear",
        "ptc（单个核）",
        4,
        ["HE", "PAS", "CD3", "CD4", "CD8"],
        requires_extra_annotation=True,
    ),
}


EXCLUDED_BANFF_TARGETS: Dict[str, BanffTaskSpec] = {
    "t_limited": BanffTaskSpec(
        "t_limited",
        "t（10x视野（中倍），≤3处小管炎，打1*）",
        4,
        ["HE", "PAS"],
        excluded=True,
        reason="Conflicts with the main t score and was requested for removal.",
    ),
    "c4d_false_positive": BanffTaskSpec(
        "c4d_false_positive",
        "C4d（备注为假阳性）",
        4,
        ["C4d"],
        excluded=True,
        reason="Sparse false-positive remark field; keep as audit metadata, not a target.",
    ),
    "ptc_pmn": BanffTaskSpec(
        "ptc_pmn",
        "ptc（仅考虑PMN）",
        4,
        ["HE", "PAS"],
        excluded=True,
        reason="Removed by policy; do not train the PMN-only ptc target.",
    ),
    "aah": BanffTaskSpec(
        "aah",
        "aah",
        4,
        ["HE", "PAS"],
        excluded=True,
        reason="No non-null labels in the current Excel.",
    ),
}


def _nonzero_classes(class_counts: Mapping[str, int]) -> List[int]:
    out = []
    for k, v in class_counts.items():
        if int(v) <= 0:
            continue
        try:
            out.append(int(float(k)))
        except Exception:
            continue
    return out


def assess_feasibility(
    label_count: int,
    aligned_count: int,
    class_counts: Mapping[str, int],
    requires_extra_annotation: bool,
    excluded: bool,
    min_aligned: int = 40,
    rare_class_threshold: int = 8,
) -> Feasibility:
    if excluded:
        return Feasibility("exclude_now", "Do not train", "Target is empty, contradictory, or removed by policy.")
    if label_count <= 0:
        return Feasibility("exclude_now", "Do not train", "No usable labels.")
    if aligned_count <= 0:
        return Feasibility("needs_extra_annotation", "Add matching WSI data", "Labels exist but no matching task stain is available.")
    if requires_extra_annotation:
        return Feasibility(
            "needs_extra_annotation",
            "Use as exploratory weak-MIL only; add region/cell/compartment labels for reliable scoring.",
            "The score depends on local structures that slide-level labels do not identify.",
        )
    if aligned_count < min_aligned:
        return Feasibility(
            "needs_extra_annotation",
            "Collect more matching slides or add targeted annotations.",
            f"Only {aligned_count} label+stain aligned samples.",
        )

    classes = _nonzero_classes(class_counts)
    if len(classes) < 2:
        return Feasibility("exclude_now", "Do not train", "Only one observed class.")
    if max(classes) < 3:
        return Feasibility(
            "binary_first",
            "Train binary score=0 vs score>0 first; full 0-3 ordinal scoring needs higher-grade cases.",
            "Current labels do not cover the full Banff 0-3 severity range.",
        )

    nonzero_rare = [int(v) for k, v in class_counts.items() if str(k) != "0" and int(v) < rare_class_threshold]
    if max(classes) > 1 and nonzero_rare:
        return Feasibility(
            "binary_first",
            "Train binary score=0 vs score>0 first; keep ordinal head experimental.",
            "Rare positive severity classes make direct ordinal training unstable.",
        )

    return Feasibility(
        "ready_slide_mil",
        "Train slide-level MIL or ordinal classification.",
        "Sufficient aligned labels and no mandatory local annotation requirement.",
    )


def task_specs(include_excluded: bool = False) -> Dict[str, BanffTaskSpec]:
    specs = dict(BANFF_TASKS)
    if include_excluded:
        specs.update(EXCLUDED_BANFF_TARGETS)
    return specs
