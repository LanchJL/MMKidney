from collections import defaultdict
from typing import Dict, List, Tuple

# Label name mappings per level
label_mapping_L1 = {
    "Normal Biopsy Or Nonspecific Changes": "Normal Biopsy Or Nonspecific Changes",
    "Microvascular inflammation/injury (MVI), DSA-negative and C4d-negative": "Rejection",
    "Suspicious (Borderline) For Acute TCMR": "Rejection",
    "Active AMR": "Rejection",
    "Chronic Active AMR": "Rejection",
    "Chronic AMR": "Rejection",
    "Probable AMR": "Rejection",
    "C4d staining with acute tubular injury (ATI)": "Rejection",
    "C4d staining without evidence of rejection": "Rejection",
    "Acute TCMR IA": "Rejection",
    "Acute TCMR IB": "Rejection",
    "Acute TCMR IIA": "Rejection",
    "Acute TCMR IIB": "Rejection",
    "Acute TCMR III": "Rejection",
    "IFTA Grade I": "IFTA",
    "IFTA Grade II": "IFTA",
    "IFTA Grade III": "IFTA",
    "Polyomavirus Nephropathy, Class 1": "Polyomavirus Nephropathy",
    "Polyomavirus Nephropathy, Class 2": "Polyomavirus Nephropathy",
    "Polyomavirus Nephropathy, Class 3": "Polyomavirus Nephropathy",
    "Calcineurin Inhibitor Toxicity": "Calcineurin Inhibitor Toxicity",
}

label_mapping_L2 = {
    "Normal Biopsy Or Nonspecific Changes": "Normal Biopsy Or Nonspecific Changes",
    "Active AMR": "Antibody-mediated rejection and microvascular inflammation/injury (AMR/MVI)",
    "Chronic Active AMR": "Antibody-mediated rejection and microvascular inflammation/injury (AMR/MVI)",
    "Chronic AMR": "Antibody-mediated rejection and microvascular inflammation/injury (AMR/MVI)",
    "Probable AMR": "Antibody-mediated rejection and microvascular inflammation/injury (AMR/MVI)",
    "C4d staining with acute tubular injury (ATI)": "Antibody-mediated rejection and microvascular inflammation/injury (AMR/MVI)",
    "C4d staining without evidence of rejection": "Antibody-mediated rejection and microvascular inflammation/injury (AMR/MVI)",
    "Microvascular inflammation/injury (MVI), DSA-negative and C4d-negative": "Antibody-mediated rejection and microvascular inflammation/injury (AMR/MVI)",
    "Suspicious (Borderline) For Acute TCMR": "TCMR",
    "Acute TCMR IA": "TCMR",
    "Acute TCMR IB": "TCMR",
    "Acute TCMR IIA": "TCMR",
    "Acute TCMR IIB": "TCMR",
    "Acute TCMR III": "TCMR",
    "IFTA Grade I": "IFTA",
    "IFTA Grade II": "IFTA",
    "IFTA Grade III": "IFTA",
    "Polyomavirus Nephropathy, Class 1": "Polyomavirus Nephropathy",
    "Polyomavirus Nephropathy, Class 2": "Polyomavirus Nephropathy",
    "Polyomavirus Nephropathy, Class 3": "Polyomavirus Nephropathy",
    "Calcineurin Inhibitor Toxicity": "Calcineurin Inhibitor Toxicity",
}

label_mapping_L3 = {
    "Normal Biopsy Or Nonspecific Changes": "Normal Biopsy Or Nonspecific Changes",
    "Active AMR": "Active AMR",
    "Chronic Active AMR": "Chronic and Chronic Active AMR",
    "Chronic AMR": "Chronic and Chronic Active AMR",
    "Probable AMR": "Other MVI and C4d",
    "C4d staining with acute tubular injury (ATI)": "Other MVI and C4d",
    "C4d staining without evidence of rejection": "Other MVI and C4d",
    "Microvascular inflammation/injury (MVI), DSA-negative and C4d-negative": "Other MVI and C4d",
    "Suspicious (Borderline) For Acute TCMR": "Suspicious (Borderline) For Acute TCMR",
    "Acute TCMR IA": "Acute TCMR",
    "Acute TCMR IB": "Acute TCMR",
    "Acute TCMR IIA": "Acute TCMR",
    "Acute TCMR IIB": "Acute TCMR",
    "Acute TCMR III": "Acute TCMR",
    "IFTA Grade I": "IFTA",
    "IFTA Grade II": "IFTA",
    "IFTA Grade III": "IFTA",
    "Polyomavirus Nephropathy, Class 1": "Polyomavirus Nephropathy",
    "Polyomavirus Nephropathy, Class 2": "Polyomavirus Nephropathy",
    "Polyomavirus Nephropathy, Class 3": "Polyomavirus Nephropathy",
    "Calcineurin Inhibitor Toxicity": "Calcineurin Inhibitor Toxicity",
}

# Original index space
LABEL_MAP = {
    "L1": {
        "Calcineurin Inhibitor Toxicity": 0,
        "IFTA": 1,
        "Normal Biopsy Or Nonspecific Changes": 2,
        "Polyomavirus Nephropathy": 3,
        "Rejection": 4,
    },
    "L2": {
        "Antibody-mediated rejection and microvascular inflammation/injury (AMR/MVI)": 0,
        "Calcineurin Inhibitor Toxicity": 1,
        "IFTA": 2,
        "Normal Biopsy Or Nonspecific Changes": 3,
        "Polyomavirus Nephropathy": 4,
        "TCMR": 5,
    },
    "L3": {
        "Active AMR": 0,
        "Acute TCMR": 1,
        "Calcineurin Inhibitor Toxicity": 2,
        "Chronic and Chronic Active AMR": 3,
        "IFTA": 4,
        "Normal Biopsy Or Nonspecific Changes": 5,
        "Other MVI and C4d": 6,
        "Polyomavirus Nephropathy": 7,
        "Suspicious (Borderline) For Acute TCMR": 8,
    },
}

NORMAL_IDX = {"L1": 2, "L2": 3, "L3": 5}

KEPT_IDXS = {
    "L1": [0, 1, 3, 4],
    "L2": [0, 1, 2, 4, 5],
    "L3": [0, 1, 2, 3, 4, 6, 7, 8],
}


def _reindex_map(kept: List[int]) -> Dict[int, int]:
    return {orig: new for new, orig in enumerate(kept)}


def _name2orig(level: str, name: str) -> int:
    if name not in LABEL_MAP[level]:
        raise KeyError(f"[{level}] missing label name: {name}")
    return LABEL_MAP[level][name]


def build_parent_map_auto() -> Tuple[Dict[str, Dict[int, List[int]]], Dict[str, Dict[int, int]]]:
    """
    Returns:
      parent_map: structured mapping in the "Normal-removed" index space
      REINDEX: original_idx -> new_idx per level
    """
    REINDEX = {lv: _reindex_map(KEPT_IDXS[lv]) for lv in ["L1", "L2", "L3"]}

    fine_keys = set(label_mapping_L1.keys()) | set(label_mapping_L2.keys()) | set(label_mapping_L3.keys())

    rel_L1_L2 = defaultdict(set)
    rel_L2_L3 = defaultdict(set)

    for fine in fine_keys:
        if fine not in label_mapping_L1 or fine not in label_mapping_L2 or fine not in label_mapping_L3:
            continue
        u1 = label_mapping_L1[fine]
        u2 = label_mapping_L2[fine]
        u3 = label_mapping_L3[fine]

        try:
            p1_orig = _name2orig("L1", u1)
            c2_orig = _name2orig("L2", u2)
            c3_orig = _name2orig("L3", u3)
        except KeyError:
            continue

        if p1_orig in REINDEX["L1"] and c2_orig in REINDEX["L2"]:
            rel_L1_L2[REINDEX["L1"][p1_orig]].add(REINDEX["L2"][c2_orig])
        if c2_orig in REINDEX["L2"] and c3_orig in REINDEX["L3"]:
            rel_L2_L3[REINDEX["L2"][c2_orig]].add(REINDEX["L3"][c3_orig])

    parent_map = {
        "L1<-L2": {p: sorted(list(ch)) for p, ch in rel_L1_L2.items()},
        "L2<-L3": {p: sorted(list(ch)) for p, ch in rel_L2_L3.items()},
    }
    return parent_map, REINDEX
