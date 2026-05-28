import math
import re
from typing import Dict, Optional


def parse_banff_score(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    match = re.search(r"\d+", text)
    if not match:
        return None
    return int(match.group(0))


def derive_ifta_grade(ci, ct) -> Optional[int]:
    ci_score = parse_banff_score(ci)
    ct_score = parse_banff_score(ct)
    if ci_score is None and ct_score is None:
        return None
    if ci_score is None:
        return ct_score
    if ct_score is None:
        return ci_score
    return max(ci_score, ct_score)


def derive_pvn_class(pvl, ci) -> Optional[int]:
    pvl_score = parse_banff_score(pvl)
    ci_score = parse_banff_score(ci)
    if pvl_score is None or ci_score is None:
        return None
    if pvl_score == 0:
        return 0
    if pvl_score == 1 and ci_score in {0, 1}:
        return 1
    if pvl_score == 3 and ci_score in {2, 3}:
        return 3
    if pvl_score in {1, 2, 3}:
        return 2
    return None


def apply_amr_rules(model_diagnosis: str, c4d=None, cg=None, pra_positive: Optional[bool] = None) -> Dict[str, str]:
    diagnosis = str(model_diagnosis or "")
    c4d_score = parse_banff_score(c4d)
    cg_score = parse_banff_score(cg)
    c4d_pos = c4d_score is not None and c4d_score > 0
    cg_pos = cg_score is not None and cg_score > 0

    if diagnosis == "Active AMR":
        if not c4d_pos and pra_positive is False:
            return {"diagnosis": "Rejected Active AMR", "reason": "C4d=0 and nearest pre-biopsy PRA is negative."}
        if c4d_pos and pra_positive is True:
            return {"diagnosis": "Probable AMR", "reason": "C4d>0 and nearest pre-biopsy PRA is positive."}
        if c4d_pos:
            return {"diagnosis": "Active AMR", "reason": "C4d>0 confirms active AMR rule path."}
        return {"diagnosis": "Active AMR", "reason": "Insufficient PRA/C4d rule evidence to reject."}

    if diagnosis == "Chronic and Chronic Active AMR":
        if not cg_pos:
            return {"diagnosis": "Rejected Chronic AMR", "reason": "cg=0 rejects chronic/chronic active AMR."}
        if c4d_pos:
            return {"diagnosis": "Chronic Active AMR", "reason": "cg>0 and C4d>0."}
        return {"diagnosis": "Chronic AMR", "reason": "cg>0 and C4d=0."}

    if diagnosis == "Other MVI and C4d":
        if c4d_pos:
            return {"diagnosis": "C4d without evidence of rejection", "reason": "C4d>0 without AMR path."}
        return {"diagnosis": "MVI", "reason": "C4d=0."}

    return {"diagnosis": diagnosis, "reason": "No Banff-first AMR rule applied."}
