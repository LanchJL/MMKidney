import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


_ALIAS_MAP = {
    "CK(PAN)": "CK_pan",
    "CK_PAN": "CK_pan",
    "CKPAN": "CK_pan",
    "CK_PAN_": "CK_pan",
    "CK-PAN": "CK_pan",
    "KI67": "KI67",
    "MASSON": "MASSON",
    "TIA-1": "TIA",
    "CD79A": "CD79",
}


def normalize_stain_name(name: str) -> str:
    """
    Normalize stain aliases:
    - strip suffixes like _v2, _v3, _V2
    - unify separators
    - map known aliases to canonical names
    """
    if name is None:
        return "UNKNOWN"
    x = str(name).strip()
    if not x:
        return "UNKNOWN"

    x = x.replace(" ", "")
    x = x.replace("/", "_")
    x = x.replace("__", "_")
    x = re.sub(r"(?i)_v\d+$", "", x)
    x = x.rstrip("_")

    # Keep canonical casing where possible
    up = x.upper()
    if up in _ALIAS_MAP:
        return _ALIAS_MAP[up]

    if up.startswith("CK_PAN"):
        return "CK_pan"
    if up.startswith("CD79"):
        return "CD79"

    # Preserve common mixed case token
    if up == "FOXP3":
        return "Foxp3"

    return x


def standardize_pathology_id(value: str) -> str:
    if value is None:
        return ""
    x = str(value).strip().upper()
    x = re.sub(r"\s+", "", x)
    return x


def build_stain_vocab(stains: Iterable[str]) -> Dict[str, int]:
    uniq = sorted({normalize_stain_name(s) for s in stains if s})
    return {s: i for i, s in enumerate(uniq)}


def dump_stain_vocab(path: Path, vocab: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)


def normalize_h5s(h5s: Dict[str, str]) -> Tuple[Dict[str, str], List[str], List[str]]:
    """Return canonical->path mapping with first-seen policy and stain name lists."""
    raw_names = sorted(list(h5s.keys()))
    norm = {}
    for k, v in h5s.items():
        kk = normalize_stain_name(k)
        if kk not in norm:
            norm[kk] = v
    return norm, raw_names, sorted(list(norm.keys()))
