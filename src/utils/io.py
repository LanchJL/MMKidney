import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def try_parse_datetime(x: str) -> Optional[datetime]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None

    # Excel serial date support (1900 system)
    if s.replace(".", "", 1).isdigit() and len(s) <= 7:
        try:
            day = float(s)
            base = datetime(1899, 12, 30)
            return base + timedelta(days=day)
        except Exception:
            pass

    for fmt in [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
    ]:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue

    try:
        return datetime.fromisoformat(s.replace("Z", ""))
    except Exception:
        return None
