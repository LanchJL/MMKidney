from pathlib import Path

from PIL import Image

from scripts.make_valis_registration_qc import make_qc_panel


def test_make_qc_panel_writes_side_by_side_overlay_and_checkerboard(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed.png"
    moving = tmp_path / "moving.png"
    out = tmp_path / "qc.jpg"
    Image.new("RGB", (64, 48), (200, 80, 80)).save(fixed)
    Image.new("RGB", (64, 48), (80, 160, 220)).save(moving)

    make_qc_panel(fixed, moving, out, max_panel_width=64, labels=("moving", "fixed"))

    result = Image.open(out)
    assert result.size[0] == 64 * 4
    assert result.size[1] > 48
