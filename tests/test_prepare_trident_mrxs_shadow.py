from pathlib import Path

from scripts.prepare_trident_mrxs_shadow import (
    build_shadow_slide,
    inspect_slide_ini,
)


def _write_fixture_slide(root: Path, name: str, ini_text: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    slide = root / f"{name}.mrxs"
    slide.write_text("dummy mrxs shell\n", encoding="utf-8")
    sidecar = root / name
    sidecar.mkdir()
    (sidecar / "Slidedat.ini").write_text(ini_text, encoding="utf-8")
    (sidecar / "Data0000.dat").write_text("tile bytes\n", encoding="utf-8")
    (root / f"{name}.xml").write_text("<annotation_meta_data/>", encoding="utf-8")
    return slide


def test_inspect_slide_ini_reads_mpp_and_missing_objective(tmp_path: Path) -> None:
    slide = _write_fixture_slide(
        tmp_path,
        "case-A1",
        "\ufeff[GENERAL]\nSLIDE_ID = abc\n[LAYER_0_LEVEL_0_SECTION]\n"
        "MICROMETER_PER_PIXEL_X = 0.273809523809524\n",
    )

    info = inspect_slide_ini(slide)

    assert info.has_objective is False
    assert info.mpp_x == 0.273809523809524


def test_build_shadow_slide_patches_ini_and_symlinks_payload(tmp_path: Path) -> None:
    slide = _write_fixture_slide(
        tmp_path / "src",
        "case-A1",
        "[GENERAL]\nSLIDE_ID = abc\n[LAYER_0_LEVEL_0_SECTION]\n"
        "MICROMETER_PER_PIXEL_X = 0.273809523809524\n",
    )
    out_dir = tmp_path / "shadow"

    result = build_shadow_slide(slide, out_dir, default_objective=20)

    assert result.status == "patched"
    shadow_ini = out_dir / "case-A1" / "Slidedat.ini"
    assert shadow_ini.exists()
    assert "OBJECTIVE_MAGNIFICATION = 20" in shadow_ini.read_text(encoding="utf-8-sig")
    assert (out_dir / "case-A1.mrxs").exists()
    assert (out_dir / "case-A1.xml").is_symlink()
    assert (out_dir / "case-A1" / "Data0000.dat").is_symlink()
