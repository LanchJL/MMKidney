import pandas as pd

from src.datasets.build_banff_first import build_banff_records, match_wsi_record


def test_match_wsi_record_allows_prefix_and_suffix_variants():
    records = [
        {"sample_id": "02266223S048744", "split": "val", "h5s": {"HE": "a.h5"}},
        {"sample_id": "22S0600871", "split": "train", "h5s": {"HE": "b.h5"}},
    ]

    assert match_wsi_record("23S048744", records)["sample_id"] == "02266223S048744"
    assert match_wsi_record("22S060087", records)["sample_id"] == "22S0600871"


def test_build_banff_records_normalizes_labels_masks_and_derived_outputs():
    df = pd.DataFrame(
        [
            {
                "病理号": "A001",
                "ci": 1,
                "ct": 2,
                "C4d.1": "2（IF）",
                "pvl": 1,
                "cg": "1b",
                "g": 0,
                "GT2": "Active AMR+IFTA Grade II",
            },
            {
                "病理号": "A002",
                "ci": 0,
                "ct": 0,
                "C4d.1": 0,
                "pvl": 0,
                "cg": 0,
                "g": 1,
                "GT2": "Normal Biopsy Or Nonspecific Changes",
            },
        ]
    )
    records = [
        {"sample_id": "A001", "split": "train", "h5s": {"MASSON": "m.h5", "C4d": "c.h5", "SV40": "s.h5", "BM": "b.h5"}},
        {"sample_id": "A002", "split": "val", "h5s": {"HE": "h.h5"}},
    ]

    out, report = build_banff_records(df, records)

    assert out[0]["banff_labels"]["ci"] == 1
    assert out[0]["banff_labels"]["ct"] == 2
    assert out[0]["banff_labels"]["c4d"] == 2
    assert out[0]["banff_labels"]["cg"] == 1
    assert out[0]["banff_masks"]["ci"] == 1.0
    assert out[0]["banff_masks"]["g"] == 0.0
    assert out[0]["derived"]["ifta_grade"] == 2
    assert out[0]["derived"]["pvn_class"] == 1
    assert out[1]["banff_masks"]["ci"] == 1.0
    assert out[1]["banff_masks"]["g"] == 1.0
    assert report["tasks"]["ci"]["aligned_count"] == 2
    assert report["tasks"]["g"]["aligned_count"] == 1


def test_ci_ct_report_accepts_he_as_masson_fallback():
    df = pd.DataFrame(
        [
            {"病理号": "A001", "ci": 1, "ct": 1, "GT2": "IFTA Grade I", "GT3": "IFTA Grade I"},
            {"病理号": "B002", "ci": 2, "ct": 2, "GT2": "IFTA Grade II", "GT3": "IFTA Grade II"},
        ]
    )
    records = [
        {"sample_id": "A001", "split": "train", "h5s": {"MASSON": "m.h5"}},
        {"sample_id": "B002", "split": "train", "h5s": {"HE": "h.h5"}},
    ]

    out, report = build_banff_records(df, records)

    assert [row["banff_masks"]["ci"] for row in out] == [1.0, 1.0]
    assert [row["banff_masks"]["ct"] for row in out] == [1.0, 1.0]
    assert report["tasks"]["ci"]["aligned_count"] == 2
    assert report["tasks"]["ct"]["aligned_count"] == 2
    assert report["tasks"]["ci"]["stain_mode_aligned_counts"] == {
        "masson": 1,
        "he": 1,
        "masson_he": 2,
    }
