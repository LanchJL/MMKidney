import pytest

from src.banff.rules import (
    apply_amr_rules,
    derive_ifta_grade,
    derive_pvn_class,
    parse_banff_score,
)
from src.banff.schema import assess_feasibility


def test_parse_banff_score_handles_stars_letter_suffixes_and_if_notes():
    assert parse_banff_score("1*") == 1
    assert parse_banff_score("1b") == 1
    assert parse_banff_score("2（IF）") == 2
    assert parse_banff_score(3.0) == 3
    assert parse_banff_score("") is None


def test_ifta_grade_uses_maximum_ci_or_ct_score():
    assert derive_ifta_grade(0, 0) == 0
    assert derive_ifta_grade(1, 0) == 1
    assert derive_ifta_grade(1, 2) == 2
    assert derive_ifta_grade(3, 2) == 3
    assert derive_ifta_grade(None, None) is None


@pytest.mark.parametrize(
    ("pvl", "ci", "expected"),
    [
        (0, 3, 0),
        (1, 0, 1),
        (1, 1, 1),
        (1, 2, 2),
        (2, 0, 2),
        (2, 3, 2),
        (3, 1, 2),
        (3, 2, 3),
    ],
)
def test_pvn_class_follows_user_pvl_ci_table(pvl, ci, expected):
    assert derive_pvn_class(pvl, ci) == expected


def test_amr_rules_use_c4d_pra_and_cg_priority():
    assert apply_amr_rules("Active AMR", c4d=0, pra_positive=False)["diagnosis"] == "Rejected Active AMR"
    assert apply_amr_rules("Active AMR", c4d=1, pra_positive=False)["diagnosis"] == "Active AMR"
    assert apply_amr_rules("Active AMR", c4d=1, pra_positive=True)["diagnosis"] == "Probable AMR"
    assert apply_amr_rules("Chronic and Chronic Active AMR", c4d=1, cg=0)["diagnosis"] == "Rejected Chronic AMR"
    assert apply_amr_rules("Chronic and Chronic Active AMR", c4d=1, cg=1)["diagnosis"] == "Chronic Active AMR"
    assert apply_amr_rules("Chronic and Chronic Active AMR", c4d=0, cg=1)["diagnosis"] == "Chronic AMR"
    assert apply_amr_rules("Other MVI and C4d", c4d=0)["diagnosis"] == "MVI"
    assert apply_amr_rules("Other MVI and C4d", c4d=2)["diagnosis"] == "C4d without evidence of rejection"


def test_feasibility_tiers_separate_ready_binary_extra_annotation_and_excluded():
    assert (
        assess_feasibility(
            label_count=120,
            aligned_count=95,
            class_counts={"0": 25, "1": 40, "2": 20, "3": 10},
            requires_extra_annotation=False,
            excluded=False,
        ).tier
        == "ready_slide_mil"
    )
    assert (
        assess_feasibility(
            label_count=180,
            aligned_count=160,
            class_counts={"0": 145, "1": 10, "2": 4, "3": 1},
            requires_extra_annotation=False,
            excluded=False,
        ).tier
        == "binary_first"
    )
    assert (
        assess_feasibility(
            label_count=180,
            aligned_count=160,
            class_counts={"0": 90, "1": 70},
            requires_extra_annotation=True,
            excluded=False,
        ).tier
        == "needs_extra_annotation"
    )
    assert (
        assess_feasibility(
            label_count=0,
            aligned_count=0,
            class_counts={},
            requires_extra_annotation=False,
            excluded=True,
        ).tier
        == "exclude_now"
    )
