import torch

from src.datasets.banff_dataset import labels_from_banff_record
from src.datasets.collate import collate_banff_patient_batch
from src.losses.banff_loss import compute_banff_loss
from src.models.banff_model import BanffTaskHeads
from src.training.banff_metrics import summarize_banff_metrics
from src.training.train_banff import banff_task_config


def test_banff_task_heads_select_relevant_stain_tokens():
    heads = BanffTaskHeads(
        in_dim=4,
        tasks={
            "ci": {"num_classes": 4, "stains": ["MASSON"]},
            "g": {"num_classes": 4, "stains": ["HE", "PAS"]},
        },
        hidden_dim=8,
        dropout=0.0,
    )
    encoded = {
        "patient_repr": torch.zeros(2, 4),
        "stain_reprs": [
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
        ],
        "aux": [
            {"stain_names": ["HE", "MASSON"]},
            {"stain_names": ["PAS"]},
        ],
    }

    out = heads(encoded)

    assert set(out["banff_logits"]) == {"ci", "g"}
    assert out["banff_logits"]["ci"].shape == (2, 4)
    assert out["banff_task_masks"]["ci"].tolist() == [1.0, 0.0]
    assert out["banff_task_masks"]["g"].tolist() == [1.0, 1.0]


def test_compute_banff_loss_ignores_masked_samples_and_tasks():
    logits = {
        "ci": torch.tensor([[4.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]], requires_grad=True),
        "pvl": torch.tensor([[0.0, 4.0], [4.0, 0.0]], requires_grad=True),
    }
    labels = {
        "ci": torch.tensor([0, 1]),
        "pvl": torch.tensor([1, 0]),
    }
    masks = {
        "ci": torch.tensor([1.0, 0.0]),
        "pvl": torch.tensor([0.0, 0.0]),
    }

    loss, logs = compute_banff_loss(logits, labels, masks)

    assert loss.item() < 0.1
    assert logs["ci_n"] == 1
    assert logs["pvl_n"] == 0
    loss.backward()
    assert logits["ci"].grad is not None


def test_labels_from_banff_record_and_collate_preserve_banff_masks():
    rec = {
        "banff_labels": {"ci": 2, "pvl": -1},
        "banff_masks": {"ci": 1.0, "pvl": 0.0},
    }
    labels, masks = labels_from_banff_record(rec, tasks=["ci", "pvl"])

    assert labels["ci"].item() == 2
    assert labels["pvl"].item() == 0
    assert masks["ci"].item() == 1.0
    assert masks["pvl"].item() == 0.0

    batch = collate_banff_patient_batch(
        [
            {
                "sample_id": "A",
                "split": "train",
                "wsi_stains_list": [],
                "banff_labels": labels,
                "banff_masks": masks,
            },
            {
                "sample_id": "B",
                "split": "val",
                "wsi_stains_list": [],
                "banff_labels": labels,
                "banff_masks": masks,
            },
        ]
    )

    assert batch["banff_labels"]["ci"].tolist() == [2, 2]
    assert batch["banff_masks"]["pvl"].tolist() == [0.0, 0.0]


def test_labels_from_banff_record_can_collapse_binary_first_tasks():
    rec = {
        "banff_labels": {"c4d": 3, "pvl": 0},
        "banff_masks": {"c4d": 1.0, "pvl": 1.0},
    }

    labels, masks = labels_from_banff_record(rec, tasks=["c4d", "pvl"], binary_tasks=["c4d", "pvl"])

    assert labels["c4d"].item() == 1
    assert labels["pvl"].item() == 0
    assert masks["c4d"].item() == 1.0


def test_summarize_banff_metrics_uses_masks():
    probs = {
        "ci": torch.tensor(
            [
                [0.9, 0.1, 0.0, 0.0],
                [0.1, 0.8, 0.1, 0.0],
                [0.1, 0.2, 0.7, 0.0],
            ]
        )
    }
    labels = {"ci": torch.tensor([0, 2, 2])}
    masks = {"ci": torch.tensor([1.0, 0.0, 1.0])}

    summary = summarize_banff_metrics(probs, labels, masks)

    assert summary["ci"]["n"] == 2
    assert summary["ci"]["accuracy"] == 1.0
    assert summary["mean_accuracy"] == 1.0


def test_banff_task_config_can_switch_ci_ct_stain_mode():
    config = banff_task_config(["ci", "ct", "g"], binary_tasks=[], ci_ct_stain_mode="he")

    assert config["ci"]["stains"] == ["HE"]
    assert config["ct"]["stains"] == ["HE"]
    assert config["g"]["stains"] == ["HE", "PAS"]

    combined = banff_task_config(["ci"], binary_tasks=[], ci_ct_stain_mode="masson_he")
    assert combined["ci"]["stains"] == ["MASSON", "HE"]
