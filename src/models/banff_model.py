from typing import Dict, List

import torch
import torch.nn as nn

from src.banff.schema import BANFF_TASKS
from src.models.wsi_model import WSIEncoder


class BanffTaskHeads(nn.Module):
    def __init__(self, in_dim: int, tasks: Dict[str, Dict], hidden_dim: int, dropout: float = 0.25):
        super().__init__()
        self.tasks = tasks
        self.heads = nn.ModuleDict()
        for name, spec in tasks.items():
            self.heads[name] = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, int(spec["num_classes"])),
            )

    @staticmethod
    def _select_task_repr(sample_tokens: torch.Tensor, stain_names: List[str], desired_stains: List[str]):
        idx = [i for i, name in enumerate(stain_names) if name in desired_stains]
        if not idx:
            return None
        selected = sample_tokens[idx]
        return selected.mean(dim=0)

    def forward(self, encoded: Dict) -> Dict:
        logits = {name: [] for name in self.tasks}
        task_masks = {name: [] for name in self.tasks}
        patient_repr = encoded["patient_repr"]

        for b_idx, sample_tokens in enumerate(encoded["stain_reprs"]):
            stain_names = encoded["aux"][b_idx]["stain_names"]
            for name, spec in self.tasks.items():
                z = self._select_task_repr(sample_tokens, stain_names, list(spec["stains"]))
                if z is None:
                    z = patient_repr[b_idx] * 0.0
                    task_masks[name].append(0.0)
                else:
                    task_masks[name].append(1.0)
                logits[name].append(self.heads[name](z))

        out_logits = {name: torch.stack(vals, dim=0) for name, vals in logits.items()}
        out_masks = {
            name: torch.tensor(vals, dtype=torch.float32, device=patient_repr.device)
            for name, vals in task_masks.items()
        }
        return {"banff_logits": out_logits, "banff_task_masks": out_masks}


class BanffFirstWSIModel(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        proj_dim: int,
        stain_vocab_size: int,
        stain_emb_dim: int,
        num_heads: int = 4,
        num_set_layers: int = 2,
        dropout: float = 0.25,
        he_stain_id: int = 0,
        use_he_adapter: bool = True,
        tasks: Dict[str, Dict] = None,
    ):
        super().__init__()
        self.wsi_encoder = WSIEncoder(
            feat_dim=feat_dim,
            proj_dim=proj_dim,
            stain_vocab_size=stain_vocab_size,
            stain_emb_dim=stain_emb_dim,
            num_heads=num_heads,
            num_set_layers=num_set_layers,
            dropout=dropout,
            he_stain_id=he_stain_id,
            use_he_adapter=use_he_adapter,
        )
        task_specs = tasks or {
            name: {"num_classes": spec.num_classes, "stains": spec.stains}
            for name, spec in BANFF_TASKS.items()
        }
        self.banff_heads = BanffTaskHeads(
            in_dim=proj_dim,
            tasks=task_specs,
            hidden_dim=proj_dim,
            dropout=dropout,
        )

    def forward(self, batch: Dict) -> Dict:
        encoded = self.wsi_encoder(batch["wsi"])
        pred = self.banff_heads(encoded)
        return {**encoded, **pred}
