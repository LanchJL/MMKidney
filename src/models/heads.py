import torch
import torch.nn as nn


class HierarchicalHeads(nn.Module):
    def __init__(self, in_dim: int, dims=(4, 5, 8), hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        d1, d2, d3 = dims
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.l1 = nn.Linear(hidden_dim, d1)
        self.l2 = nn.Linear(hidden_dim + d1, d2)
        self.l3 = nn.Linear(hidden_dim + d1 + d2, d3)

    def forward(self, h):
        z = self.backbone(h)
        logits_l1 = self.l1(z)
        p1 = torch.sigmoid(logits_l1)
        logits_l2 = self.l2(torch.cat([z, p1], dim=1))
        p2 = torch.sigmoid(logits_l2)
        logits_l3 = self.l3(torch.cat([z, p1, p2], dim=1))
        return {
            "logits_L1": logits_l1,
            "logits_L2": logits_l2,
            "logits_L3": logits_l3,
            "prob_L1": p1,
            "prob_L2": p2,
            "prob_L3": torch.sigmoid(logits_l3),
        }
