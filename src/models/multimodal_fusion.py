from typing import Optional, Tuple

import torch
import torch.nn as nn


class MultiModalFusion(nn.Module):
    def __init__(self, img_dim: int, tab_dim: int, lab_dim: int, out_dim: int):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, out_dim)
        self.tab_proj = nn.Linear(tab_dim, out_dim)
        self.lab_proj = nn.Linear(lab_dim, out_dim)
        self.gate = nn.Sequential(
            nn.Linear(out_dim * 3, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, 3),
        )

    def forward(
        self,
        img_repr: torch.Tensor,
        tab_repr: Optional[torch.Tensor] = None,
        lab_repr: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = img_repr.shape[0]
        dev = img_repr.device

        ip = self.img_proj(img_repr)
        tp = torch.zeros_like(ip) if tab_repr is None else self.tab_proj(tab_repr)
        lp = torch.zeros_like(ip) if lab_repr is None else self.lab_proj(lab_repr)

        g_logits = self.gate(torch.cat([ip, tp, lp], dim=1))

        if modality_mask is None:
            modality_mask = torch.ones((b, 3), device=dev)
        masked_logits = g_logits.masked_fill(modality_mask <= 0, -1e9)
        g = torch.softmax(masked_logits, dim=1)

        fused = g[:, 0:1] * ip + g[:, 1:2] * tp + g[:, 2:3] * lp
        return fused, g
