import torch
import torch.nn as nn


class TabularEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(128, 128), out_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2, h1),
            nn.LayerNorm(h1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h2, out_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        if mask is None:
            mask = torch.ones_like(x)
        return self.net(torch.cat([x, mask], dim=1))
