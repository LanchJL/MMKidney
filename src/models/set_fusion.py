from typing import Optional, Tuple

import torch
import torch.nn as nn


class StainSetFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=dim * 4,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.attn_pool = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)

    def forward(self, stain_tokens: torch.Tensor, stain_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # stain_tokens: [S, H] or [B, S, H]
        if stain_tokens.ndim == 2:
            x = stain_tokens.unsqueeze(0)
        else:
            x = stain_tokens

        for lyr in self.layers:
            x = lyr(x)

        q = self.query.expand(x.shape[0], -1, -1)
        key_padding_mask = None
        if stain_mask is not None:
            # stain_mask=1 for present, convert to key padding (True means ignore)
            if stain_mask.ndim == 1:
                stain_mask = stain_mask.unsqueeze(0)
            key_padding_mask = stain_mask <= 0

        pooled, attn = self.attn_pool(q, x, x, key_padding_mask=key_padding_mask, need_weights=True)
        pooled = pooled[:, 0, :]  # [B, H]
        if stain_tokens.ndim == 2:
            pooled = pooled[0]
            attn = attn[0, 0]
        return pooled, attn
