from typing import Optional, Tuple

import torch
import torch.nn as nn


class GatedAttentionPool(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.25):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh())
        self.u = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Sigmoid())
        self.w = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: [N, D]
        if x.ndim != 2:
            raise ValueError(f"Expected [N, D], got {tuple(x.shape)}")
        a = self.w(self.v(x) * self.u(x)).squeeze(-1)
        a = torch.softmax(a, dim=0)
        z = (a.unsqueeze(-1) * self.drop(x)).sum(dim=0)
        if return_attn:
            return z, a
        return z, None


class StainBagEncoder(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        proj_dim: int,
        stain_vocab_size: int,
        stain_emb_dim: int,
        dropout: float = 0.25,
        use_he_adapter: bool = True,
        he_stain_id: int = 0,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = GatedAttentionPool(in_dim=proj_dim, hidden_dim=proj_dim, dropout=dropout)
        self.stain_emb = nn.Embedding(stain_vocab_size, stain_emb_dim)
        self.out = nn.Sequential(
            nn.Linear(proj_dim + stain_emb_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.use_he_adapter = use_he_adapter
        self.he_stain_id = int(he_stain_id)
        self.he_adapter = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.GELU()) if use_he_adapter else nn.Identity()
        self.opt_adapter = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.GELU()) if use_he_adapter else nn.Identity()

    def forward(self, bag_feats: torch.Tensor, stain_id: int, coords=None, return_attn: bool = False):
        x = self.proj(bag_feats)
        if self.use_he_adapter:
            if int(stain_id) == self.he_stain_id:
                x = self.he_adapter(x)
            else:
                x = self.opt_adapter(x)

        pooled, attn = self.pool(x, return_attn=return_attn)
        sid = torch.tensor([max(int(stain_id), 0)], device=bag_feats.device)
        se = self.stain_emb(sid).squeeze(0)
        z = self.out(torch.cat([pooled, se], dim=0))
        return z, attn
