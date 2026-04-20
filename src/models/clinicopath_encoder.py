from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class GatedBranchFusion(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_branch: int):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(n_branch)])
        self.gate = nn.Sequential(nn.Linear(out_dim * n_branch, out_dim), nn.GELU(), nn.Linear(out_dim, n_branch))

    def forward(self, xs, present_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # xs: list[Tensor[B, D]]
        z = [self.proj[i](x) for i, x in enumerate(xs)]
        cat = torch.cat(z, dim=1)
        logits = self.gate(cat)
        if present_mask is not None:
            logits = logits.masked_fill(present_mask <= 0, -1e9)
        g = torch.softmax(logits, dim=1)
        fused = sum(g[:, i : i + 1] * z[i] for i in range(len(z)))
        return fused, g


class FeatureTokenizer(nn.Module):
    """
    FT-Transformer style tokenization:
    each scalar feature x_i -> token_i = w_i * x_i + b_i + feature_emb_i + group_emb_g(i).
    """

    def __init__(self, n_features: int, d_token: int, n_groups: int = 9):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        self.feature_emb = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.group_emb = nn.Embedding(n_groups, d_token)
        self.mask_emb = nn.Parameter(torch.zeros(1, 1, d_token))

    def forward(self, x: torch.Tensor, feature_mask: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        # x [B,F], feature_mask [B,F], group_ids [B,F] or [F]
        if group_ids.ndim == 1:
            gid = group_ids.unsqueeze(0).expand(x.shape[0], -1)
        else:
            gid = group_ids
        tok = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        tok = tok + self.feature_emb.unsqueeze(0) + self.group_emb(gid.long())
        miss = (feature_mask <= 0).unsqueeze(-1)
        tok = torch.where(miss, self.mask_emb.expand_as(tok), tok)
        return tok


class BranchPool(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(n_layers)
            ]
        )
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True)

    def forward(self, x: torch.Tensor, present: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x [B,F,D], present [B,F] 1 means valid
        key_padding_mask = present <= 0
        for lyr in self.layers:
            x = lyr(x, src_key_padding_mask=key_padding_mask)
        q = self.query.expand(x.shape[0], -1, -1)
        pooled, attn = self.attn(q, x, x, key_padding_mask=key_padding_mask, need_weights=True)
        return pooled[:, 0, :], attn[:, 0, :]


class TimelineEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + 1, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, m: torch.Tensor):
        return self.net(torch.cat([x, m], dim=1))


class TreatmentEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + 1, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, m: torch.Tensor):
        return self.net(torch.cat([x, m], dim=1))


class HybridClinicopathEncoder(nn.Module):
    """
    Input from batch:
      tabular [B,F], tabular_mask [B,F], tabular_group_ids [B,F] (or [F])
    Group ids:
      0 structured, 1 banff, 2 timeline, 3 treatment, 4 text_stats/keywords,
      5 text_hash, 6 qwen_global, 7 medbert_local, 8 other
    """

    def __init__(self, in_dim: int, out_dim: int = 128, d_token: int = 128, n_groups: int = 9, dropout: float = 0.2):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features=in_dim, d_token=d_token, n_groups=n_groups)
        self.pool = BranchPool(d_model=d_token, n_heads=4, n_layers=2, dropout=dropout)
        self.branch_proj = nn.Linear(d_token, out_dim)
        self.timeline_encoder = TimelineEncoder(in_dim=d_token, out_dim=out_dim)
        self.treatment_encoder = TreatmentEncoder(in_dim=d_token, out_dim=out_dim)
        self.fusion = GatedBranchFusion(in_dim=out_dim, out_dim=out_dim, n_branch=6)

    @staticmethod
    def _group_mask(g: torch.Tensor, group_ids) -> torch.Tensor:
        m = torch.zeros_like(g, dtype=torch.float32)
        for gid in group_ids:
            m = torch.maximum(m, (g == gid).float())
        return m

    @staticmethod
    def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # x [B,F,D], m [B,F]
        den = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (x * m.unsqueeze(-1)).sum(dim=1) / den

    def forward(self, x: torch.Tensor, m: torch.Tensor, group_ids: torch.Tensor):
        tok = self.tokenizer(x, m, group_ids)  # [B,F,D]
        g = group_ids if group_ids.ndim == 2 else group_ids.unsqueeze(0).expand(x.shape[0], -1)
        valid = m > 0

        # Structured / Banff branch via attention pooling on selected tokens
        sm = self._group_mask(g, [0, 8]) * valid.float()
        bm = self._group_mask(g, [1]) * valid.float()
        tm = self._group_mask(g, [2]) * valid.float()
        trm = self._group_mask(g, [3]) * valid.float()
        gtxt = self._group_mask(g, [6, 5, 4]) * valid.float()
        ltxt = self._group_mask(g, [7, 5, 4]) * valid.float()

        s_repr = self.branch_proj(self.pool(tok, sm)[0])
        b_repr = self.branch_proj(self.pool(tok, bm)[0])
        t_raw = self._masked_mean(tok, tm)
        tr_raw = self._masked_mean(tok, trm)
        gtxt_repr = self.branch_proj(self.pool(tok, gtxt)[0])
        ltxt_repr = self.branch_proj(self.pool(tok, ltxt)[0])
        t_repr = self.timeline_encoder(t_raw, (tm.sum(dim=1, keepdim=True) > 0).float())
        tr_repr = self.treatment_encoder(tr_raw, (trm.sum(dim=1, keepdim=True) > 0).float())

        branch_present = torch.stack(
            [
                (sm.sum(dim=1) > 0).float(),
                (bm.sum(dim=1) > 0).float(),
                (tm.sum(dim=1) > 0).float(),
                (trm.sum(dim=1) > 0).float(),
                (gtxt.sum(dim=1) > 0).float(),
                (ltxt.sum(dim=1) > 0).float(),
            ],
            dim=1,
        )
        fused, gates = self.fusion([s_repr, b_repr, t_repr, tr_repr, gtxt_repr, ltxt_repr], present_mask=branch_present)
        aux = {
            "clinicopath_branch_gates": gates,
            "clinicopath_branch_present": branch_present,
        }
        return fused, aux
