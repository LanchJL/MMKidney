from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn

class HierMultiLabelNet(nn.Module):
    """Simple MLP that outputs 3 heads for L1/L2/L3."""
    def __init__(
        self,
        in_dim: int = 768,
        extra_dim: int = 0,
        hidden_dims: Tuple[int, int] = (1024, 512),
        out_dims: Dict[str, int] = None,
        dropout: float = 0.3,
        use_layernorm: bool = True,
    ):
        super().__init__()
        out_dims = out_dims or {"L1": 4, "L2": 5, "L3": 8}
        h1, h2 = hidden_dims
        total_in = in_dim + int(extra_dim)

        layers = [nn.Linear(total_in, h1), nn.GELU(), nn.Dropout(dropout)]
        if use_layernorm:
            layers.append(nn.LayerNorm(h1))
        layers += [nn.Linear(h1, h2), nn.GELU(), nn.Dropout(dropout)]
        if use_layernorm:
            layers.append(nn.LayerNorm(h2))
        self.backbone = nn.Sequential(*layers)
        self.head_l1 = nn.Linear(h2, out_dims["L1"])
        self.head_l2 = nn.Linear(h2, out_dims["L2"])
        self.head_l3 = nn.Linear(h2, out_dims["L3"])
        self._mc_dropout_enabled = False
        self.out_dims = out_dims
        self.extra_dim = int(extra_dim)

    def enable_mc_dropout(self, enabled: bool = True):
        self._mc_dropout_enabled = enabled
        for m in self.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train(enabled)

    def forward(self, x: torch.Tensor, extra: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if self._mc_dropout_enabled and not self.training:
            for m in self.modules():
                if m.__class__.__name__.startswith("Dropout"):
                    m.train(True)

        if self.extra_dim > 0:
            if extra is None:
                raise ValueError("extra_dim > 0 but extra is None")
            x = torch.cat([x, extra], dim=1)

        feat = self.backbone(x)
        return {"L1": self.head_l1(feat), "L2": self.head_l2(feat), "L3": self.head_l3(feat)}


class MultiHeadGatedAttnMIL(nn.Module):
    def __init__(self, in_dim=768, attn_dim=256, proj_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.H = num_heads
        self.fc_v = nn.ModuleList([nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Tanh()) for _ in range(self.H)])
        self.fc_u = nn.ModuleList([nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Sigmoid()) for _ in range(self.H)])
        self.fc_a = nn.ModuleList([nn.Linear(attn_dim, 1) for _ in range(self.H)])
        self.project = nn.Sequential(nn.Linear(in_dim, proj_dim), nn.GELU(), nn.Dropout(dropout))

    def _normalize(self, a):
        return torch.softmax(a, dim=0)

    def forward(self, xs: List[torch.Tensor], return_attn: bool = False):
        bags, attn_list = [], []
        proj_dim = self.project[0].out_features
        output_dim = self.H * proj_dim

        for x in xs:
            if x.shape[0] == 0:
                z = torch.zeros(output_dim, device=x.device)
                bags.append(z)
                if return_attn:
                    attn_list.append(torch.zeros(self.H, 0, device=x.device))
                continue

            V = self.project(x)
            z_heads, ws = [], []
            for h in range(self.H):
                v = self.fc_v[h](x)
                u = self.fc_u[h](x)
                a = self.fc_a[h](v * u).squeeze(-1)
                w = self._normalize(a)
                z_h = (w.unsqueeze(-1) * V).sum(0)
                if z_h.dim() == 0:
                    z_h = z_h.view(1)
                z_heads.append(z_h)
                if return_attn:
                    ws.append(w)

            z = torch.cat(z_heads, dim=0)
            bags.append(z)
            if return_attn:
                attn_list.append(torch.stack(ws, dim=0))

        bags = torch.stack(bags, 0)
        if return_attn:
            return bags, {"attn": attn_list}
        return bags, None


class HierMultiLabelMIL(nn.Module):
    def __init__(
        self,
        proj_dim=256,
        hidden=(512, 256),
        out_dims=None,
        dropout=0.3,
        num_heads=4,
        extra_dim: int = 0,
    ):
        super().__init__()
        out_dims = out_dims or {"L1": 4, "L2": 5, "L3": 8}
        self.mil = MultiHeadGatedAttnMIL(in_dim=768, attn_dim=256, proj_dim=proj_dim, num_heads=num_heads, dropout=dropout)
        h1, h2 = hidden
        in_all = proj_dim * num_heads + int(extra_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_all, h1), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(h1),
            nn.Linear(h1, h2), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(h2),
        )
        self.head_l1 = nn.Linear(h2, out_dims["L1"])
        self.head_l2 = nn.Linear(h2, out_dims["L2"])
        self.head_l3 = nn.Linear(h2, out_dims["L3"])
        self.out_dims = out_dims
        self.extra_dim = int(extra_dim)

    def forward(self, xs: List[torch.Tensor], extra: Optional[torch.Tensor] = None, return_attn: bool = False):
        z, aux = self.mil(xs, return_attn=return_attn)
        if self.extra_dim > 0:
            if extra is None:
                raise ValueError("extra_dim > 0 but extra is None")
            z = torch.cat([z, extra], dim=1)
        feat = self.mlp(z)
        logits = {"L1": self.head_l1(feat), "L2": self.head_l2(feat), "L3": self.head_l3(feat)}
        if return_attn:
            return logits, aux
        return logits, None


class StainFusionAttention(nn.Module):
    """Gated attention across stains (late fusion over stain-level embeddings)."""
    def __init__(self, in_dim: int, attn_dim: int = 256):
        super().__init__()
        self.fc_v = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Tanh())
        self.fc_u = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Sigmoid())
        self.fc_a = nn.Linear(attn_dim, 1)

    def forward(self, z_stains: List[torch.Tensor]):
        if len(z_stains) == 1:
            return z_stains[0], torch.ones(1, device=z_stains[0].device)

        Z = torch.stack(z_stains, dim=0)
        v = self.fc_v(Z)
        u = self.fc_u(Z)
        a = self.fc_a(v * u).squeeze(-1)
        w = torch.softmax(a, dim=0)
        z = (w.unsqueeze(-1) * Z).sum(0)
        return z, w


class HierMultiLabelMIL_MultiStain(nn.Module):
    """
    Multi-stain MIL:
      - per-stain MIL -> stain-level embeddings
      - cross-stain attention fusion -> patient embedding
      - hierarchical multi-label heads
    """
    def __init__(
        self,
        proj_dim=256,
        hidden=(512, 256),
        out_dims=None,
        dropout=0.3,
        num_heads=4,
        extra_dim: int = 0,
    ):
        super().__init__()
        out_dims = out_dims or {"L1": 4, "L2": 5, "L3": 8}
        self.mil_shared = MultiHeadGatedAttnMIL(in_dim=768, attn_dim=256, proj_dim=proj_dim, num_heads=num_heads, dropout=dropout)
        self.fusion = StainFusionAttention(in_dim=proj_dim * num_heads)

        h1, h2 = hidden
        in_all = proj_dim * num_heads + int(extra_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_all, h1), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(h1),
            nn.Linear(h1, h2), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(h2),
        )
        self.head_l1 = nn.Linear(h2, out_dims["L1"])
        self.head_l2 = nn.Linear(h2, out_dims["L2"])
        self.head_l3 = nn.Linear(h2, out_dims["L3"])
        self.out_dims = out_dims
        self.extra_dim = int(extra_dim)

    def _mil_one_stain(self, x: torch.Tensor, return_attn: bool):
        dev = next(self.parameters()).device
        if x.device != dev:
            x = x.to(dev, non_blocking=True)
        z_bag, aux = self.mil_shared([x], return_attn=return_attn)
        z = z_bag[0]
        attn = None
        if return_attn and aux is not None and "attn" in aux:
            attn = aux["attn"][0]
        return z, attn

    def forward(self, xs_ms: List[Dict[str, torch.Tensor]], extra: Optional[torch.Tensor] = None, return_attn: bool = False):
        B = len(xs_ms)
        logits = {"L1": [], "L2": [], "L3": []}
        aux_all = {"patch_attn": [], "stain_weights": [], "stains": []} if return_attn else None

        if self.extra_dim > 0 and extra is None:
            raise ValueError("extra_dim > 0 but extra is None")

        for i in range(B):
            x_dict = xs_ms[i]
            stain_names = sorted(list(x_dict.keys()))
            z_stains, attn_per_stain = [], {}

            for s in stain_names:
                z_s, attn_s = self._mil_one_stain(x_dict[s], return_attn)
                z_stains.append(z_s)
                if return_attn:
                    attn_per_stain[s] = attn_s

            z_fused, w = self.fusion(z_stains)
            if self.extra_dim > 0:
                z_fused = torch.cat([z_fused, extra[i]], dim=0)

            feat = self.mlp(z_fused.unsqueeze(0))
            logits["L1"].append(self.head_l1(feat)[0])
            logits["L2"].append(self.head_l2(feat)[0])
            logits["L3"].append(self.head_l3(feat)[0])

            if return_attn:
                aux_all["patch_attn"].append(attn_per_stain)
                aux_all["stain_weights"].append(w)
                aux_all["stains"].append(stain_names)

        for k in logits:
            logits[k] = torch.stack(logits[k], dim=0)

        if return_attn:
            return logits, aux_all
        return logits, None
