from typing import Dict, List

import torch
import torch.nn as nn

from .heads import HierarchicalHeads
from .mil import StainBagEncoder
from .set_fusion import StainSetFusion


class WSIEncoder(nn.Module):
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
    ):
        super().__init__()
        self.stain_encoder = StainBagEncoder(
            feat_dim=feat_dim,
            proj_dim=proj_dim,
            stain_vocab_size=stain_vocab_size,
            stain_emb_dim=stain_emb_dim,
            dropout=dropout,
            use_he_adapter=use_he_adapter,
            he_stain_id=he_stain_id,
        )
        self.set_fusion = StainSetFusion(dim=proj_dim, num_heads=num_heads, num_layers=num_set_layers, dropout=dropout)

    def forward(self, patient_wsi: List[Dict]):
        patient_reprs = []
        stain_reprs_all = []
        aux = []

        for patient in patient_wsi:
            stain_reprs = []
            stain_names = []
            patch_attn = {}
            for stain in patient["stains"]:
                x = stain["features"]
                sid = int(stain["id"])
                z, attn = self.stain_encoder(x, stain_id=sid, coords=stain.get("coords"), return_attn=True)
                stain_reprs.append(z)
                stain_names.append(stain["name"])
                patch_attn[stain["name"]] = attn

            S = torch.stack(stain_reprs, dim=0)
            patient_repr, stain_fusion_attn = self.set_fusion(S)
            patient_reprs.append(patient_repr)
            stain_reprs_all.append(S)
            aux.append(
                {
                    "stain_names": stain_names,
                    "patch_attn": patch_attn,
                    "stain_fusion_attn": stain_fusion_attn,
                }
            )

        patient_reprs = torch.stack(patient_reprs, dim=0)
        return {
            "patient_repr": patient_reprs,
            "stain_reprs": stain_reprs_all,
            "aux": aux,
        }


class MMKidneyWSIModel(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        proj_dim: int,
        stain_vocab_size: int,
        stain_emb_dim: int,
        label_dims=(4, 5, 8),
        num_heads: int = 4,
        num_set_layers: int = 2,
        dropout: float = 0.25,
        he_stain_id: int = 0,
        use_he_adapter: bool = True,
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
        self.heads = HierarchicalHeads(in_dim=proj_dim, dims=label_dims, hidden_dim=proj_dim, dropout=dropout)

    def forward(self, batch: Dict):
        img_out = self.wsi_encoder(batch["wsi"])
        pred = self.heads(img_out["patient_repr"])
        return {**img_out, **pred}
