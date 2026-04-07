from typing import Dict

import torch
import torch.nn as nn

from .heads import HierarchicalHeads
from .lab_encoder import LabEncoder
from .multimodal_fusion import MultiModalFusion
from .tabular_encoder import TabularEncoder
from .wsi_model import WSIEncoder


class MMKidneyTeacherModel(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        proj_dim: int,
        stain_vocab_size: int,
        stain_emb_dim: int,
        tab_in_dim: int,
        lab_in_dim: int,
        tab_dim: int = 128,
        lab_dim: int = 128,
        fused_dim: int = 256,
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
        self.tab_encoder = TabularEncoder(in_dim=tab_in_dim, out_dim=tab_dim)
        self.lab_encoder = LabEncoder(in_dim=lab_in_dim, out_dim=lab_dim)
        self.fusion = MultiModalFusion(img_dim=proj_dim, tab_dim=tab_dim, lab_dim=lab_dim, out_dim=fused_dim)
        self.heads = HierarchicalHeads(in_dim=fused_dim, dims=label_dims, hidden_dim=fused_dim, dropout=dropout)

    def forward(self, batch: Dict):
        img_out = self.wsi_encoder(batch["wsi"])
        img_repr = img_out["patient_repr"]

        tab_repr = None
        if batch.get("tabular") is not None:
            tab_repr = self.tab_encoder(batch["tabular"], batch.get("tabular_mask"))

        lab_repr = None
        if batch.get("lab") is not None:
            lab_repr = self.lab_encoder(batch["lab"], batch.get("lab_mask"))

        fused_repr, gates = self.fusion(
            img_repr=img_repr,
            tab_repr=tab_repr,
            lab_repr=lab_repr,
            modality_mask=batch.get("modality_mask"),
        )
        pred = self.heads(fused_repr)
        pred.update(
            {
                "img_repr": img_repr,
                "tab_repr": tab_repr,
                "lab_repr": lab_repr,
                "fused_repr": fused_repr,
                "gates": gates,
                "aux": img_out.get("aux", []),
            }
        )
        return pred
