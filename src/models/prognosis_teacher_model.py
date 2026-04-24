from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .clinicopath_encoder import HybridClinicopathEncoder
from .lab_encoder import LabEncoder
from .multimodal_fusion import MultiModalFusion
from .prognosis_heads import CoxSurvivalHead, DiscreteTimeSurvivalHead, pack_survival_outputs
from .tabular_encoder import TabularEncoder
from .wsi_model import WSIEncoder


class MMKidneyPrognosisTeacher(nn.Module):
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
        num_heads: int = 4,
        num_set_layers: int = 2,
        dropout: float = 0.25,
        he_stain_id: int = 0,
        use_he_adapter: bool = True,
        head_type: str = "cox",
        n_bins: int = 4,
        use_clinicopath_encoder: bool = True,
    ):
        super().__init__()
        self.head_type = head_type
        self.use_clinicopath_encoder = bool(use_clinicopath_encoder)
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
        self.clinicopath_encoder = HybridClinicopathEncoder(in_dim=tab_in_dim, out_dim=tab_dim)
        self.lab_encoder = LabEncoder(in_dim=lab_in_dim, out_dim=lab_dim)
        self.fusion = MultiModalFusion(img_dim=proj_dim, tab_dim=tab_dim, lab_dim=lab_dim, out_dim=fused_dim)

        self.cox_head = CoxSurvivalHead(in_dim=fused_dim, hidden_dim=fused_dim, dropout=dropout) if head_type == "cox" else None
        self.discrete_head = (
            DiscreteTimeSurvivalHead(in_dim=fused_dim, n_bins=n_bins, hidden_dim=fused_dim, dropout=dropout)
            if head_type == "discrete"
            else None
        )

    def forward(self, batch: Dict):
        img_out = self.wsi_encoder(batch["wsi"])
        img_repr = img_out["patient_repr"]

        tab_repr = None
        cp_aux = None
        if batch.get("tabular") is not None:
            if self.use_clinicopath_encoder and batch.get("tabular_group_ids") is not None:
                tab_repr, cp_aux = self.clinicopath_encoder(
                    batch["tabular"],
                    batch.get("tabular_mask"),
                    batch["tabular_group_ids"],
                )
            else:
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
        out = pack_survival_outputs(
            head_type=self.head_type,
            fused_repr=fused_repr,
            cox_head=self.cox_head,
            discrete_head=self.discrete_head,
        )
        out.update(
            {
                "img_repr": img_repr,
                "tab_repr": tab_repr,
                "lab_repr": lab_repr,
                "fused_repr": fused_repr,
                "gates": gates,
                "aux": img_out.get("aux", []),
                "clinicopath_aux": cp_aux,
            }
        )
        return out
