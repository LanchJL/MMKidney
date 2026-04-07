from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalConsistencyLoss(nn.Module):
    def __init__(self, mapping_L2_to_L1: Optional[Dict[int, list]] = None, mapping_L3_to_L2: Optional[Dict[int, list]] = None):
        super().__init__()
        self.mapping_L2_to_L1 = mapping_L2_to_L1
        self.mapping_L3_to_L2 = mapping_L3_to_L2

    def forward(self, pred: Dict[str, torch.Tensor]):
        if self.mapping_L2_to_L1 is None or self.mapping_L3_to_L2 is None:
            return pred["logits_L1"].new_tensor(0.0)

        p1 = torch.sigmoid(pred["logits_L1"])
        p2 = torch.sigmoid(pred["logits_L2"])
        p3 = torch.sigmoid(pred["logits_L3"])

        loss = 0.0
        count = 0
        for p_idx, children in self.mapping_L2_to_L1.items():
            parent = p1[:, p_idx].unsqueeze(1)
            child = p2[:, children]
            loss = loss + F.relu(child - parent).mean()
            count += 1

        for p_idx, children in self.mapping_L3_to_L2.items():
            parent = p2[:, p_idx].unsqueeze(1)
            child = p3[:, children]
            loss = loss + F.relu(child - parent).mean()
            count += 1

        if count == 0:
            return p1.new_tensor(0.0)
        return loss / count
