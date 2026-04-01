from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class HierMultiLabelLoss(nn.Module):
    """
    BCE for L1/L2/L3 + hierarchical consistency penalty:
    child probability should not exceed parent probability.
    """
    def __init__(self, pos_weight: Optional[Dict[str, torch.Tensor]] = None, lambda_hier: float = 0.2):
        super().__init__()
        self.pos_weight = pos_weight or {}
        self.lambda_hier = float(lambda_hier)
        self._bce = {}
        for lv in ["L1", "L2", "L3"]:
            pw = self.pos_weight.get(lv, None)
            self._bce[lv] = nn.BCEWithLogitsLoss(pos_weight=pw) if pw is not None else nn.BCEWithLogitsLoss()

    def forward(self, logits: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], parent_map: Dict[str, Dict[int, list]]):
        loss_l1 = self._bce["L1"](logits["L1"], targets["L1"])
        loss_l2 = self._bce["L2"](logits["L2"], targets["L2"])
        loss_l3 = self._bce["L3"](logits["L3"], targets["L3"])
        total = loss_l1 + loss_l2 + loss_l3

        if self.lambda_hier <= 0.0:
            return total

        p = {k: torch.sigmoid(v) for k, v in logits.items()}
        hier_loss = 0.0

        for p_idx, child_list in parent_map.get("L1<-L2", {}).items():
            parent_p = p["L1"][:, p_idx].unsqueeze(1)
            child_p = p["L2"][:, child_list]
            hier_loss = hier_loss + F.relu(child_p - parent_p).mean()

        for p_idx, child_list in parent_map.get("L2<-L3", {}).items():
            parent_p = p["L2"][:, p_idx].unsqueeze(1)
            child_p = p["L3"][:, child_list]
            hier_loss = hier_loss + F.relu(child_p - parent_p).mean()

        return total + self.lambda_hier * hier_loss


def attention_diversity_loss(attn_list, lambda_div: float = 0.05):
    """Encourage multi-head attention diversity."""
    loss = 0.0
    count = 0
    for attn in attn_list:
        attn = F.normalize(attn, p=2, dim=1)
        sim = torch.matmul(attn, attn.T)
        mask = torch.eye(sim.size(0), device=sim.device).bool()
        sim = sim[~mask]
        loss += sim.mean()
        count += 1
    if count > 0:
        loss = loss / count
    return lambda_div * loss
