from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class CoxSurvivalHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # risk score, higher means higher hazard
        return self.net(x).squeeze(-1)


class DiscreteTimeSurvivalHead(nn.Module):
    def __init__(self, in_dim: int, n_bins: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.n_bins = int(n_bins)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.n_bins),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # logits for per-bin hazard
        return self.net(x)


def cox_ph_loss(risk: torch.Tensor, event: torch.Tensor, time_days: torch.Tensor) -> torch.Tensor:
    """
    Negative partial log-likelihood (Breslow-style risk set).
    risk: [B], event/time: [B]
    """
    if risk.ndim != 1:
        risk = risk.view(-1)
    e = event.float().view(-1)
    t = time_days.float().view(-1)
    r = risk.view(-1)

    # sort descending time so risk set is prefix
    order = torch.argsort(t, descending=True)
    r = r[order]
    e = e[order]

    log_cum = torch.logcumsumexp(r, dim=0)
    # only event terms contribute
    n_event = torch.clamp(e.sum(), min=1.0)
    loss = -((r - log_cum) * e).sum() / n_event
    return loss


def _time_to_bin(time_days: torch.Tensor, bins_days: List[int]) -> torch.Tensor:
    # bins_days are upper boundaries for K-1 bins, final bin is >last
    dev = time_days.device
    b = torch.tensor(bins_days, dtype=torch.float32, device=dev)
    # bucketize returns index in [0..len(bins)]
    return torch.bucketize(time_days.float(), b)


def discrete_time_nll(hazard_logits: torch.Tensor, event: torch.Tensor, time_days: torch.Tensor, bins_days: List[int]) -> torch.Tensor:
    """
    Discrete-time survival NLL.
    """
    if hazard_logits.ndim != 2:
        raise ValueError(f"hazard_logits must be [B,K], got {tuple(hazard_logits.shape)}")
    h = torch.sigmoid(hazard_logits)
    # stabilize for log
    h = torch.clamp(h, min=1e-7, max=1.0 - 1e-7)

    B, K = h.shape
    bin_idx = _time_to_bin(time_days, bins_days)  # [B], in [0..K-1]
    if int(bin_idx.max().item()) >= K:
        raise ValueError(f"Found bin index >= K ({int(bin_idx.max().item())} >= {K}). Check bins.")

    e = event.float().view(-1)
    losses = []
    for i in range(B):
        k = int(bin_idx[i].item())
        # survive all previous bins
        if k > 0:
            s_prev = torch.log(1.0 - h[i, :k]).sum()
        else:
            s_prev = h[i, :0].sum() * 0.0
        if e[i] > 0.5:
            # event in bin k
            ll = s_prev + torch.log(h[i, k])
        else:
            # censored at/after bin k: survive through k
            ll = s_prev + torch.log(1.0 - h[i, k])
        losses.append(-ll)
    return torch.stack(losses).mean()


def survival_probs_from_logits(hazard_logits: torch.Tensor) -> torch.Tensor:
    """
    Returns survival at end of each bin, shape [B, K]
    """
    h = torch.sigmoid(hazard_logits)
    h = torch.clamp(h, min=1e-7, max=1.0 - 1e-7)
    s = torch.cumprod(1.0 - h, dim=1)
    return s


def horizon_risk_from_logits(hazard_logits: torch.Tensor, bins_days: List[int], horizon_days: int) -> torch.Tensor:
    s = survival_probs_from_logits(hazard_logits)
    # map horizon to bin index
    b = torch.tensor(bins_days, dtype=torch.float32, device=s.device)
    idx = int(torch.bucketize(torch.tensor([float(horizon_days)], device=s.device), b)[0].item())
    idx = min(max(idx, 0), s.shape[1] - 1)
    return 1.0 - s[:, idx]


def pack_survival_outputs(
    head_type: str,
    fused_repr: torch.Tensor,
    cox_head: Optional[CoxSurvivalHead],
    discrete_head: Optional[DiscreteTimeSurvivalHead],
) -> Dict[str, torch.Tensor]:
    if head_type == "cox":
        if cox_head is None:
            raise ValueError("cox_head is None")
        risk = cox_head(fused_repr)
        return {"risk": risk}
    if head_type == "discrete":
        if discrete_head is None:
            raise ValueError("discrete_head is None")
        logits = discrete_head(fused_repr)
        return {"hazard_logits": logits}
    raise ValueError(f"Unknown head_type: {head_type}")
