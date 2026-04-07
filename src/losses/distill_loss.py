from typing import Dict

import torch
import torch.nn.functional as F


def compute_supervised_loss(pred: Dict, targets: Dict, asl_fn, hier_fn=None, w=(1.0, 1.0, 1.2), lambda_cons=0.1):
    l1 = asl_fn(pred["logits_L1"], targets["L1"])
    l2 = asl_fn(pred["logits_L2"], targets["L2"])
    l3 = asl_fn(pred["logits_L3"], targets["L3"])
    cons = pred["logits_L1"].new_tensor(0.0)
    if hier_fn is not None:
        cons = hier_fn(pred)
    total = w[0] * l1 + w[1] * l2 + w[2] * l3 + lambda_cons * cons
    return total, {"loss_l1": float(l1.item()), "loss_l2": float(l2.item()), "loss_l3": float(l3.item()), "loss_cons": float(cons.item())}


def distill_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 2.0):
    ps = F.log_softmax(student_logits / T, dim=-1)
    pt = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(ps, pt, reduction="batchmean") * (T * T)


def compute_distill_loss(student_out: Dict, teacher_out: Dict, targets: Dict, asl_fn, hier_fn=None, T=2.0, alpha=1.0, beta=0.5, gamma=1.0):
    sup, sup_log = compute_supervised_loss(student_out, targets, asl_fn=asl_fn, hier_fn=hier_fn)
    kl = distill_kl(student_out["logits_L3"], teacher_out["logits_L3"], T=T)
    # fallback alignment target
    t_repr = teacher_out.get("fused_repr", teacher_out.get("img_repr"))
    s_repr = student_out["patient_repr"]
    if s_repr.shape[1] != t_repr.shape[1]:
        # zero-pad/crop fallback if no projection used externally
        if s_repr.shape[1] < t_repr.shape[1]:
            pad = t_repr.shape[1] - s_repr.shape[1]
            s_repr = torch.cat([s_repr, torch.zeros(s_repr.shape[0], pad, device=s_repr.device)], dim=1)
        else:
            s_repr = s_repr[:, : t_repr.shape[1]]
    mse = F.mse_loss(s_repr, t_repr.detach())
    total = gamma * sup + alpha * kl + beta * mse
    logs = {**sup_log, "loss_kl": float(kl.item()), "loss_mse": float(mse.item())}
    return total, logs
