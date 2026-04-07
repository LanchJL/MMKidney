from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillWrapper(nn.Module):
    def __init__(self, student: nn.Module, teacher: nn.Module, student_dim: int, teacher_dim: int):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.student_proj = nn.Linear(student_dim, teacher_dim)

    @staticmethod
    def distill_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 2.0):
        ps = F.log_softmax(student_logits / T, dim=-1)
        pt = F.softmax(teacher_logits / T, dim=-1)
        return F.kl_div(ps, pt, reduction="batchmean") * (T * T)

    def forward(self, batch: Dict, use_teacher: bool = True, T: float = 2.0):
        out_s = self.student(batch)
        out_t: Optional[Dict] = None
        if use_teacher:
            with torch.no_grad():
                out_t = self.teacher(batch)

        return {
            "student": out_s,
            "teacher": out_t,
            "student_proj_repr": self.student_proj(out_s["patient_repr"]),
            "temperature": T,
        }
