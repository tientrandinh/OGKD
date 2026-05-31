"""Geometry-aware distillation losses: GAD (global) and LGD (label-guided patch)."""
import torch
import torch.nn.functional as F


def loss_gad(student_logits: torch.Tensor,   # [B, C]
             teacher_logits: torch.Tensor,   # [B, C]
             W: torch.Tensor,                 # [C, C] class-similarity graph
             *,
             gamma: float = 0.3) -> torch.Tensor:
    """Global Geometry-Aware Distillation (GAD).

    KL between the student global distribution and a graph-smoothed teacher
    distribution, where the teacher log-probabilities are diffused along ``W``
    with strength ``gamma``.
    """
    q_log = F.log_softmax(teacher_logits.detach(), dim=-1)
    q_smooth_log = (1.0 - gamma) * q_log + gamma * (q_log @ W)
    logp = F.log_softmax(student_logits, dim=-1)
    return F.kl_div(logp, q_smooth_log, reduction="sum", log_target=True) / logp.numel()


def loss_lgd(student_logits: torch.Tensor,   # [B, N, C] patch-class logits
             teacher_logits: torch.Tensor,   # [B, N, C] patch-class logits
             W: torch.Tensor,                 # [C, C] class-similarity graph
             labels: torch.Tensor,            # [B]
             *,
             gamma: float = 0.3) -> torch.Tensor:
    """Label-Guided Geometry Distillation (LGD).

    Over the selected patch tokens, smooth the teacher patch-class distribution
    along ``W``, gather the ground-truth class channel for both student and
    teacher, and take the patch-wise KL.
    """
    B, N, _ = student_logits.shape
    q_cls = F.log_softmax(teacher_logits.detach(), dim=-1)
    q_smooth = (1.0 - gamma) * q_cls + gamma * torch.matmul(q_cls, W)
    idx = labels.view(B, 1, 1).expand(B, N, 1)
    p_cls = F.log_softmax(student_logits, dim=-1)
    p_cls = torch.gather(p_cls, dim=-1, index=idx).squeeze(-1)
    q_smooth = torch.gather(q_smooth, dim=-1, index=idx).squeeze(-1)
    return F.kl_div(p_cls, q_smooth, reduction="sum", log_target=True) / p_cls.numel()
