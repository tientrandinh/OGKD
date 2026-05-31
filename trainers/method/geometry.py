"""Class-similarity graph construction for geometry-aware distillation."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def cosine_kernel_from_protos(protos: torch.Tensor, alpha: float = 4.0) -> torch.Tensor:
    """Row-stochastic class graph ``W = softmax(alpha * cos(protos, protos))``.

    Args:
        protos: ``[C, D]`` class text prototypes (normalized internally).
        alpha:  graph sharpness; larger -> sharper neighborhoods.
    Returns:
        ``[C, C]`` row-stochastic similarity matrix.
    """
    P = F.normalize(protos.float(), dim=-1)
    S = P @ P.t()
    return torch.softmax(alpha * S, dim=-1)


@torch.no_grad()
def effective_base_kernel_from_all(W_all: torch.Tensor,
                                   base_idx: torch.Tensor,
                                   lambda_novel: float = 0.1) -> torch.Tensor:
    """Fold novel-class structure back into the base block of the graph:
    ``W_eff = W_bb + lambda_novel * (W_bn @ W_nb)``.

    Args:
        W_all:        ``[C, C]`` graph over all classes.
        base_idx:     indices of the base classes within ``W_all``.
        lambda_novel: weight of the second-order novel back-projection.
    Returns:
        ``[C_base, C_base]`` effective base-class graph.
    """
    device = W_all.device
    base_idx = base_idx.to(device)
    all_idx = torch.arange(W_all.size(0), device=device)
    mask = torch.ones_like(all_idx, dtype=torch.bool)
    mask[base_idx] = False
    novel_idx = all_idx[mask]

    W_bb = W_all.index_select(0, base_idx).index_select(1, base_idx)
    if novel_idx.numel() == 0:
        return W_bb
    W_bn = W_all.index_select(0, base_idx).index_select(1, novel_idx)
    W_nb = W_all.index_select(0, novel_idx).index_select(1, base_idx)
    return W_bb + lambda_novel * (W_bn @ W_nb)
