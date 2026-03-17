import torch
import torch.nn.functional as F


def focal_cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    gamma: float = 0.0,
    ignore_index: int = -100,
    class_dim: int = 1,
) -> torch.Tensor:
    logits = logits.float()
    target = target.long()
    gamma = float(gamma)
    if gamma < 0.0:
        raise ValueError(f"focal gamma must be >= 0, got {gamma}")

    log_probs = F.log_softmax(logits, dim=class_dim)
    moved = log_probs.movedim(class_dim, -1)
    valid = target != int(ignore_index)
    if not bool(valid.any()):
        return logits.sum() * 0.0

    target_safe = target.clamp_min(0)
    true_log_probs = moved.gather(dim=-1, index=target_safe.unsqueeze(-1)).squeeze(-1)
    true_log_probs = true_log_probs[valid]
    ce = -true_log_probs
    focal_factor = (1.0 - true_log_probs.exp()).pow(gamma)

    sample_weights = None
    if class_weights is not None:
        sample_weights = class_weights[target_safe[valid]].float()

    loss = focal_factor * ce
    if sample_weights is not None:
        loss = loss * sample_weights
        denom = sample_weights.sum().clamp(min=1.0)
    else:
        denom = loss.new_tensor(max(1, int(valid.sum().item())), dtype=loss.dtype)
    return loss.sum() / denom
