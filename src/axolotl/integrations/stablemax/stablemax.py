"""StableMax loss implementation."""

import torch
import torch.nn.functional as F


def stablemax_fn(x):
    """Numerically stable alternative to softmax."""
    s = torch.where(x >= 0, x + 1, 1 / (1 - x))
    return s / s.sum(dim=-1, keepdim=True)


def stablemax_cross_entropy(
    logits,
    target,
    weight=None,
    ignore_index=-100,
    size_average=None,
    reduce=None,
    reduction="mean",
    label_smoothing=0.0,
):
    """Cross-entropy loss using StableMax instead of softmax."""
    del size_average, reduce
    probs = stablemax_fn(logits)
    log_probs = torch.log(probs + 1e-12)

    if target.dim() == logits.dim():
        targets_one_hot = target.float()
        valid_mask = torch.ones(target.shape[0], dtype=torch.bool, device=target.device)
    else:
        valid_mask = target != ignore_index
        num_classes = logits.shape[-1]
        targets_one_hot = torch.zeros_like(logits)
        valid_targets = target[valid_mask]
        if valid_targets.numel() > 0:
            targets_one_hot[valid_mask] = F.one_hot(valid_targets, num_classes).float()

    if label_smoothing > 0.0:
        num_classes = logits.shape[-1]
        uniform_dist = torch.ones_like(targets_one_hot) / num_classes
        targets_one_hot = (1.0 - label_smoothing) * targets_one_hot + label_smoothing * uniform_dist

    loss = -(targets_one_hot * log_probs).sum(dim=-1)

    if weight is not None:
        if target.dim() == logits.dim():
            class_weights = (targets_one_hot * weight.unsqueeze(0)).sum(dim=-1)
        else:
            class_weights = torch.ones_like(loss)
            class_weights[valid_mask] = weight[target[valid_mask]]
        loss = loss * class_weights

    if ignore_index != -100 or not valid_mask.all():
        loss = loss[valid_mask]

    if reduction == "none":
        if not valid_mask.all():
            full_loss = torch.zeros(valid_mask.shape[0], dtype=loss.dtype, device=loss.device)
            full_loss[valid_mask] = loss
            return full_loss
        return loss
    if reduction == "mean":
        return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=logits.device)
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Invalid reduction mode: {reduction}")
