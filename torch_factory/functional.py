import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, List

from config import CLASSES

__all__ = ["LabelSmoothingCrossEntropy", "FocalLoss", 
           "LDAMLoss", "LDAMFocalLoss", "FocalLossWeighted"]

def _focal_loss_term(loss_values: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Compute the focal loss term for given per-sample loss values.

    Args:
        loss_values: Tensor of per-sample cross-entropy losses.
        gamma: Focusing parameter; >= 0.

    Returns:
        The mean focal loss over the batch.
    """
    # Convert negative log-probabilities back to probabilities
    prob = torch.exp(-loss_values)
    # Scale loss by (1 - p)^gamma
    focal_factor = (1 - prob) ** gamma
    return (focal_factor * loss_values).mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Args:
        eps: Smoothing factor in [0, 1].
        num_classes: Number of classes.
    """

    def __init__(self, eps: float = 0.1, num_classes: int = len(CLASSES)):
        super().__init__()
        if not 0.0 <= eps <= 1.0:
            raise ValueError(f"eps must be in [0,1], got {eps}")
        self.eps = eps
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the label-smoothed cross-entropy.

        Args:
            logits: Raw model outputs of shape (batch_size, num_classes).
            targets: Ground truth class indices of shape (batch_size,).

        Returns:
            Scalar loss.
        """
        # Log-probabilities
        log_probs = F.log_softmax(logits, dim=1)
        # Create one-hot targets and apply smoothing
        with torch.no_grad():
            true_dist = F.one_hot(targets, self.num_classes).float()
            true_dist = true_dist * (1 - self.eps) + self.eps / self.num_classes

        # Compute per-sample loss and average
        loss = -torch.sum(true_dist * log_probs, dim=1)
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal loss for classification, wrapping cross-entropy.

    Args:
        weight: Optional tensor of class weights.
        gamma: Focusing parameter; >= 0.
    """

    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 0.0):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute per-sample CE loss without reduction
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        # Apply focal term
        return _focal_loss_term(ce_loss, self.gamma)


class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin (LDAM) Loss.

    Args:
        cls_counts: List of class sample counts.
        max_margin: Maximum margin.
        weight: Optional tensor of class weights.
        scale: Logit scaling factor.
    """

    def __init__(
        self,
        cls_counts: Sequence[int],
        max_margin: float = 0.5,
        weight: Optional[torch.Tensor] = None,
        scale: float = 30.0,
    ):
        super().__init__()
        # Compute per-class margins
        margins = 1.0 / np.sqrt(np.sqrt(np.array(cls_counts, dtype=float)))
        margins = margins * (max_margin / np.max(margins))
        m_list = torch.tensor(margins, dtype=torch.float32, device='cuda')
        self.m_list  = m_list

        if scale <= 0:
            raise ValueError("scale must be > 0")
        self.scale = scale
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute LDAM loss.

        Args:
            logits: Model outputs, shape (batch_size, num_classes).
            targets: Ground truth indices, shape (batch_size,).

        Returns:
            Scalar loss.
        """
        # Create mask for target classes
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, targets.data.view(-1, 1), 1)

        # Gather margin for each sample
        mask_float = mask.detach().to(dtype=torch.float32, device='cuda')
        margins = torch.matmul(self.m_list[None, :], mask_float.transpose(0,1))
        margins = margins.view((-1,1))

        # Subtract margin from logits of target class
        mask_margin = logits - margins
        adjusted_logits = torch.where(mask, mask_margin, logits)
        return F.cross_entropy(self.scale * adjusted_logits, targets, weight=self.weight)


class LDAMFocalLoss(nn.Module):
    """
    Hybrid of LDAM and Focal loss.

    Args:
        cls_counts: Class sample counts.
        weight: Optional class weights.
        max_margin: Maximum margin for LDAM.
        scale: Logit scaling factor.
        gamma: Focusing parameter for focal loss.
        alpha: Weight for LDAM component.
        beta: Weight for focal component.
    """

    def __init__(
        self,
        cls_counts: Sequence[int],
        weight: Optional[torch.Tensor] = None,
        max_margin: float = 0.5,
        scale: float = 30.0,
        gamma: float = 1.0,
        alpha: float = 0.8,
        beta: float = 0.8,
    ):
        super().__init__()
        self.ldam = LDAMLoss(cls_counts, max_margin, weight, scale)
        self.focal = FocalLoss(weight, gamma)
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ldam_loss = self.ldam(logits, targets)
        focal_loss = self.focal(logits, targets)
        return self.alpha * ldam_loss + self.beta * focal_loss