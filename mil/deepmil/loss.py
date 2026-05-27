from torch.nn import (
    Module,
)
import torch
import torch.nn.functional as F


class FocalLoss(Module):
    """Binary focal loss.
    Args:
        alpha (float): weight for positive class (for imbalance), default=1.0
        gamma (float): focusing parameter, default=2.0
        reduction (str): 'mean', 'sum', or 'none'
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # expects raw logits as inputs (like BCEWithLogitsLoss)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction="none"
        )
        # pt = exp(-bce_loss) = predicted probability assigned to the true class
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
