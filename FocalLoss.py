import torch
import torch.nn as nn

class FocalCELoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        loss = nn.functional.cross_entropy(y_pred, y_true, reduction='none')

        focal_loss = ((1 - torch.exp(-loss)) ** self.gamma) * loss

        if self.alpha is not None:
            alpha = self.alpha[y_true]
            focal_loss = alpha * focal_loss

        return focal_loss.mean()

class FocalBCELoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')

        probs = torch.sigmoid(y_pred)

        pt = torch.where(y_true == 1, probs, 1 - probs)

        focal_weight = (1 - pt) ** self.gamma

        return ((1 - pt) ** self.gamma) * (a)
