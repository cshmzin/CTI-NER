# coding=utf-8
import torch
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001

    def forward(self, logits, labels):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([logits.size(0), logits.size(1)]).to(device).scatter_(1, new_label, 1)
        # calculate log
        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt) ** self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()


class DiceLoss(nn.Module):
    """
    Examples:
        >>> loss = DiceLoss()
        >>> input = torch.randn(3, 1, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self,
                 smooth: Optional[float] = 1e-8,
                 square_denominator: Optional[bool] = False,
                 with_logits: Optional[bool] = True,
                 reduction: Optional[str] = "mean") -> None:
        super(DiceLoss, self).__init__()

        self.reduction = reduction
        self.with_logits = with_logits
        self.smooth = smooth
        self.square_denominator = square_denominator

    def forward(self,
                input: Tensor,
                target: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:

        flat_input = input.view(-1)
        flat_target = target.view(-1)

        if self.with_logits:
            flat_input = torch.sigmoid(flat_input)

        if mask is not None:
            mask = mask.view(-1).float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask

        interection = torch.sum(flat_input * flat_target, -1)
        if not self.square_denominator:
            return 1 - ((2 * interection + self.smooth) /
                        (flat_input.sum() + flat_target.sum() + self.smooth))
        else:
            return 1 - ((2 * interection + self.smooth) /
                        (torch.sum(torch.square(flat_input, ), -1) + torch.sum(torch.square(flat_target),
                                                                               -1) + self.smooth))

    def __str__(self):
        return f"Dice Loss smooth:{self.smooth}"