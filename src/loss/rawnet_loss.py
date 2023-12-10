import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


class RawNetLoss(nn.Module):
    def __init__(self, bonafide_weight=9.0, spoof_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss(weight=Tensor([spoof_weight, bonafide_weight]))

    def forward(self, prediction, target, **batch):
        return self.loss(prediction, target)
