from typing import List

import torch
from torch import Tensor
import torch.nn.functional as F

from src.base.base_metric import BaseMetric
from src.metric.compute_eer import compute_eer


class EERMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self, prediction: Tensor, target: Tensor, **kwargs
    ):
        prediction = F.softmax(prediction, dim=-1).detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        bonafide_scores = prediction[target==1][:,1]
        other_scores = prediction[target==0][:,1]
        eer, _ = compute_eer(bonafide_scores, other_scores)
        return eer