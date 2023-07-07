import torch as tr
from torch import nn


# -----------------------------------
# --- Fairness Distribution Distances
# -----------------------------------

class FirstOrderMatching(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        if reduction not in ("mean", 'sum', "norm"):
            raise ValueError("Bad argument value for the reduction parameter")
        self.reduction = reduction

    def __str__(self):
        return "FOM"

    @staticmethod
    def mean_distance(d1, d2, reduction="mean"):
        loss = (d1.mean(dim=0) - d2.mean(dim=0)).abs()
        
        if reduction == "sum":
            loss = loss.sum()
        elif reduction == "norm":
            loss = tr.linalg.norm(loss, ord=2)
        else:
            loss = loss.mean()
        
        return loss

    def forward(self, distr_1, distr_2):
        return FirstOrderMatching.mean_distance(distr_1, distr_2, reduction=self.reduction)
