from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, ground_truth, prediction):
        return F.mse_loss(ground_truth, prediction)

class Criterion(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, ground_truth, prediction):
        criterion = nn.BCEWithLogitsLoss()# MSELoss()
        loss = criterion(prediction, ground_truth)
        return loss