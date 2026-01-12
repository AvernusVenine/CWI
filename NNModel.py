import numpy as np
import torch
import torch.nn as nn

class IntervalNeuralNetwork(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class IntervalIntegratedLoss(nn.Module):

    def __init__(self, lam=0, gamma=2):
        super().__init__()
        self.lam = lam
        self.gamma = gamma

    def forward(self, model, depth_top, depth_bot, X, y_true):

        batch_size = X.shape[0]
        device = model.device

        widths = depth_bot - depth_top
        points = (widths * self.gamma + 2)

        integrals = []

        for i in range(batch_size):
            n_points = points[i].item()

            jitter =  (torch.rand(n_points - 2, device=device) - .5) * (1.0/(n_points - 1))

            pass

        logits = model(X)