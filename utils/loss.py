#!/usr/bin/python3
# -*- coding: utf-8 -*-

from logging import getLogger

import torch
import torch.nn as nn

class WeightedLoss(nn.Module):
    
    def __init__(self, device, pre_min, pre_max, pre_interval):

        super().__init__()

        self.device = device
        self.base_weights = [1/factor for factor in range(pre_min, pre_max+pre_interval, pre_interval)]

    def forward(self, outputs, targets):

        weights = torch.Tensor([self.base_weights] * outputs.shape[0])
        weights = weights.to(self.device)

        loss = torch.abs(outputs - targets)
        loss = loss * weights
        loss = torch.sum(loss)

        return loss
