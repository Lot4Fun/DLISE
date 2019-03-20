#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class DLModel(nn.Module):

    def __init__(self, in_c, in_h, in_w, out_size, conv_kernel=2, max_pool_kernel=2):
        """
        Convolutional Neural Network
        
        Network Structure：

            input(map) ─ CONV ─ CONV ─ MaxPool ─ CONV ─ CONV ─ MaxPool ┬ FC ─ FC ─ output
            input(info) ───────────────────────────────────────────────┘

            # Apply batch normalizetion following MaxPool
        """

        super(DLModel, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=8, kernel_size=conv_kernel, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=conv_kernel, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=max_pool_kernel, stride=1),
            nn.BatchNorm2d(8)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=conv_kernel, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=conv_kernel, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=max_pool_kernel, stride=1),
            nn.BatchNorm2d(16)
        )
        self.full_connection = nn.Sequential(
            nn.Linear(in_features=16 * (in_h - 2) * (in_h - 2) + 3, out_features=1024), # '+3' means date, latitude and longitude
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=out_size, bias=False)
        )


    # Define a process of 'Forward'
    def forward(self, maps, infos):

        # Convolutional layers
        x = self.block1(maps)
        x = self.block2(x)

        # Change 2-D to 1-D
        x = x.view(x.size(0), 16 * 15 * 15)

        # Full connection layers
        y = self.full_connection(torch.cat([x, infos], dim=1))
        
        return y

