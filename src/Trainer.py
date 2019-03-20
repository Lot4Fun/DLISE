#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import numpy as np
from .lib import utils
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from .model.CNN import DLModel

from functools import reduce
from sklearn.metrics import mean_squared_error

from logging import DEBUG
from logging import getLogger

logger = getLogger('impulso')
logger.setLevel(DEBUG)

# Set HOME directory.
IMPULSO_HOME = os.environ['IMPULSO_HOME']


class Trainer(object):

    def __init__(self, hparams, experiment_id):
        logger.info('Begin init of Trainer')
        self.hparams = hparams
        self.experiment_id = experiment_id
        self.output_dir = Path(IMPULSO_HOME).joinpath(f'experiments/{self.experiment_id}')

        # Check initial weights
        if self.hparams['model_path'] and os.path.exists(self.hparams['model_path']):
            self.model_path = self.hparams['model_path']
        else:
            logger.info('Model path is not defined or existed in hparams. Use initial weights.')
            self.model_path = None

        logger.info('End init of Trainer')



    def train(self, data_loader, model_obj, optimizer, loss_fn, total_epoch, epoch, device):
        
        model_obj.train() # Set as train mode

        # Train in each mini-batch
        for infos, maps, targets in data_loader:

            # Send data to GPU dvice
            infos = infos.to(device)
            maps = maps.to(device)
            targets = targets.to(device)

            optimizer.zero_grad() # Initialize gradients
            outputs = model_obj(maps, infos) # Calculate forward
            loss = loss_fn(outputs, targets) # Calculate loss

            loss.backward() # Back propagation
            optimizer.step() # Update weights
        
        print('Epoch [%d/%d], Loss: %.4f' % (epoch, total_epoch, loss.item()))


    def test(self, data_loader, trained_model, device):

        trained_model.eval() # Set as estimation mode

        # Define temporary variables to calculate accuracy
        data_size = 0
        sum_of_squared_error = 0

        # Estimate in each mini-batch
        with torch.no_grad(): # Not use gradient for estimation

            for infos, maps, targets in data_loader:

                # Send data to GPU dvice
                infos = infos.to(device)
                maps = maps.to(device)
                targets = targets.to(device)

                # Forward propagation
                outputs = trained_model(maps, infos)

                # Calculate error
                sum_of_squared_error += np.square(targets - outputs).sum()
                data_size += reduce(lambda x, y: x * y, targets.size())

        # Output error for test data
        print(f'\nAccuracy: {sum_of_squared_error / data_size}\n')


    def run(self, train_data_loader, test_data_loader):

        # 1. GPUの設定（PyTorchでは明示的に指定する必要がある）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

        # 6. モデル作成
        model = DLModel(
            self.hparams['input_c'], self.hparams['input_h'], self.hparams['input_w'],
            self.hparams['output_n'],
            conv_kernel=3, max_pool_kernel=2
        ).to(device)
        print(model) # ネットワークの詳細を確認用に表示

        # 7. 損失関数を定義
        loss_fn = nn.MSELoss()

        # 8. 最適化手法を定義（ここでは例としてAdamを選択）
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # 9. 学習（エポック終了時点ごとにテスト用データで評価）
        print('Begin train')
        for epoch in range(1, self.hparams['epoch']+1):
            self.train(train_data_loader, model, optimizer, loss_fn, self.hparams['epoch'], epoch, device)
            self.test(test_data_loader, model, device)


if __name__ == '__main__':
    """add"""
