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
            self.model_path = False

        logger.info('End init of Trainer')



    def train(self, data_loader, model_obj, optimizer, loss_fn, device):
        
        # Set as train mode
        model_obj.train()

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
        
        return loss.item()


    def validate(self, data_loader, trained_model, device):

        trained_model.eval() # Set as estimation mode

        # Define temporary variables to calculate accuracy
        data_size = 0
        sum_of_squared_error = 0

        # Not use gradient for inference
        with torch.no_grad():

            # Validate in each mini-batch
            for infos, maps, targets in data_loader:

                # Send data to GPU dvice
                infos = infos.to(device)
                maps = maps.to(device)

                # Forward propagation
                outputs = trained_model(maps, infos)

                # Calculate error
                sum_of_squared_error += np.square(targets - outputs.to('cpu')).sum() # 'cpu' : RuntimeError: expected type torch.cuda.FloatTensor but got torch.FloatTensor
                data_size += reduce(lambda x, y: x * y, targets.size())

        return sum_of_squared_error / data_size


    def run(self, train_data_loader, validate_data_loader):

        # GPU setting
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Device information: {device}')

        # Create initial model
        model = DLModel(
            self.hparams['input_c'], self.hparams['input_h'], self.hparams['input_w'],
            self.hparams['output_n'],
            conv_kernel=3, max_pool_kernel=2
        ).to(device)
        logger.info(model)
        
        # Load pre-trained model
        if self.model_path:
            logger.info(f'Loading model: {self.model_path}')
            model.load_state_dict(torch.load(self.model_path))
        else:
            logger.info('Model path is not defined in hparams. Use initial weights.')

        # Define loss function
        loss_fn = nn.MSELoss()

        # Define a method of optimization
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Train and validate in each epoch
        logger.info('Begin train')
        for epoch in range(1, self.hparams['epoch']+1):

            # Train and validate
            train_loss = self.train(train_data_loader, model, optimizer, loss_fn, device)
            validate_MSE = self.validate(validate_data_loader, model, device)
            logger.info(f'Epoch [{epoch:05}/{self.hparams["epoch"]:05}], Loss: {train_loss:.2f}, Val MSE: {validate_MSE:.2f}')

            # Save model in each period
            if epoch % self.hparams['period'] == 0:
                save_path = self.output_dir.joinpath(f'models/model-{str(epoch).zfill(5)}.pth')
                utils.save_model(model, save_path)
                logger.info(f'Saved model at Epoch : {epoch:05}')


if __name__ == '__main__':
    """add"""
