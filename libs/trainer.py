#!/usr/bin/python3
# -*- coding: utf-8 -*-

from logging import getLogger

import sys
from pathlib import Path
sys.path.append(str(Path('__file__').resolve().parent))

import torch
import torch.nn as nn
from torch import optim
from utils.common import CommonUtils
from utils.optimizers import Optimizers

logger = getLogger('DLISE')

class Trainer(object):

    def __init__(self, model, device, config, save_dir):
        
        self.model = model
        self.device = device
        self.config = config
        self.save_dir = save_dir

    def run(self, train_loader, valid_loader):

        if self.config.train.weighted_loss:

            from utils.loss import WeightedLoss

            loss_fn = WeightedLoss(self.device,
                                   self.config.preprocess.interpolation.pre_min,
                                   self.config.preprocess.interpolation.pre_max,
                                   self.config.preprocess.interpolation.pre_interval,)
        else:
            loss_fn = nn.L1Loss()

        optimizer = Optimizers.get_optimizer(self.config.train.optimizer, self.model.parameters())
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        
        logger.info('Begin training')
        for epoch in range(1, self.config.train.epoch+1):

            train_loss = self._train(loss_fn, optimizer, train_loader)
            valid_loss = self._validate(loss_fn, valid_loader)

            scheduler.step(valid_loss)

            logger.info(f'Epoch [{epoch:05}/{self.config.train.epoch:05}], Loss: {train_loss:.5f}, Val Loss: {valid_loss:.5f}')

            if epoch % self.config.train.weight_save_period == 0:
                save_path = self.save_dir.joinpath('weights', f'weight-{str(epoch).zfill(5)}_{train_loss:.5f}_{valid_loss:.5f}.pth')
                CommonUtils.save_weight(self.model, save_path)
                logger.info(f'Saved weight at Epoch : {epoch:05}')


    def _train(self, loss_fn, optimizer, train_loader):

        # Keep track of training loss
        train_loss = 0.

        # Train the model in each mini-batch
        self.model.train()
        for mini_batch in train_loader:

            # Send data to GPU dvice
            input_lats = mini_batch[1].to(self.device)
            input_lons = mini_batch[2].to(self.device)
            input_maps = mini_batch[3].to(self.device)
            targets = mini_batch[4].to(self.device)

            # Forward
            optimizer.zero_grad()
            outputs = self.model(input_lats, input_lons, input_maps)
            loss = loss_fn(outputs, targets)

            # Backward and update weights
            loss.backward()
            optimizer.step()

            # Update training loss
            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)

        return train_loss


    def _validate(self, loss_fn, valid_loader):

        # Keep track of validation loss
        valid_loss = 0.0

        # Do not use gradient for inference
        self.model.eval()
        with torch.no_grad():

            # Validate in each mini-batch
            for mini_batch in valid_loader:

                # Send data to GPU dvice
                input_lats = mini_batch[1].to(self.device)
                input_lons = mini_batch[2].to(self.device)
                input_maps = mini_batch[3].to(self.device)
                targets = mini_batch[4].to(self.device)

                # Forward
                outputs = self.model(input_lats, input_lons, input_maps)
                loss = loss_fn(outputs, targets)

                # Update validation loss
                valid_loss += loss.item()

        valid_loss /= len(valid_loader.dataset)

        return valid_loss
