#!/usr/bin/python3
# -*- coding: utf-8 -*-

from logging import getLogger
import math
import torch
from torch import optim
from torch.autograd import Variable
from model.modules.multi_box_loss import MultiBoxLoss
from utils.common import CommonUtils
from utils.optimizers import Optimizers

logger = getLogger('DLISE')

class Trainer(object):

    def __init__(self, model, device, config, save_dir):
        
        self.model = model
        self.device = device
        self.config = config
        self.save_dir = save_dir


    def run(self, train_loader, validate_loader):

        loss_fn = MultiBoxLoss(self.config, self.device)

        optimizer = Optimizers.get_optimizer(self.config.train.optimizer, self.model.parameters())
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.train.optimizer.T_max)
        
        logger.info('Begin training')
        for epoch in range(1, self.config.train.epoch+1):

            enable_scheduler = (epoch > self.config.train.optimizer.wait_decay_epoch)
            if epoch == self.config.train.optimizer.wait_decay_epoch + 1:
                logger.info(f'Enable learning rate scheduler at Epoch: {epoch:05}')

            # Warm restart
            if enable_scheduler and (epoch % self.config.train.optimizer.T_max == 1):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.config.train.optimizer.lr

            train_loss = self._train(loss_fn, optimizer, train_loader)
            val_loss = self._validate(loss_fn, validate_loader)

            if enable_scheduler:
                scheduler.step()

            logger.info(f'Epoch [{epoch:05}/{self.config.train.epoch:05}], Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}')

            if epoch % self.config.train.weight_save_period == 0:
                save_path = self.save_dir.joinpath('weights', f'weight-{str(epoch).zfill(5)}_{train_loss:.5f}_{val_loss:.5f}.pth')
                CommonUtils.save_weight(self.model, save_path)
                logger.info(f'Saved weight at Epoch : {epoch:05}')


    def _train(self, loss_fn, optimizer, train_data_loader):

        # Keep track of training loss
        loc_loss = 0.
        conf_loss = 0.
        train_loss = 0.

        # Train the model in each mini-batch
        self.model.train()
        for mini_batch in train_data_loader:

            # Send data to GPU dvice
            if self.device.type == 'cuda':
                images = mini_batch[0].to(self.device)
                targets = [ann.to(self.device) for ann in mini_batch[1]]
            else:
                images = mini_batch[0]
                targets = [ann for ann in mini_batch[1]]

            # Forward
            optimizer.zero_grad()
            outputs = self.model(images)
            loss_l, loss_c = loss_fn(outputs, targets)
            loss = loss_l + loss_c

            # Backward and update weights
            loss.backward()
            optimizer.step()

            # Update training loss
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            train_loss += loss.item()

        train_loss /= math.ceil(len(train_data_loader.dataset) / self.config.train.batch_size)

        return train_loss


    def _validate(self, loss_fn, validate_data_loader):

        # Keep track of validation loss
        loc_loss = 0.
        conf_loss = 0.
        valid_loss = 0.0

        # Not use gradient for inference
        self.model.eval()
        with torch.no_grad():

            # Validate in each mini-batch
            for mini_batch in validate_data_loader:

                # Send data to GPU dvice
                if self.device.type == 'cuda':
                    images = Variable(mini_batch[0].to(self.device))
                    targets = [ann.to(self.device) for ann in mini_batch[1]]
                else:
                    images = Variable(mini_batch[0])
                    targets = [ann for ann in mini_batch[1]]

                # Forward
                outputs = self.model(images)
                loss_l, loss_c = loss_fn(outputs, targets)
                loss = loss_l + loss_c

                # Update validation loss
                loc_loss += loss_l.item()
                conf_loss += loss_c.item()
                valid_loss += loss.item()

        valid_loss /= math.ceil(len(validate_data_loader.dataset) / self.config.train.batch_size)

        return valid_loss
