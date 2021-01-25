#!/usr/bin/python3
# -*- coding: utf-8 -*-

from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from utils.common import CommonUtils

logger = getLogger('DLISE')

class Evaluator(object):

    def __init__(self, model, device, config, save_dir):
        
        self.model = model
        self.device = device
        self.config = config
        self.save_dir = save_dir

    def run(self, eval_loader):

        loss_fn =nn.L1Loss()

        logger.info('Begin evaluating')

        eval_loss = self._evaluate(loss_fn, eval_loader)

        logger.info(f'Loss for test dataset: {eval_loss:.5f}')

    def _evaluate(self, loss_fn, eval_loader):

        # Keep track of validation loss
        eval_loss = 0.0

        # Not use gradient for inference
        self.model.eval()
        with torch.no_grad():

            # evaluate in each mini-batch
            n_figure = 0
            for mini_batch in eval_loader:

                # Send data to GPU dvice
                input_lats = mini_batch[1].to(self.device)
                input_lons = mini_batch[2].to(self.device)
                input_maps = mini_batch[3].to(self.device)
                pressure = mini_batch[4]
                targets = mini_batch[5].to(self.device)

                # Forward
                outputs = self.model(input_lats, input_lons, input_maps)
                loss = loss_fn(outputs, targets)

                # Update validation loss
                eval_loss += loss.item()

                # Draw vertical profile figure
                n_figure += 1
                if n_figure <= self.config.evaluate.n_figure:
                    filename = mini_batch[6][0]
                    self.draw_profile(outputs, targets, pressure, filename)
                
                if n_figure % 100 == 0:
                    logger.info(f'Evaluate [{n_figure:07}/{len(eval_loader.dataset):07}]')

        eval_loss /= len(eval_loader.dataset)

        return eval_loss

    def draw_profile(self, output, target, pressure, filename):

        output = output.to('cpu').squeeze().detach().numpy().copy()
        target = target.to('cpu').squeeze().detach().numpy().copy()
        pressure = pressure.squeeze().detach().numpy().copy()
    
        save_path = self.save_dir.joinpath(self.config.evaluate.objective, filename + '.png')

        # Draw figure
        plt.figure(figsize=(4,8))
        plt.plot(target, pressure, color='red')
        plt.plot(output, pressure, color='blue')

        plt.title(self.config.evaluate.objective)
        plt.xlim([0, 35])
        plt.ylim([1000, 10])

        plt.grid(color='gray', linestyle='dotted', linewidth=0.5)

        ticks = np.linspace(0, 1000, 11)
        plt.yticks(ticks)

        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
