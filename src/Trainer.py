#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import numpy as np
from .lib import utils
from pathlib import Path

from .lib.callbacks import select_callbacks 

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
        if os.path.exists(self.hparams['model_path']):
            self.model_path = self.hparams['model_path']
        else:
            logger.info('Model path is not defined or existed in hparams. Use initial weights.')
            self.model_path = None

        logger.info('End init of Trainer')


    def run(self, train_data_loader, test_data_loader):
        hparams_fit = self.hparams[self.exec_type]['fit']
        self.model.fit(self.x_train,
                       self.t_train,
                       batch_size=hparams_fit['batch_size'],
                       epochs=hparams_fit['epochs'],
                       verbose=hparams_fit['verbose'],
                       validation_split=hparams_fit['validation_split'],
                       shuffle=hparams_fit['shuffle'],
                       initial_epoch=hparams_fit['initial_epoch'],
                       callbacks=self.callbacks)


if __name__ == '__main__':
    """add"""
