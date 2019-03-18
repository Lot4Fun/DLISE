#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
from pathlib import Path
from .lib import utils

from logging import DEBUG
from logging import getLogger

logger = getLogger('impulso')
logger.setLevel(DEBUG)

# Set HOME directory.
IMPULSO_HOME = os.environ['IMPULSO_HOME']


class Preparer(object):

    def __init__(self):
        logger.info('Begin init of Preparer')
        self.experiment_id = utils.issue_id()
        self.output_dir = Path(IMPULSO_HOME).joinpath(f'experiments/{self.experiment_id}')
        logger.info('End of init of Preparer')


    def session(self, data_id):
        """
        Args:
            all_hparams : hparams.yml with parameters of train, test and estimate 
        """
        # Load template_hparams.yml and add data_id
        all_hparams = utils.load_hparams(Path(IMPULSO_HOME).joinpath(f'datasets/{data_id}/hparams.yml'))
        all_hparams['train']['data_id'] = data_id
        all_hparams['test']['data_id'] = data_id

        # Save
        utils.save_hparams(self.output_dir, 'hparams.yml', all_hparams)
        utils.copy_directory(Path(IMPULSO_HOME).joinpath('src'),self.output_dir.joinpath('src'))


if __name__ == '__main__':
    """add"""
