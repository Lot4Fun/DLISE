#!/usr/bin/python3
# -*- coding: utf-8 -*-

from logging import getLogger
from pathlib import Path
from attrdict import AttrDict
import torch

logger = getLogger('DLISE')

class Config(object):

    def __init__(self):

        # Requirements : model
        # Requirements : preprocess
        _preprocess_ssh_input_dir = '/PTAH/TO/SSH/DIRECTORY'
        _preprocess_sst_input_dir = '/PTAH/TO/SST/DIRECTORY'
        _preprocess_argo_input_dir = '/PTAH/TO/ARGO/DIRECTORY'
        _preprocess_save_dir = None
        # Requirements : train
        _train_input_dir = '/PTAH/TO/DATA/DIRECTORY'
        _train_save_dir = None
        # Requirements : detect
        _trained_weight_path = '/PATH/TO/PRETRAINED/WEIGHT/MODEL'

        self.model = {
            # General
            'input_size': int(_model_type.split('_')[1]),
            'variance': [0.1, 0.2],
            'rgb_means': (104.0, 117.0, 123.0),
        }

        self.preproces = {
            'input_dir': _preprocess_input_dir,
            'save_dir': _preprocess_save_dir,
        }

        self.train = {
            'input_dir': _train_input_dir,
            'save_dir': _train_save_dir,
            'loss_function': {
            },
            'resume_weight_path': '',
            'num_workers': 0,
            'batch_size': 64,
            'epoch': 1000,
            'shuffle': True,
            'split_random_seed': 0,
            'weight_save_period': 5,
            'optimizer': {
                'lr': 5e-4,
                'wait_decay_epoch': 100,
                'momentum': 0.9,
                'weight_decay': 5e-4,
                'T_max': 10
            }
        }

        self.detect = {
            'trained_weight_path': _trained_weight_path,
            'visualize': True,
            'save_results': True,
        }

        assert not (self.train['optimizer']['wait_decay_epoch'] % self.train['weight_save_period']), 'wait_decay_epoch must be multiples of weight_save_period.'


    def build_config(self):

        config = {
            'model': self.model,
            'preprocess': self.preproces,
            'train': self.train,
            'detect': self.detect,
        }

        logger.info(config)

        return AttrDict(config)


if __name__ == '__main__':

    from pprint import pprint

    config = Config().build_config()
    pprint(config)
