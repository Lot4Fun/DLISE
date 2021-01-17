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
        """
        _preprocess_ssh_input_dir = '/PTAH/TO/SSH/DIRECTORY'
        _preprocess_sst_input_dir = '/PTAH/TO/SST/DIRECTORY'
        _preprocess_bio_input_dir = '/PTAH/TO/BIO/DIRECTORY'
        _preprocess_argo_input_dir = '/PTAH/TO/ARGO/DIRECTORY'
        """
        _preprocess_ssh_input_dir = '/archive/DLISE/download/20210117/ssh'
        _preprocess_sst_input_dir = '/archive/DLISE/download/20210117/sst'
        _preprocess_bio_input_dir = '/archive/DLISE/download/20210117/bio'
        _preprocess_argo_input_dir = '/archive/DLISE/download/20210117/_argo'
        _preprocess_save_dir = None
        # Requirements : train
        _train_input_dir = '/PTAH/TO/DATA/DIRECTORY'
        _train_save_dir = None
        # Requirements : detect
        _trained_weight_path = '/PATH/TO/PRETRAINED/WEIGHT/MODEL'

        self.model = {
            # General
        }

        self.preprocess = {
            'ssh_input_dir': _preprocess_ssh_input_dir,
            'sst_input_dir': _preprocess_sst_input_dir,
            'bio_input_dir': _preprocess_bio_input_dir,
            'argo_input_dir': _preprocess_argo_input_dir,
            'save_dir': _preprocess_save_dir,
            'end_of_train': '2019-12-31',
            'interpolation': {
                'pre_min': 10, # Minimum pressure
                'pre_max': 1000, # Maximum pressure
                'pre_interval': 10 # Interval to interpolate
            },
            'crop': {
                'zonal': 4, # -zonal to +zonal in degree
                'meridional': 4 # -meridional to +meridional in degree
            },
            'argo': { # Extract argo profiles in the following region and term
                'lat_min': 35,
                'lat_max': 40,
                'lon_min': 140,
                'lon_max': 180,
                'date_min': '2018-01-01',
                'date_max': '2021-01-17'                
            }
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
            'preprocess': self.preprocess,
            'train': self.train,
            'detect': self.detect,
        }

        logger.info(config)

        return AttrDict(config)


if __name__ == '__main__':

    from pprint import pprint

    config = Config().build_config()
    pprint(config)
