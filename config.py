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
        _backbone_pretrained = True
        _input_size = 224
        _objective = 'temperature' # 'temperature' or 'salinity'
        # Requirements : preprocess
        """
        _preprocess_ssh_input_dir = '/PATH/TO/SSH/DIRECTORY'
        _preprocess_sst_input_dir = '/PATH/TO/SST/DIRECTORY'
        _preprocess_bio_input_dir = '/PATH/TO/BIO/DIRECTORY'
        _preprocess_argo_input_dir = '/PATH/TO/ARGO/DIRECTORY'
        """
        _preprocess_ssh_input_dir = '/archive/DLISE/download/20210117/ssh'
        _preprocess_sst_input_dir = '/archive/DLISE/download/20210117/sst'
        _preprocess_bio_input_dir = '/archive/DLISE/download/20210117/bio'
        _preprocess_argo_input_dir = '/archive/DLISE/download/20210117/_argo'
        _preprocess_save_dir = None
        # Requirements : train
        #####_train_input_dir = '/PATH/TO/DATA/DIRECTORY'
        _train_input_dir = './data_storage/2021-0117-1817-3568'
        _train_save_dir = None

        self.model = {
            'backbone_pretrained': _backbone_pretrained,
            'input_size': _input_size,
            'objective': _objective
        }

        self.preprocess = {
            'ssh_input_dir': _preprocess_ssh_input_dir,
            'sst_input_dir': _preprocess_sst_input_dir,
            'bio_input_dir': _preprocess_bio_input_dir,
            'argo_input_dir': _preprocess_argo_input_dir,
            'save_dir': _preprocess_save_dir,
            'end_of_train': '2019-12-31',
            'interpolation': { # If change these values, neet to modify model structure
                'pre_min': 10, # Minimum pressure
                'pre_max': 1000, # Maximum pressure
                'pre_interval': 10 # Interval to interpolate
            },
            'crop': {
                'zonal': 4, # -zonal/2 to +zonal/2 in degree
                'meridional': 4 # -meridional/2 to +meridional\2 in degree
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
            'split_random_seed': 0,
            'resize_method': 'bicubic',
            'resume_weight_path': '',
            'num_workers': 0,
            'batch_size': 64,
            'epoch': 100,
            'shuffle': True,
            'weight_save_period': 5,
            'optimizer': {
                'optim_type': 'adam',
                'sgd': {
                    'lr': 5e-4,
                    'wait_decay_epoch': 100,
                    'momentum': 0.9,
                    'weight_decay': 5e-4,
                    'T_max': 10
                },
                'adam': {
                    'lr': 0.001,
                    'betas': (0.9, 0.999),
                    'eps': 1e-08,
                    'weight_decay': 0,
                    'amsgrad': False
                }
            }
        }

        self.predict = {
            'input_dir': '/PATH/TO/DATA/HOME',
            'prediction': {
                '20210101': {
                    'lat_min': 35,
                    'lat_max': 40,
                    'lon_min': 140,
                    'lon_max': 180
                },
                '20210110': {
                    'lat_min': 35,
                    'lat_max': 40,
                    'lon_min': 140,
                    'lon_max': 180
                },
            'trained_weight_path': '/PATH/TO/PRETRAINED/WEIGHT',
            'save_results': True,
        }

    def build_config(self):

        config = {
            'model': self.model,
            'preprocess': self.preprocess,
            'train': self.train,
            'predict': self.predict,
        }

        logger.info(config)

        return AttrDict(config)


if __name__ == '__main__':

    from pprint import pprint

    config = Config().build_config()
    pprint(config)
