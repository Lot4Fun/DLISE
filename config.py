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
        _backbone_pretrained = False
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
        _preprocess_argo_input_dir = '/archive/DLISE/download/20210117/argo'
        _preprocess_save_dir = None
        # Requirements : train
        """
        _train_input_dir = '/PATH/TO/DATA/DIRECTORY'
        """
        _train_input_dir = './data_storage/2021-0120-0112-4935'
        _train_save_dir = None
        # Requirements : evaluate
        """
        _evaluate_input_dir = '/PATH/TO/DATA/DIRECTORY'
        """
        _evaluate_input_dir = './data_storage/2021-0120-0112-4935'

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
                'lat_min': 10, # Default: 35
                'lat_max': 40, # Default: 40
                'lon_min': 140, # Default: 140
                'lon_max': 220, # Default: 180
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
            'batch_size': 512,
            'epoch': 1000,
            'shuffle': True,
            'weight_save_period': 5,
            'weighted_loss': True,
            'optimizer': {
                'optim_type': 'adam',
                'sgd': {
                    'lr': 5e-4,
                    'momentum': 0.9,
                    'weight_decay': 5e-4,
                },
                'adam': {
                    'lr': 0.0001,
                    'betas': (0.9, 0.999),
                    'eps': 1e-08,
                    'weight_decay': 0,
                    'amsgrad': False
                }
            }
        }

        self.evaluate = {
            #'trained_weight_path': '/PATH/TO/PRETRAINED/WEIGHT',
            'trained_weight_path': './results/train/2021-0128-0041-5374/weights/weight-00225_0.98844_0.44609.pth',
            'objective': _objective,
            'input_dir': _evaluate_input_dir,
            'n_figure': 100
        }

        self.predict = {
            'crop': { # Must be the same of preprocess
                'zonal': 4,
                'meridional': 4
            },
            'objectives': {
                # Add information to predict as follows.
                #    'YYYYMMDD': {
                #        'lat_min': southern_limit,
                #        'lat_max': northern_limit,
                #        'lon_min': western_limit,
                #        'lon_max': eastern_limit
                #    }
                '20201001': {
                    'lat_min': 10,
                    'lat_max': 40,
                    'lon_min': 140,
                    'lon_max': 220
                },
                '20201015': {
                    'lat_min': 10,
                    'lat_max': 40,
                    'lon_min': 140,
                    'lon_max': 220
                }
            },
            'trained_weight_path': '/PATH/TO/PRETRAINED/WEIGHT',
            'save_results': True
        }

        self.visualize = {
            #'predicted_dir': '/PATH/TO/PREDICTION/DIR',
            'predicted_dir': './results/predict/2021-0125-0233-5371',
            'objectives': [
                {
                    'date': '20201001',
                    'map': {
                        'draw': True,
                        'lat_min': 10,
                        'lat_max': 40,
                        'lon_min': 140,
                        'lon_max': 220
                    },
                    'draw_lines_on_map': True,
                    'zonal_sections': [
                        {
                            'lat': 20,
                            'lon_min': 170,
                            'lon_max': 180,
                            'pre_min': 10,
                            'pre_max': 1000
                        },
                        {
                            'lat': 30,
                            'lon_min': 180,
                            'lon_max': 190,
                            'pre_min': 10,
                            'pre_max': 1000
                        },
                        {
                            'lat': 40,
                            'lon_min': 190,
                            'lon_max': 200,
                            'pre_min': 10,
                            'pre_max': 1000
                        },
                    ],
                    'meridional_sections': [
                        {
                            'lon': 150,
                            'lat_min': 20,
                            'lat_max': 30,
                            'pre_min': 10,
                            'pre_max': 1000
                        },
                        {
                            'lon': 160,
                            'lat_min': 30,
                            'lat_max': 40,
                            'pre_min': 10,
                            'pre_max': 1000
                        },
                    ]
                },
                {
                    'date': '20201015',
                    'map': {
                        'draw': True,
                        'lat_min': 10,
                        'lat_max': 40,
                        'lon_min': 140,
                        'lon_max': 220
                    },
                    'draw_lines_on_map': True,
                    'zonal_sections': [
                        {
                            'lat': 20,
                            'lon_min': 170,
                            'lon_max': 180,
                            'pre_min': 10,
                            'pre_max': 1000
                        },
                        {
                            'lat': 30,
                            'lon_min': 180,
                            'lon_max': 190,
                            'pre_min': 10,
                            'pre_max': 1000
                        },
                        {
                            'lat': 40,
                            'lon_min': 190,
                            'lon_max': 200,
                            'pre_min': 10,
                            'pre_max': 1000
                        },
                    ],
                    'meridional_sections': [
                        {
                            'lon': 150,
                            'lat_min': 20,
                            'lat_max': 30,
                            'pre_min': 10,
                            'pre_max': 1000
                        },
                        {
                            'lon': 160,
                            'lat_min': 30,
                            'lat_max': 40,
                            'pre_min': 10,
                            'pre_max': 1000
                        },
                    ]
                }
            ]
        }

    def build_config(self):

        config = {
            'model': self.model,
            'preprocess': self.preprocess,
            'train': self.train,
            'evaluate': self.evaluate,
            'predict': self.predict,
            'visualize': self.visualize,
        }

        logger.info(config)

        return AttrDict(config)


if __name__ == '__main__':

    from pprint import pprint

    config = Config().build_config()
    pprint(config)
