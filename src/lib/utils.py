#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-

import sys, os
import copy, shutil, glob
import yaml
import datetime
from pathlib import Path
import pickle

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


from logging import DEBUG, INFO
from logging import getLogger

# Set HOME directory.
IMPULSO_HOME = os.environ['IMPULSO_HOME']

# Set logger.
logger = getLogger('impulso')


def load_hparams(yaml_path):
    logger.debug(f'Load hyperparameters: {yaml_path}')
    with open(yaml_path) as f:
        hparams = yaml.load(f)
    logger.info('Load hparams')
    logger.info(hparams)
    return hparams


def issue_id():
    logger.debug('Generate issue ID.')
    id = datetime.datetime.now().strftime('%m%d-%H%M-%S%f')[:-4]
    return id


def save_hparams(output_dir, output_file, hparams):
    logger.debug('Save hyperparameters.')
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir.joinpath(output_file), 'w') as f:
        yaml.dump(hparams, f, default_flow_style=False)


def save_as_pickle(obj_to_save, save_path):
    logger.info('Begin saving ' + save_path.name)
    with open(save_path, 'wb') as f:
        pickle.dump(obj_to_save, f)
    logger.info('End saving ' + save_path.name)



def copy_directory(from_dir, to_dir):
    """
    Copy 'from_dir' directory as 'to_dir'.
    """
    logger.debug('Copy directory.')
    shutil.copytree(from_dir, to_dir)



def create_default_hparams(output_path,
                           input_h, input_w, input_c,
                           output_n,
                           num_workers=4,
                           test_split=0.2,
                           batch_size=128,
                           epoch=100):
    """
    Create hparams.yml for train, test and estimate phase.
    
    Args:
        output_path (str) : Output path of hparams.yml
        input_h  (int)    : Height of input data
        input_w  (int)    : Width of input data
        input_c  (int)    : Channel of input data
        output_n (int)    : Dimension of output data
    """
    logger.info('Create default hparams.yml')

    # Hyperparameters for train
    train_hparams = {}
    train_hparams['input_h'] = input_h
    train_hparams['input_w'] = input_w
    train_hparams['input_c'] = input_c
    train_hparams['output_n'] = output_n
    train_hparams['num_workers'] = num_workers
    train_hparams['test_split'] = test_split
    train_hparams['batch_size'] = batch_size
    train_hparams['epoch'] = epoch
    train_hparams['shuffle'] = True
    train_hparams['model_path'] = None
    train_hparams['objective_variable'] = 'temperature'
    train_hparams['split_random_seed'] = 0

    # Hyperparameters of test
    test_hparams = {}

    # Hyperparameters of estimate
    estimate_hparams = {}

    # Save
    hparams = {}
    hparams['train'] = train_hparams
    hparams['test'] = test_hparams
    hparams['estimate'] = estimate_hparams
    with open(output_path, 'w') as f:
        yaml.dump(hparams, f, default_flow_style=False)

    logger.info('Saved default hparams.yml')


def load_data(data_path, objective_variable='temperature'):
    """
    Args:
        data_path : Full path to 'data_id' directory
        objective_variable : Temperature or Salinity
    """
    
    obj_var = objective_variable.lower()

    if obj_var in ['temperature', 'tem', 'temp']:
        obj_var = 'tem'
    elif obj_var in ['salinity', 'sal', 'sali']:
        obj_var = 'sal'
    else:
        logger.info('Indicated objective variable does not exist. Use temperature')
        obj_var = 'tem'

    logger.info(f'Load data: {data_path}')

    logger.info('Loading argo profiles')
    with open(Path(data_path).joinpath('argo.pkl'), 'rb') as f:
        argo = pickle.load(f)
    
    with open(Path(data_path).joinpath('map.pkl'), 'rb') as f:
        map = pickle.load(f)

    logger.info(f'Finished loading data: {data_path}')

    return argo['info'], argo['pre'], argo[obj_var], map


def create_data_loader(info, pre, obj, map, batch_size= 128, shuffle=True, split_random_seed=0):
    """
    Args:
        info : argo_info
        pre : argo pressure
        obj : Objective variable of argo
        map : SSH/SST
        split_random_seed : random seed to split data into train and test
    """

    # Split data into train and test
    info_train, info_test, pre_train, pre_test, obj_train, obj_test, map_train, map_test = train_test_split(info, pre, obj, map, test_size=1/7, random_state=split_random_seed)

    # Convert numpy top Tensor for PyTorch
    info_train = torch.Tensor(info_train)
    info_test = torch.Tensor(info_test)
    pre_train = torch.Tensor(pre_train)
    pre_test = torch.Tensor(pre_test)
    obj_train = torch.Tensor(obj_train)
    obj_test = torch.Tensor(obj_test)
    map_train = torch.Tensor(map_train)
    map_test = torch.Tensor(map_test)

    # Combine all data
    ds_train = TensorDataset(info_train, map_train, obj_train)
    ds_test = TensorDataset(info_test, map_test, obj_test)

    # Create DataLoader
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == '__main__':
    """
    __main__ is for DEBUG.
    """
    # Check hparams.
    from pprint import pprint
    hparams = load_hparams(os.path.join(IMPULSO_HOME, 'hparams/hparams.yaml'))
    pprint(hparams)

    # Check ID.
    id = issue_id()
    print(id)
