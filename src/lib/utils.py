#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-

import sys, os
import copy, shutil, glob
import yaml
import jdcal
import datetime
from pathlib import Path
import pickle
from decimal import Decimal, ROUND_HALF_UP

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


def get_minmax_index_from_degree(argo_in_degree, distance_in_degree, data_type): 
    """
    Get min-max index of a small map based on the centerl latitude and latitude
    """
    # Latitude  : index=0 -> -83.0 degree, index=691 -> 89.75 degree
    # Longitude : index=0 -> XX degree, index=1439 -> XX degree
    if data_type == 'latitude':
        min_idx = int(((argo_in_degree - distance_in_degree / 2) + 83) * 4)
        max_idx = int(((argo_in_degree + distance_in_degree / 2) + 83) * 4)
    elif data_type == 'longitude':
        min_idx = int((argo_in_degree - distance_in_degree / 2) * 4)
        max_idx = int((argo_in_degree + distance_in_degree / 2) * 4)
    else:
        sys.exit('Error in "get_minmax_index_from_degree" function. Inappropriate "data_type"')
    
    return min_idx, max_idx


def change_degree2index(degree, data_type):
    """
    Args:
        degree : Latitude(E) or Longitude(N)
        data_type : 'latitude' or 'longitude'
    """
    # Latitude  : index=0 -> -83.0 degree, index=691 -> 89.75 degree
    # Longitude : index=0 -> XX degree, index=1439 -> XX degree
    if data_type == 'latitude':
        idx = int((degree + 83) * 4)
    elif data_type == 'longitude':
        idx = int(degree * 4)
    else:
        sys.exit('Error in "change_degree2index" function. Inappropriate "data_type"')

    return idx

def round_location_in_grid(in_degree):
    """
    Round 'in_degree' in 0.25 degree units
    """
    return float(Decimal(str(in_degree * 4)).quantize(Decimal('0'), rounding=ROUND_HALF_UP) / 4)


def calc_days_elapsed(current_date, ref_date='2000-01-01'):
    """
    Args:
        current_date: YYYYMMDD (String)
        ref_date:     YYYY-MM-DD (String)
    """
    argo_jd = sum(jdcal.gcal2jd(current_date[:4], current_date[4:6], current_date[6:]))
    ref_jd = sum(jdcal.gcal2jd(ref_date.year, ref_date.month, ref_date.day))
    
    return int(argo_jd - ref_jd)


def create_default_hparams(output_path,
                           input_h, input_w, input_c,
                           output_n,
                           reference_date='2000-01-01',
                           region={
                               'latitude':{
                                   'min':0,
                                   'max':40
                                },
                                'longitude':{
                                    'min':140,
                                    'max':220
                                }
                           },
                           crop_size={
                               'zonal_distance_in_degree':4,
                               'meridional_distance_in_degree':4
                           },
                           num_workers=4,
                           test_split=0.2,
                           batch_size=128,
                           epoch=500):
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
    train_hparams['period'] = 50

    # Hyperparameters of test
    test_hparams = {}

    # Hyperparameters of inference
    inference_hparams = {}
    inference_hparams['reference_date'] = reference_date
    inference_hparams['region'] = region
    inference_hparams['crop_size'] = crop_size
    inference_hparams['input_h'] = input_h
    inference_hparams['input_w'] = input_w
    inference_hparams['input_c'] = input_c
    inference_hparams['output_n'] = output_n

    # Save
    hparams = {}
    hparams['train'] = train_hparams
    hparams['test'] = test_hparams
    hparams['inference'] = inference_hparams
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


def create_data_loader(info, obj, map, batch_size=128, shuffle=True, split_random_seed=0):
    """
    Args:
        info : argo_info
        obj : Objective variable of argo
        map : SSH/SST
        split_random_seed : random seed to split data into train and validation
    """
    # Split data into train and validation
    info_train, info_val, obj_train, obj_val, map_train, map_val = train_test_split(info, obj, map, test_size=1/7, random_state=split_random_seed)

    # Convert numpy to Tensor for PyTorch
    info_train = torch.Tensor(info_train)
    info_val = torch.Tensor(info_val)
    obj_train = torch.Tensor(obj_train)
    obj_val = torch.Tensor(obj_val)
    map_train = torch.Tensor(map_train)
    map_val = torch.Tensor(map_val)

    # Combine all data
    ds_train = TensorDataset(info_train, map_train, obj_train)
    ds_val = TensorDataset(info_val, map_val, obj_val)

    # Create DataLoader
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def infer_data_loader(info, maps):

    # Convert numpy to Tensor for PyTorch
    info = torch.Tensor(info)
    maps = torch.Tensor(maps)

    ds = TensorDataset(info, maps)

    return DataLoader(ds, batch_size=1, shuffle=False)


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    return 'Model was saved'


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
