#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-

import sys, os
import copy, shutil, glob
import yaml
import datetime
from pathlib import Path
import pickle

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


def update_hparams(hparams):
    logger.debug('Update hyperparameters.')
    pass


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
