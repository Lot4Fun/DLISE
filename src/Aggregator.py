#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from .lib import utils
from .lib import pre_process

from logging import DEBUG
from logging import getLogger

logger = getLogger('impulso')
logger.setLevel(DEBUG)

# Set HOME directory.
IMPULSO_HOME = os.environ['IMPULSO_HOME']


class Aggregator(object):

    def __init__(self, exec_type, hparams):
        logger.info('Begin init of Aggregator')
        self.exec_type = exec_type
        self.hparams = hparams
        self.hparams[exec_type]['data_id'] = utils.issue_id()
        self.hparams[exec_type]['output_train'] = os.path.join(IMPULSO_HOME,
                                                               'datasets',
                                                               self.hparams[exec_type]['data_id'],
                                                               'train')
        self.hparams[exec_type]['output_test'] = os.path.join(IMPULSO_HOME, 'datasets', 'test')
        self.hparams[exec_type]['output_predict'] = os.path.join(IMPULSO_HOME, 'datasets', 'predict')

        logger.info('Check hparams.yaml')
        utils.check_hparams(self.exec_type, self.hparams)

        logger.info('Backup hparams.yaml and src')
        utils.backup_before_run(self.exec_type, self.hparams)
        logger.info('End init of Aggregator')


    def load_data(self):

        logger.info('Load dataset')
        #x_cols = ['Age', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp', 'DeckNumber', 'Room']
        x_cols = ['Age', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp']
        t_cols = ['Survived']

        train_path = os.path.join(self.hparams['dataset']['input_path'], 'train.csv')
        test_path = os.path.join(self.hparams['dataset']['input_path'], 'test.csv')
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        train['dtype'] = 'train'
        test['dtype'] = 'test'
        data = pd.concat([train, test], axis=0)
        data.index = range(len(data))

        logger.info('Pre-process')
        data = pre_process.replace_value(data)

        logger.info('Split')
        train_and_test = data[data['dtype'] == 'train']
        predict = data[data['dtype'] == 'test']

        logger.info('Change train_and_test to numpy array')
        x_train_and_test = []
        t_train_and_test = []
        passenger_id_train_and_test = []
        for idx in train_and_test.index:
            x_train_and_test.append(train_and_test.loc[idx, x_cols])
            t_train_and_test.append(train_and_test.loc[idx, t_cols])
            passenger_id_train_and_test.append(train_and_test.loc[idx, 'PassengerId'])

        logger.info('Change predict to numpy array')
        x_predict = []
        passenger_id_predict = []
        for idx in predict.index:
            x_predict.append(predict.loc[idx, x_cols])
            passenger_id_predict.append(predict.loc[idx, 'PassengerId'])

        logger.info('Change list to numpy array')
        x_train_and_test = np.array(x_train_and_test)
        t_train_and_test = np.array(t_train_and_test)
        passenger_id_train_and_test = np.array(passenger_id_train_and_test)
        x_predict = np.array(x_predict)
        passenger_id_predict = np.array(passenger_id_predict)

        logger.info('Shuffle before splitting into train and test data')
        zipped = list(zip(x_train_and_test, t_train_and_test, passenger_id_train_and_test))
        np.random.seed(self.hparams[self.exec_type]['random_seed'])
        np.random.shuffle(zipped)
        x_train_and_test, t_train_and_test, passenger_id_train_and_test = zip(*zipped)
        x_train_and_test = np.array(x_train_and_test)
        t_train_and_test = np.array(t_train_and_test)
        passenger_id_train_and_test = np.array(passenger_id_train_and_test)

        logger.info('Split train_and_test to train and test')
        self.x_train, self.x_test = np.split(x_train_and_test, [int(x_train_and_test.shape[0] * (1. - self.hparams[self.exec_type]['test_split']))])
        self.t_train, self.t_test = np.split(t_train_and_test, [int(t_train_and_test.shape[0] * (1. - self.hparams[self.exec_type]['test_split']))])
        self.passenger_id_train, self.passenger_id_test = np.split(passenger_id_train_and_test, [int(len(passenger_id_train_and_test) * (1. - self.hparams[self.exec_type]['test_split']))])

        logger.info('Prepare predict data')
        self.x_predict = x_predict
        self.passenger_id_predict = passenger_id_predict

        assert len(self.x_train) == len(self.t_train) == len(self.passenger_id_train), 'Lengths of train data are different'
        assert len(self.x_test) == len(self.t_test) == len(self.passenger_id_test), 'Lengths of test data are different'
        assert len(self.x_predict) == len(self.passenger_id_predict), 'Lengths of predict data are different'
        logger.info('End loading dataset')


    def save_data(self):
        logger.info('Save data')
        train_x_dir = os.path.join(self.hparams['dataset']['output_train'], 'x')
        train_t_dir = os.path.join(self.hparams['dataset']['output_train'], 't')
        test_x_dir = os.path.join(self.hparams['dataset']['output_test'], 'x')
        test_t_dir = os.path.join(self.hparams['dataset']['output_test'], 't')
        predict_x_dir = os.path.join(self.hparams['dataset']['output_predict'], 'x')
        
        for output_dir in [train_x_dir, train_t_dir, test_x_dir, test_t_dir, predict_x_dir]:
            os.makedirs(output_dir, exist_ok=True)
        
        np.save(file=os.path.join(train_x_dir, 'x.npy'), arr=self.x_train)
        np.save(file=os.path.join(train_t_dir, 't.npy'), arr=self.t_train)
        np.save(file=os.path.join(test_x_dir, 'x.npy'), arr=self.x_test)
        np.save(file=os.path.join(test_t_dir, 't.npy'), arr=self.t_test)
        np.save(file=os.path.join(test_x_dir, 'passenger_id.npy'), arr=self.passenger_id_test)
        np.save(file=os.path.join(predict_x_dir, 'x.npy'), arr=self.x_predict)
        np.save(file=os.path.join(predict_x_dir, 'passenger_id.npy'), arr=self.passenger_id_predict)

        logger.info('End saving data')


if __name__ == '__main__':
    """add"""
