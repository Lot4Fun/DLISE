#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Import Machine Learning modules.
import os
os.environ['IMPULSO_HOME'] = os.path.dirname(os.path.abspath(__file__))

import importlib
from pathlib import Path
import datetime
import argparse
import numpy as np
from tqdm import tqdm

import src.lib.utils as utils
from src.Aggregator import Aggregator
from src.Preparer import Preparer

from logging import DEBUG, INFO
from logging import getLogger, StreamHandler, FileHandler, Formatter


# Set HOME directory.
IMPULSO_HOME = os.environ['IMPULSO_HOME']

# Set loger.
log_date = datetime.datetime.today().strftime('%Y%m%d')
log_path = Path(IMPULSO_HOME).joinpath(f'log/log_{log_date}.log')
logger = getLogger('impulso')
logger.setLevel(DEBUG)

stream_handler = StreamHandler()
file_handler = FileHandler(log_path)

stream_handler.setLevel(INFO)
file_handler.setLevel(DEBUG)

handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(handler_format)
file_handler.setFormatter(handler_format)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)


class Impulso(object):

    def __init__(self, args, hparams_yaml='hparams.yml'):
        logger.info('Begin init of Impulso')
        self.args = args
        if self.args.exec_type == 'dataset':
            self.hparams_path = Path(IMPULSO_HOME).joinpath(f'hparams/{hparams_yaml}')
        elif self.args.exec_type == 'prepare':
            self.hparams_path = Path(IMPULSO_HOME).joinpath(f'datasets/{self.args.data_id}/{hparams_yaml}')
        elif self.args.exec_type == 'train':
            self.hparams_path = Path(IMPULSO_HOME).joinpath(f'experiments/{self.args.experiment_id}/{hparams_yaml}')
        elif self.args.exec_type == 'inference':
            self.hparams_path = self.args.hparams
        else:
            """
            要チェック
            """
            self.hparams_path = Path(IMPULSO_HOME).joinpath(f'experiments/{self.args.experiment_id}/{hparams_yaml}')

        self.hparams = utils.load_hparams(self.hparams_path)
        logger.info('End init of Impulso')


    def dataset(self):
        logger.info('Begin dataset of Impulso')

        aggregator = Aggregator(self.hparams['dataset'])
        argo_info, pre_profiles, sal_profiles, tem_profiles, map_db = aggregator.generate_dataset()

        # Create Database
        argo_db = {}
        argo_db['info'] = argo_info
        argo_db['pre'] = pre_profiles
        argo_db['sal'] = sal_profiles
        argo_db['tem'] = tem_profiles

        # Save as Pickle
        utils.save_as_pickle(argo_db, aggregator.output_argo)
        utils.save_as_pickle(map_db, aggregator.output_map)

        # Show results
        n_profiles, n_layers_of_profile = pre_profiles.shape
        _, n_channel, n_lat_grid, n_lon_grid = map_db.shape
        logger.info(f'Number of prifiles: {n_profiles}')
        logger.info(f'Number of layers of a profile; {n_layers_of_profile}')
        logger.info(f'Number of meridional grids: {n_lat_grid}')
        logger.info(f'Number of zonal grids: {n_lon_grid}')
        logger.info(f'Number of channel: {n_channel}')

        # Create default hparams.yml for train, test and estimate
        utils.create_default_hparams(
            aggregator.hparams,
            Path(aggregator.output_dir).joinpath('hparams.yml'),
            n_lat_grid, n_lon_grid, n_channel,
            n_layers_of_profile,
            aggregator.hparams['preprocess']['reference_date'],
            aggregator.hparams['argo_selection'],
            aggregator.hparams['preprocess']['crop']
        )

        logger.info(f'DATA-ID: {aggregator.data_id}')
        logger.info('End dataset of Impulso')


    def prepare(self):
        logger.info('Begin prepare of Impuslo')

        preparer = Preparer()
        preparer.session(self.args.data_id, preparer.experiment_id)

        logger.info(f'EXPERIMENT-ID: {preparer.experiment_id}')
        logger.info('End prepare of Impulso')


    def train(self):
        logger.info('Begin train of Impulso')

        Trainer = importlib.import_module(f'experiments.{self.args.experiment_id}.src.Trainer')
        trainer = Trainer.Trainer(self.hparams['train'], self.args.experiment_id)

        # Load data
        """
        もし圧力のデータが必要なら，returnの第二引数を取得する
        """
        argo_info, _, argo_obj, map_data = utils.load_data(
            Path(IMPULSO_HOME).joinpath(f'datasets/{trainer.hparams["data_id"]}'),
            trainer.hparams['objective_variable']
        )
        
        # Create DataLoader
        train_data_loader, test_data_loader = utils.create_data_loader(
            argo_info, argo_obj, map_data,
            trainer.hparams['batch_size'],
            trainer.hparams['shuffle'],
            trainer.hparams['split_random_seed']
        )

        # Train and validate
        trainer.run(train_data_loader, test_data_loader)

        logger.info('End train of Impulso')


    def inference(self):
        logger.info('Begin inference of Impulso')

        #Inferencer = importlib.import_module(f'experiments.{self.args.experiment_id}.src.Inferencer')
        #inferencer = Inferencer.Inferencer(self.hparams['inference'], self.args.model_path, self.args.x_dir, self.args.y_dir)
        from src.Inferencer import Inferencer
        inferencer = Inferencer(self.hparams['inference'], self.args.model_path, self.args.x_dir, self.args.y_dir)

        # Get array data
        input_info, input_maps, date_array = inferencer.load_data()

        # Create DataLoader
        data_loader = utils.infer_data_loader(input_info, input_maps)

        # Get pressure information
        pre_min = inferencer.hparams['interpolation']['min_pressure']
        pre_max = inferencer.hparams['interpolation']['max_pressure']
        pre_interval = inferencer.hparams['interpolation']['pressure_interval']

        # Inference
        output_generator = inferencer.infer(data_loader)

        with open(inferencer.y_dir.joinpath(f'{inferencer.hparams["objective_variable"]}.csv'), 'w') as f:

            # Write header
            f.write('Date,Days,Latitude,Longitude,Pressure,Variable\n')

            for output, date in tqdm(zip(output_generator, date_array)):
                info, profile = output
                info = info[0]
                profile = profile[0]
                for pressure, obj_value in zip(range(pre_min, pre_max+pre_interval, pre_interval), profile):
                    f.write(f'{date},{info[0]},{info[1]},{info[2]},{pressure},{obj_value}\n')

        logger.info('End inference of Impulso')


if __name__ == '__main__':

    logger.info('Prepare arguments.')
    parser = argparse.ArgumentParser()
    parser.add_argument('exec_type',
                        help='Execution type',
                        nargs=None,
                        default=None,
                        type=str,
                        choices=['dataset', 'prepare', 'train', 'test', 'inference'])
    parser.add_argument('-d', '--data_id',
                        help='Dataset ID',
                        nargs=None,
                        default=None,
                        type=str)
    parser.add_argument('-e', '--experiment_id',
                        help='Experiment ID',
                        nargs=None,
                        default=None,
                        type=str)
    parser.add_argument('-m', '--model_path',
                        help='Full path to the model file',
                        nargs=None,
                        default=None,
                        type=str)
    parser.add_argument('-n', '--n_core',
                        help='The number of CPU core',
                        nargs=None,
                        default=1,
                        type=int)
    parser.add_argument('-p', '--hparams',
                        help='Path to hparams.yml',
                        nargs=None,
                        default=None,
                        type=str)
    parser.add_argument('-x', '--x_dir',
                        help='Path to input data directory',
                        nargs=None,
                        default=None,
                        type=str)
    parser.add_argument('-y', '--y_dir',
                        help='Path to output data directory',
                        nargs=None,
                        default='',
                        type=str)
    parser.add_argument('-t', '--t_dir',
                        help='Path to ground truth data directory',
                        nargs=None,
                        default=None,
                        type=str)
    args = parser.parse_args()

    logger.info('Check args')
    if args.exec_type == 'dataset':
        # zonal/meridional_dist_in_degreeが0.5°単位になっていることのチェックを追加
        # reference dateのフォーマットのチェックを追加
        pass
    elif args.exec_type == 'prepare':
        assert args.data_id, 'DATA-ID must be specified.'
    elif args.exec_type == 'train':
        assert args.experiment_id, 'EXPERIMENT-ID must be specified.'
    elif args.exec_type == 'inference':
        assert args.model_path, 'MODEL-PATH must be specified.'
        assert args.x_dir, 'X_DIR must be specified'
    else:
        pass
    logger.info(args)
    
    logger.info('Begin main processes.')
    impulso = Impulso(args, hparams_yaml='hparams.yml')

    if args.exec_type == 'dataset':
        impulso.dataset()

    elif args.exec_type == 'prepare':
        impulso.prepare()

    elif args.exec_type == 'train':
        impulso.train()

    elif args.exec_type == 'inference':
        impulso.inference()
    
    else:
        pass

    logger.info('Finish main processes.')
