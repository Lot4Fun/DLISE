#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Import Machine Learning modules.
import os
os.environ['IMPULSO_HOME'] = os.path.dirname(os.path.abspath(__file__))

from pathlib import Path
import datetime
import argparse
import src.lib.utils as utils
from src.Aggregator import Aggregator
from src.Preparer import Preparer
from src.Trainer import Trainer
"""
from src.Estimator import Estimator
from src.Evaluator import Evaluator

from src.model.ImpulsoNet import ImpulsoNet

"""
from logging import DEBUG, INFO
from logging import getLogger, StreamHandler, FileHandler, Formatter

# 採取的には消したいライブラリ
import glob


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
        else:
            self.hparams_path = Path(IMPULSO_HOME).joinpath(f'experiments/{self.args.experiment_id}/hparams/{hparams_yaml}')
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
        print(f'  - Number of prifiles: {n_profiles}')
        print(f'  - Number of layers of a profile; {n_layers_of_profile}')
        print(f'  - Number of meridional grids: {n_lat_grid}')
        print(f'  - Number of zonal grids: {n_lon_grid}')
        print(f'  - Number of channel: {n_channel}')

        # Create default hparams.yml for train, test and estimate
        utils.create_default_hparams(
            Path(aggregator.output_dir).joinpath('hparams.yml'),
            n_lat_grid, n_lon_grid, n_channel,
            n_layers_of_profile
        )

        logger.info('End dataset of Impulso')


    def prepare(self):
        logger.info('Begin prepare of Impuslo')
        preparer = Preparer()
        preparer.session(self.args.data_id)
        logger.info('EXPERIMENT-ID: ' + preparer.experiment_id)
        logger.info('End prepare of Impulso')


    def train(self):
        logger.info('Begin train of Impulso')

        # hparamsでモデルのフルパスが指定されていれば（そのパスのモデルが存在すれば）それを読んで追加学習（-eは試行バージョン管理のために必須）
        # -eだけ,もしくはhparamsのモデルのフルパスで指定しているモデルが存在しなければ，その試行IDで初期値から学習
        trainer = Trainer(self.hparams['train'], self.args.experiment_id)
        argo_info, argo_pre, argo_obj, map = utils.load_data(
            Path(IMPULSO_HOME).joinpath(f'datasets/{train.hparams['data_id']}'),
            trainer.hparams['objective_variable']
        )
        
        train_data_loader, test_data_loader = utils.create_data_loader(
            argo_info, argo_pre, argo_obj, map,
            trainer.hparams['batch_size'],
            trainer.hparams['shuffle'],
            trainer.hparams['split_random_seed']
        )

        trainer.run(train_data_loader, test_data_loader)

        logger.info('End train of Impulso')


    def estimate(self):
        logger.info('Begin estimate of Impulso')
        # -mオプションでモデルのフルパスが指定されていれば（そのパスのモデルが存在すれば）それを読んで推論
        print('BEGIN: ESTIMATE')
        estimator = Estimator(self.args.exec_type, self.hparams, self.model, self.args.x_dir, self.args.y_dir)
        estimator.load_data()
        estimator.estimate()
        estimator.save_results()
        logger.info('End estimate of Impulso')


    def evaluate(self):
        logger.info('Begin evaluate of Impulso')
        evaluator = Evaluator(self.args.exec_type, self.hparams)
        evaluator.load_data()
        evaluator.evaluate()
        logger.info('End evaluate of Impulso')


    def load_model(self):
        logger.info('Load model')
        modeler = ImpulsoNet(self.args.exec_type, self.hparams)
        modeler.create_model()
        if self.args.experiment_id and self.args.model_id:
            models = glob.glob(os.path.join(IMPULSO_HOME, 'experiments', self.args.experiment_id, 'models', '*'))
            while models:
                model = models.pop(0)
                i_model = int(os.path.basename(model).split('.')[1].split('-')[0])
                if self.args.model_id == i_model:
                    logger.info('Load model: ' + model)
                    self.hparams[self.args.exec_type]['model'] = model
                    modeler.model = load_model(model)
                    modeler.model.summary()
        elif self.args.exec_type == 'train':
            modeler.select_optimizer()
            modeler.compile()
        else:
            pass
        self.model = modeler.model
        

if __name__ == '__main__':

    logger.info('Prepare arguments.')
    parser = argparse.ArgumentParser()
    parser.add_argument('exec_type',
                        help='Execution type',
                        nargs=None,
                        default=None,
                        type=str,
                        choices=['dataset', 'prepare', 'train', 'test', 'estimate'])
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
    parser.add_argument('-m', '--model',
                        help='Full path of model or Model ID',
                        nargs=None,
                        default=None,
                        type=int)
    parser.add_argument('-n', '--n_core',
                        help='The number of CPU core',
                        nargs=None,
                        default=1,
                        type=int)
    parser.add_argument('-x', '--x_dir',
                        help='Path to input data directory',
                        nargs=None,
                        default=None,
                        type=str)
    parser.add_argument('-y', '--y_dir',
                        help='Path to output data directory',
                        nargs=None,
                        default=None,
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
    elif args.exec_type == 'test':
        assert args.experiment_id, 'EXPERIMENT-ID must be specified.'
        assert args.model_id, 'MODEL-ID must be specified.'
    elif args.exec_type == 'predict':
        assert args.experiment_id, 'EXPERIMENT-ID must be specified.'
        assert args.model_id, 'MODEL-ID must be specified.'
        assert args.x_dir, 'X_DIR must be specified'
        assert args.y_dir, 'Y_DIR must be specified'
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
        impulso.load_model()
        impulso.train()

    elif args.exec_type == 'validate':
        assert not args.exec_type, 'validate is still disable.'

    elif args.exec_type == 'test':
        impulso.load_model()
        impulso.estimate()
        impulso.evaluate()

    elif args.exec_type == 'predict':
        impulso.load_model()
        impulso.estimate()
    
    else:
        pass

    logger.info('Finish main processes.')
