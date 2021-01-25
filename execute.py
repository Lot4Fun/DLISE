#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Set environment variable
import os
os.environ['DLISE_HOME'] = os.path.dirname(os.path.abspath(__file__))
DLISE_HOME = os.environ['DLISE_HOME']

# Standard library
import argparse
import datetime
import json
from logging import DEBUG, INFO
from logging import getLogger, StreamHandler, FileHandler, Formatter
from pathlib import Path
import re
import shutil

# Third party library
from attrdict import AttrDict
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Original library
from config import Config
from model.dlise import DLISE
from utils.common import CommonUtils

# Set logger
log_date = datetime.datetime.today().strftime('%Y%m%d')
log_path = Path(DLISE_HOME).joinpath(f'log/{log_date}.log')
log_path.parent.mkdir(exist_ok=True, parents=True)
logger = getLogger('DLISE')
logger.setLevel(DEBUG)
# Set handler
stream_handler = StreamHandler()
file_handler = FileHandler(log_path)
# Set log level
stream_handler.setLevel(INFO)
file_handler.setLevel(DEBUG)
# Set log format
handler_format = Formatter('%(asctime)s %(name)s %(levelname)s : %(message)s')
stream_handler.setFormatter(handler_format)
file_handler.setFormatter(handler_format)
# Add handler
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

class Executor(object):

    def __init__(self, exec_type, config=None, y_dir=None):

        assert exec_type in ['preprocess', 'train', 'evaluate', 'predict', 'visualize'], 'exec_type is not correct.'

        self.exec_type = exec_type
        if config:
            self.config = config
        else:
            self.config = Config().build_config()
    
        # Prepare
        issue_id = CommonUtils().issue_id()
        if self.exec_type == 'preprocess':
            if self.config.preprocess.save_dir:
                self.save_dir = Path(self.config.preprocess.save_dir)
            else:
                self.save_dir = Path(DLISE_HOME).joinpath('data_storage', issue_id)
        elif self.exec_type == 'train':
            if self.config.train.save_dir:
                self.save_dir = Path(self.config.train.save_dir)
            else:
                self.save_dir = Path(DLISE_HOME).joinpath('results', 'train', issue_id)
        else:
            if y_dir:
                self.save_dir = Path(y_dir)
            else:
                self.save_dir = Path(DLISE_HOME).joinpath('results', exec_type, issue_id)

        logger.info(f'Save directory: {self.save_dir}')
        CommonUtils.prepare(self.exec_type, self.config, self.save_dir)


    def preprocess(self, n_process=None):

        from libs.preprocessor import Preprocessor

        preprocessor = Preprocessor(self.config)

        ssh_files = list(Path(self.config.preprocess.ssh_input_dir).glob('*.nc'))
        sst_files = list(Path(self.config.preprocess.sst_input_dir).glob('*.nc'))
        bio_files = list(Path(self.config.preprocess.bio_input_dir).glob('*.nc'))
        arg_files = list(Path(self.config.preprocess.argo_input_dir).glob('*.txt'))

        # Prepare a file to save argo information
        with open(self.save_dir.joinpath('db.csv'), 'w') as f:
            f.write('data_id,wmo_id,date,latitude,longitude,rounded_latitude,rounded_longitude,data_split\n')

        # Interpolate Argo profile by Akima method and crop related SSH/SST
        argo_id = 0
        for arg_file in tqdm(arg_files):
            
            # Read all lines
            with open(arg_file, 'r') as f:
                lines = f.readlines()

            # Reverse lines for pop() at the end of lines
            #   - pop() at the begging of list is too slow
            lines.reverse()

            # Begin reading profiles
            while lines:

                # Get profile information
                header = lines.pop()
                wmo_id, argo_date, argo_lat, argo_lon, n_layer = preprocessor.parse_argo_header(header)

                # Get flags to check date and location of Argo and SSH/SST
                is_in_region = preprocessor.check_lat_and_lon(argo_lat, argo_lon)
                within_the_period = preprocessor.check_period(
                    argo_date,
                    self.config.preprocess.argo.date_min,
                    self.config.preprocess.argo.date_max
                )
                ssh_file = preprocessor.check_file_existance('ssh', argo_date, ssh_files)
                sst_file = preprocessor.check_file_existance('sst', argo_date, sst_files)
                bio_file = preprocessor.check_file_existance('bio', argo_date, bio_files)

                # Skip a profile if some related data do not exist
                if not (is_in_region and within_the_period and ssh_file and sst_file and bio_file):
                    for _ in range(n_layer + 2):
                        lines.pop()
                    continue

                # Skip line with data label (line of 'pr sa te')
                lines.pop()

                # Get parameters of a profile
                pre_profile, sal_profile, tem_profile = [], [], []
                for _ in range(n_layer):
                    line = lines.pop()
                    pre, sal, tem = map(float, re.split(' +', line.replace('\n', '').lstrip(' ')))
                    pre_profile.append(pre)
                    sal_profile.append(sal)
                    tem_profile.append(tem)

                # Interpolate a profile by Akima method
                pre_min = self.config.preprocess.interpolation.pre_min
                pre_max = self.config.preprocess.interpolation.pre_max
                pre_interval = self.config.preprocess.interpolation.pre_interval
                pre_interpolated = list(range(pre_min, pre_max+pre_interval, pre_interval))

                if pre_profile[0] > pre_interpolated[0] or pre_profile[-1] < pre_interpolated[-1]:
                    # If extrapolation exists, do not use this profile
                    lines.pop()
                    continue
                else:
                    sal_interpolated = preprocessor.interpolate_by_akima(pre_profile, sal_profile, pre_min, pre_max, pre_interval)
                    tem_interpolated = preprocessor.interpolate_by_akima(pre_profile, tem_profile, pre_min, pre_max, pre_interval)

                # Crop SSH/SST
                cropped_ssh = preprocessor.crop_map(argo_lat, argo_lon, ssh_file, 'ssh')
                cropped_sst = preprocessor.crop_map(argo_lat, argo_lon, sst_file, 'sst')
                cropped_bio = preprocessor.crop_map(argo_lat, argo_lon, bio_file, 'bio')

                # Make argo location grid location
                round_argo_lat = preprocessor.round_location_in_grid(argo_lat)
                round_argo_lon = preprocessor.round_location_in_grid(argo_lon)

                # Increment argo id to save all data with unique id
                argo_id += 1

                # Store header data of Argo profile
                with open(self.save_dir.joinpath('db.csv'), 'a') as f:
                    f.write(
                        str(argo_id).zfill(7)+','+
                        str(wmo_id)+','+
                        str(argo_date)+','+
                        str(argo_lat)+','+
                        str(argo_lon)+','+
                        str(round_argo_lat)+','+
                        str(round_argo_lon)+','+
                        ('train_val' if pd.to_datetime(argo_date) <= pd.to_datetime(self.config.preprocess.end_of_train) else 'test')+'\n'
                    )

                # Store profiles
                np.save(self.save_dir.joinpath('pressure', str(argo_id).zfill(7)+'.npy'), np.array(pre_interpolated))
                np.save(self.save_dir.joinpath('temperature', str(argo_id).zfill(7)+'.npy'), np.array(tem_interpolated))
                np.save(self.save_dir.joinpath('salinity', str(argo_id).zfill(7)+'.npy'), np.array(sal_interpolated))

                # Store SSH/SST
                cropped_ssh.dump(self.save_dir.joinpath('ssh', str(argo_id).zfill(7)+'.npy'))
                cropped_sst.dump(self.save_dir.joinpath('sst', str(argo_id).zfill(7)+'.npy'))
                cropped_bio.dump(self.save_dir.joinpath('bio', str(argo_id).zfill(7)+'.npy'))

                # Skip separater (line of '**')
                lines.pop()


    def load_model(self, gpu_id=None):

        # Check the number of GPU
        on_multi_gpu = True if len(gpu_id.split(',')) > 1 else False
        self.config.model.n_gpu = len(gpu_id.split(','))

        # GPU setting
        if torch.cuda.is_available():
            if on_multi_gpu:
                gpu_ids = list(map(int, gpu_id.split(',')))
                device = torch.device(f'cuda:{gpu_ids[0]}')
                logger.info(f'Use multi GPUs: {gpu_ids}')
            else:
                device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cpu')
        logger.info(f"Device information: {device}")

        # Create initial model
        model = DLISE(self.exec_type, self.config)
        # Multi-GPU mode
        if on_multi_gpu:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        # Model to GPU
        model = model.to(device)
        logger.info(model)
        
        # Load pre-trained weight
        if self.exec_type == 'train':
            weight_path = self.config.train.resume_weight_path
        elif self.exec_type == 'evaluate':
            weight_path = self.config.evaluate.trained_weight_path
        else:
            weight_path = self.config.predict.trained_weight_path

        if Path(weight_path).exists() and Path(weight_path).suffix == '.pth':
            if on_multi_gpu:
                model.module.load_weights(weight_path)
            else:
                model.load_weights(weight_path)
            logger.info(f'Loaded pretrained weight: {weight_path}')
        else:
            if on_multi_gpu:
                model.module.init_weights()
            else:
                model.init_weights()
            logger.info('Use initial weights.')

        return model, device


    def train(self, model, device):

        from libs.trainer import Trainer
        from utils.data_loader import CreateDataLoader
        from utils.common import CommonUtils

        train_loader, validate_loader = CreateDataLoader.build_for_train(self.exec_type, self.config)

        trainer = Trainer(model, device, self.config, self.save_dir)
        trainer.run(train_loader, validate_loader)


    def evaluate(self, model, device):

        from libs.evaluator import Evaluator
        from utils.data_loader import CreateDataLoader
        from utils.common import CommonUtils

        self.save_dir.joinpath(self.config.evaluate.objective).mkdir(exist_ok=True, parents=True)

        eval_loader = CreateDataLoader.build_for_evaluate(self.exec_type, self.config)

        evaluator = Evaluator(model, device, self.config, self.save_dir)
        evaluator.run(eval_loader)


    def predict(self, trained_model, device, x_dir):

        from libs.predictor import Predictor
        from utils.data_loader import CreateDataLoader

        x_dir = Path(x_dir)
        predictor = Predictor(trained_model, device, self.config, self.save_dir)

        # Load netCDF files
        dates, pred_db, ssh_paths, sst_paths, bio_paths = predictor.load_netcdf(x_dir)

        # Copy netCDFs to save directory
        for date, ssh_path, sst_path, bio_path in zip(dates, ssh_paths, sst_paths, bio_paths):
            
            tmp_save_dir = self.save_dir.joinpath('predicted', date)
            tmp_save_dir.mkdir(exist_ok=True, parents=True)

            shutil.copy2(ssh_path, tmp_save_dir.joinpath('ssh.nc'))
            shutil.copy2(sst_path, tmp_save_dir.joinpath('sst.nc'))
            shutil.copy2(bio_path, tmp_save_dir.joinpath('bio.nc'))

            with open(tmp_save_dir.joinpath('input_files.txt'), 'w') as f:
                f.write('SSH: ' + str(ssh_path) + '\n')
                f.write('SST: ' + str(sst_path) + '\n')
                f.write('BIO: ' + str(bio_path) + '\n')

        # Crop
        dates, lats, lons, sshs, ssts, bios = predictor.crop(dates, pred_db, ssh_paths, sst_paths, bio_paths)

        # Build DataLoader
        data_loader = CreateDataLoader.build_for_predict(self.exec_type, self.config, dates, lats, lons, sshs, ssts, bios)

        # Predict and save(optional) for each element
        predictor.run(data_loader)    


    def visualize(self):

        from libs.visualizer import Visualizer
        from utils.data_loader import CreateDataLoader

        visualizer = Visualizer(self.config, self.save_dir)

        input_dir = Path(self.config.visualize.predicted_dir)

        # Columns: seq_id,data_id,date,latitude,longitude
        db = pd.read_csv(input_dir.joinpath('db.csv'), dtype={'data_id': str, 'date': str})

        # Loop for each config
        for obj in self.config.visualize.objectives:

            logger.info(f'Visualizing on {obj.date} ...')

            # Draw maps
            if obj.map.draw:

                for data_type in ['ssh', 'sst', 'bio']:

                    nc = visualizer.load_netcdf(input_dir, obj.date, data_type)
                    visualizer.draw_map(nc, obj, data_type)

                    ##### [ToDo] Draw each section's line on maps (option)

            # Draw vertical sections
            for sec_info in obj.zonal_sections:

                x, y, profiles = visualizer.prepare_section('zonal', input_dir, db, obj.date, sec_info)

                # Set output path
                lat_info = 'zonal_lat=' + str(sec_info.lat)
                lon_info = 'lon_' + str(sec_info.lon_min) + '_' + str(sec_info.lon_max)

                filename = '_'.join([lat_info, lon_info]) + '.png'
                save_path = self.save_dir.joinpath(obj.date, filename)

                visualizer.draw_section(save_path, x, y, profiles)

            for sec_info in obj.meridional_sections:

                x, y, profiles = visualizer.prepare_section('meridional', input_dir, db, obj.date, sec_info)

                # Set output path
                lat_info = 'meridional_lon=' + str(sec_info.lon)
                lon_info = 'lat_' + str(sec_info.lat_min) + '_' + str(sec_info.lat_max)

                filename = '_'.join([lat_info, lon_info]) + '.png'
                save_path = self.save_dir.joinpath(obj.date, filename)

                visualizer.draw_section(save_path, x, y, profiles)

            ##### [ToDo] Draw all profiles


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('exec_type',
                        help='Execution type',
                        nargs=None,
                        default=None,
                        type=str,
                        choices=['preprocess', 'train', 'evaluate', 'predict', 'visualize'])
    parser.add_argument('-c', '--config',
                        help='Path to config.json',
                        nargs=None,
                        default=None,
                        type=str)
    parser.add_argument('-g', '--gpu_id',
                        help='GPU ID',
                        nargs=None,
                        default='0',
                        type=str)
    parser.add_argument('-n', '--n_core',
                        help='The number of CPU corefor preprocessing',
                        nargs=None,
                        default=4,
                        type=int)
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
    args = parser.parse_args()

    # Validate arguments
    if args.exec_type == 'predict':
        assert args.config, 'Configuration file is not specified.'
        assert args.x_dir, 'Input directory is not specified.'

    logger.info(f'Begin DLISE in {args.exec_type.upper()} mode')
    logger.info(f'Log file: {str(log_path)}')

    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        logger.info(config)
        config = AttrDict(config)
    else:
        config = Config().build_config()
    executor = Executor(args.exec_type, config, args.y_dir)

    if args.exec_type == 'preprocess':
        executor.preprocess(n_process=args.n_core)
    elif args.exec_type == 'visualize':
        executor.visualize()
    else:
        model, device = executor.load_model(args.gpu_id)

        if args.exec_type == 'train':
            executor.train(model, device)
        elif args.exec_type == 'evaluate':
            executor.evaluate(model, device)
        elif args.exec_type == 'predict':
            executor.predict(model, device, args.x_dir)
