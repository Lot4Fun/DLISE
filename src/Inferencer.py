#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
from pathlib import Path
import numpy as np
import netCDF4
from .lib import utils as utils
"""
from .lib import visualizer
"""
import torch
from .model.CNN import DLModel

from logging import DEBUG
from logging import getLogger

logger = getLogger('impulso')
logger.setLevel(DEBUG)

# Set HOME directory.
IMPULSO_HOME = os.environ['IMPULSO_HOME']


class Inferencer(object):

    def __init__(self, hparams, model_path, x_dir, y_dir=''):
        logger.info('Begin init of Inferencer')
        self.hparams = hparams
        self.model_path = model_path
        self.infer_id = utils.issue_id()
        self.x_dir = Path(x_dir)

        # Set output directory
        if y_dir:
            self.y_dir = Path(y_dir)
        else:
            self.y_dir = Path(IMPULSO_HOME).joinpath(f'tmp/{self.infer_id}')
        os.makedirs(self.y_dir, exist_ok=True)

        logger.info('End init of Inferencer')
    
    
    def get_array_data(self, netcdf_path,
                       min_lat, max_lat, min_lon, max_lon,
                       lat_dist_in_degree, lon_dist_in_degree,
                       data_type='ssh', grid_unit_in_degree=0.25):
        """
        Get cropped map list
        Args:
            data_type : 'ssh' or 'sst'

            Ref. argo_info.append([n_days_elapsed, argo_lat, argo_lon])
        """
        logger.info(f'Creating input data of {data_type.upper()} for inference...')

        cropped_maps = []
        days_and_location = []
        dates = []

        # Get date information
        current_date = netcdf_path.stem[-8:]
        days_elapsed = utils.calc_days_elapsed(current_date, self.hparams['reference_date'])

        # Load SSH/SST
        map_nc = netCDF4.Dataset(netcdf_path, 'r')

        for lat in np.arange(min_lat, max_lat+grid_unit_in_degree, grid_unit_in_degree):
            min_lat_idx, max_lat_idx = utils.get_minmax_index_from_degree(lat, lat_dist_in_degree, 'latitude')
            for lon in np.arange(min_lon, max_lon+grid_unit_in_degree, grid_unit_in_degree):
                min_lon_idx, max_lon_idx = utils.get_minmax_index_from_degree(lon, lon_dist_in_degree, 'longitude')

                if data_type == 'ssh':
                    cropped = map_nc.variables['zos'][0, min_lat_idx:max_lat_idx+1, min_lon_idx:max_lon_idx+1]
                elif data_type == 'sst':
                    cropped = map_nc.variables['thetao'][0, 0, min_lat_idx:max_lat_idx+1, min_lon_idx:max_lon_idx+1]
                else:
                    logger.info('Map data type is not appropriate. Use default type (SSH)')
                    cropped = map_nc.variables['zos'][0, min_lat_idx:max_lat_idx+1, min_lon_idx:max_lon_idx+1]

                # Fill missing values
                cropped[cropped.mask] = 0.0
                
                # Store data
                cropped_maps.append(cropped)
                days_and_location.append([days_elapsed, lat, lon])
                dates.append(current_date)
        
        return cropped_maps, days_and_location, dates


    def load_data(self):
        logger.info('Loading data...')

        ssh_cropped = []
        sst_cropped = []
        input_info  = []

        ssh_files = self.x_dir.joinpath('ssh').glob('*.nc')
        sst_files = self.x_dir.joinpath('sst').glob('*.nc')
        
        # Create input data
        for ssh_file, sst_file in zip(ssh_files, sst_files):

            # Check date match between two files
            if not (ssh_file.stem[-8:] == sst_file.stem[-8:]):
                continue

            ssh_crop_list, _, _= self.get_array_data(
                ssh_file,
                self.hparams['region']['latitude']['min'],
                self.hparams['region']['latitude']['max'],
                self.hparams['region']['longitude']['min'],
                self.hparams['region']['longitude']['max'],
                self.hparams['crop_size']['meridional_distance_in_degree'],
                self.hparams['crop_size']['zonal_distance_in_degree'],
                data_type='ssh', grid_unit_in_degree=0.25)

            sst_crop_list, info, dates = self.get_array_data(
                sst_file,
                self.hparams['region']['latitude']['min'],
                self.hparams['region']['latitude']['max'],
                self.hparams['region']['longitude']['min'],
                self.hparams['region']['longitude']['max'],
                self.hparams['crop_size']['meridional_distance_in_degree'],
                self.hparams['crop_size']['zonal_distance_in_degree'],
                data_type='sst', grid_unit_in_degree=0.25)

            ssh_cropped.extend(ssh_crop_list)
            sst_cropped.extend(sst_crop_list)
            input_info.extend(info)

            # Pack SSH and SST
            logger.info('Packing SSH and SST...')
            input_maps = []
            for idx in range(len(ssh_cropped)):
                input_maps.append([ssh_cropped[idx], sst_cropped[idx]])

        return np.array(input_info), np.array(input_maps), np.array(dates)


    def infer(self, data_loader):

        # GPU setting
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Device information: {device}')

        # Create initial model
        model = DLModel(
            self.hparams['input_c'], self.hparams['input_h'], self.hparams['input_w'],
            self.hparams['output_n'],
            conv_kernel=3, max_pool_kernel=2
        ).to(device)
        logger.info(model)
        
        # Load pre-trained model
        logger.info(f'Loading model: {self.model_path}')
        model.load_state_dict(torch.load(self.model_path))

        # Set as estimation mode
        model.eval()

        # Not use gradient for inference
        with torch.no_grad():

            # Infer each input data
            for infos, maps in data_loader:

                # Send data to GPU dvice
                infos = infos.to(device)
                maps = maps.to(device)

                # Forward propagation
                output = model(maps, infos)

                yield infos.to('cpu'), output.to('cpu')

if __name__ == '__main__':
    """add"""
