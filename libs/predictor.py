#!/usr/bin/python3
# -*- coding: utf-8 -*-

import json
from logging import getLogger
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import torch

from .preprocessor import Preprocessor

logger = getLogger('DLISE')

class Predictor(object):

    def __init__(self, model, device, config, save_dir):
        
        self.preprocessor = Preprocessor(config)

        self.model = model
        self.device = device
        self.config = config
        self.save_dir = save_dir.joinpath('predicted')

        self.save_dir.mkdir(exist_ok=True, parents=True)


    def run(self, data_loader):

        logger.info('Begin prediction.')

        with open(self.save_dir.parent.joinpath('db.csv'), 'w') as f:
            f.write('seq_id,data_id,date,latitude,longitude\n')

        self.model.eval()
        with torch.no_grad():

            n_predicted = 0
            each_date_id = {}

            for date, lat, lon, in_map, _ in data_loader:

                date = date[0] # Original date is tuple

                in_map = in_map.to(self.device)
                predicted = self.model(lat, lon, in_map).to('cpu').squeeze().detach().numpy().copy()

                n_predicted += 1
                if str(date) in each_date_id.keys():
                    each_date_id[str(date)] += 1
                else:
                    each_date_id[str(date)] = 1

                if self.config.predict.save_results:

                    with open(self.save_dir.parent.joinpath('db.csv'), 'a') as f:
                        f.write(str(n_predicted).zfill(7) + ',' +
                                str(each_date_id[date]).zfill(7) + ',' + 
                                str(date) + ',' + 
                                str(lat.item()) + ',' + 
                                str(lon.item()) + '\n')

                    if not self.save_dir.joinpath(date, 'profiles').exists():
                        self.save_dir.joinpath(date, 'profiles').mkdir(exist_ok=True, parents=True)
                    np.save(self.save_dir.joinpath(date, 'profiles', str(each_date_id[date]).zfill(7)+'.npy'), predicted)
                
                if not (n_predicted % 100):
                    logger.info(f'Progress: [{n_predicted:08}/{len(data_loader.dataset):08}]')

        logger.info('Prediction has finished.')


    def load_netcdf(self, x_dir):

        logger.info('Loading netCDF files...')

        objectives = self.config.predict.objectives

        dates = []
        pred_db = []
        ssh_paths = []
        sst_paths = []
        bio_paths = []

        ssh_files = list(x_dir.joinpath('ssh').glob('*.nc'))
        sst_files = list(x_dir.joinpath('sst').glob('*.nc'))
        bio_files = list(x_dir.joinpath('bio').glob('*.nc'))

        for obj_date in objectives.keys():

            # Get the relevant file path
            ssh_file = self.preprocessor.check_file_existance('ssh', obj_date, ssh_files)
            sst_file = self.preprocessor.check_file_existance('sst', obj_date, sst_files)
            bio_file = self.preprocessor.check_file_existance('bio', obj_date, bio_files)

            if not (ssh_file and sst_file and bio_file):
                continue

            dates.append(obj_date)
            pred_db.append(objectives[obj_date])
            ssh_paths.append(ssh_file)
            sst_paths.append(sst_file)
            bio_paths.append(bio_file)

        return dates, pred_db, ssh_paths, sst_paths, bio_paths 

    def crop(self, dates, pred_db, ssh_paths, sst_paths, bio_paths):

        logger.info('Begin to crop map files.')

        crop_dates = []
        crop_lats = []
        crop_lons = []
        crop_sshs = []
        crop_ssts = []
        crop_bios = []

        for date, db, ssh_path, sst_path, bio_path in zip(dates, pred_db, ssh_paths, sst_paths, bio_paths):

            logger.info(f'Cropping maps on {date} ...')

            center_lat = (db['lat_max'] + db['lat_min']) / 2.
            center_lon = (db['lon_max'] + db['lon_min']) / 2.
            meridional_dist = db['lat_max'] - db['lat_min']
            zonal_dist = db['lon_max'] - db['lon_min']

            # Load netCDF data
            ssh = netCDF4.Dataset(ssh_path, 'r')
            sst = netCDF4.Dataset(sst_path, 'r')
            bio = netCDF4.Dataset(bio_path, 'r')

            # Get min-max latitude and longitude for the center of each crooped map
            lat_min_idx, lat_max_idx = self.preprocessor.get_minmax_index_from_degree(center_lat, meridional_dist, 'latitude')
            lon_min_idx, lon_max_idx = self.preprocessor.get_minmax_index_from_degree(center_lon, zonal_dist, 'longitude')

            # Crop SSH/SST
            for center_lat in np.arange(db['lat_min'], db['lat_max']+0.25, 0.25):

                for center_lon in np.arange(db['lon_min'], db['lon_max']+0.25, 0.25):

                    # Get min-max latitude and longitude for boundaries of each crooped map
                    lat_min_idx, lat_max_idx = self.preprocessor.get_minmax_index_from_degree(center_lat,
                                                                                              self.config.predict.crop.meridional,
                                                                                              'latitude')
                    lon_min_idx, lon_max_idx = self.preprocessor.get_minmax_index_from_degree(center_lon,
                                                                                              self.config.predict.crop.zonal,
                                                                                              'longitude')

                    # Crop
                    crop_ssh = ssh.variables['zos'][0, lat_min_idx:lat_max_idx+1, lon_min_idx:lon_max_idx+1]
                    crop_sst = sst.variables['thetao'][0, 0, lat_min_idx:lat_max_idx+1, lon_min_idx:lon_max_idx+1]
                    crop_bio = bio.variables['chl'][0, 0, lat_min_idx:lat_max_idx+1, lon_min_idx:lon_max_idx+1]

                    # Fill missilng values
                    crop_ssh[crop_ssh.mask] = 0.0
                    crop_sst[crop_sst.mask] = 0.0
                    crop_bio[crop_bio.mask] = 0.0

                    # Store data
                    crop_dates.append(date)
                    crop_lats.append(center_lat)
                    crop_lons.append(center_lon)
                    crop_sshs.append(crop_ssh)
                    crop_ssts.append(crop_sst)
                    crop_bios.append(crop_bio)

        return crop_dates, crop_lats, crop_lons, crop_sshs, crop_ssts, crop_bios
