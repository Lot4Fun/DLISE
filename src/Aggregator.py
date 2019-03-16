#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
from pathlib import Path
import re
import numpy as np
from tqdm import tqdm
from .lib import utils
import pickle
from scipy import interpolate

from logging import DEBUG
from logging import getLogger

logger = getLogger('impulso')
logger.setLevel(DEBUG)

# Set HOME directory.
IMPULSO_HOME = os.environ['IMPULSO_HOME']


class Aggregator(object):

    def __init__(self, exec_type, hparams):
        logger.info('Initialize Aggregator')
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


    def generate_dataset(self):

        argo_info = []
        pre_profiles = []
        sal_profiles = []
        tem_profiles = []

        # Interpolate Argo profile by Akima method and crop related SSH/SST
        for file in tqdm(Path(self.hparams['arg_in_dir']).glob('**/*.txt')):
            
            # Read all lines
            with open(file, 'r') as f:
                lines = f.readlines()

            # Reverse lines for pop() at the end of lines
            # pop() at the begging of list is too slow
            lines.reverse()

            # Begin reading profiles
            while lines:

                # FOR DEBUG
                print(len(lines))

                # Get profile information
                header = lines.pop()
                argo_date = header[20:28]
                argo_lat = float(header[29:36])
                argo_lon = float(header[37:44])
                n_layer = int(header[44:48])

                # 対応するSSH/SSTが存在しなければ，1プロファイル分読み飛ばす
                is_in_region = self.check_lat_and_lon(argo_lat, argo_lon)
                
                if True: # 条件を後で追記
                    for _ in range(n_layer + 2):
                        lines.pop()
                    continue

                # Store header information of Argo profile
                argo_info.append({
                    'date': argo_date,
                    'lat': argo_lat,
                    'lon': argo_lon,
                    'n_layer': n_layer
                })

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
                pre_interpolated = list(range(10, 1000+10, 10))
                sal_interpolated = self.interpolate_by_akima(pre_profile, sal_profile, 10, 1000, 10)
                tem_interpolated = self.interpolate_by_akima(pre_profile, tem_profile, 10, 1000, 10)

                # Store profile
                pre_profiles.append(pre_interpolated)
                sal_profiles.append(sal_interpolated)
                tem_profiles.append(tem_interpolated)

                # Skip separater (line of '**')
                lines.pop()

        return np.array(argo_info), np.array(pre_profiles), np.array(sal_profiles), np.array(tem_profiles)


    def interpolate_by_akima(self, pre_profile, obj_profile, min_pressure, max_pressure, interval):
        func = interpolate.Akima1DInterpolator(pre_profile, obj_profile)
        return func(range(min_pressure, max_pressure+interval, interval))


    def check_lat_and_lon(self, argo_lat, argo_lon):
        lat_min = self.hparams['lat_min']
        lat_max = self.hparams['lat_max']
        lon_min = self.hparams['lon_min']
        lon_max = self.hparams['lon_max']

        if argo_lat < lat_min or argo_lat > lat_max or argo_lon < lon_min or argo > lon_max:
            return False
        else:
            return True


    def save_as_pickle(self, obj_to_save, save_name):
        logger.info('Begin saving ' + save_name)
        
        with open(Path(self.hparams['out_dir']).joinpath(save_name), 'w') as out_argo:
            pickle.dump(obj_to_save, out_argo)
        
        logger.info('End saving ' + save_name)


if __name__ == '__main__':
    pass
