#!/usr/bin/python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from logging import getLogger
import cv2
import torch

logger = getLogger('DLISE')

class Predictor(object):

    def __init__(self, model, device, config, save_dir):
        
        self.model = model
        self.device = device
        self.config = config
        self.save_dir = save_dir.joinpath('predicted')

        self.save_dir.mkdir(exist_ok=True, parents=True)


    def run(self, data_loader):

        logger.info('Begin detection.')
        self.model.eval()
        with torch.no_grad():

            detected_list = [] if self.config.detect.save_results else None

            n_detected = 0
            for img_path, img_h, img_w, img in data_loader:

                # Convert tuple of length 1 to string
                img_path = img_path[0]

                if self.device.type == 'cuda':
                    img = img.to(self.device)

                detected = self.model(img).to('cpu')
                scale_factors = torch.Tensor([img_w, img_h, img_w, img_h])

                if self.config.detect.save_results:
                    detected_list.append(detected.tolist())

                if self.config.detect.visualize:
                    self._visualize(img_path, detected, scale_factors)
                
                n_detected += 1
                if not (n_detected % 100):
                    logger.info(f'Progress: [{n_detected:08}/{len(data_loader.dataset):08}]')

        if self.config.detect.save_results:
            with open(str(self.save_dir.parent.joinpath('detected.json')), 'w') as f:
                json.dump(detected_list, f, ensure_ascii=False, indent=4)

        logger.info('Detection has finished.')


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
