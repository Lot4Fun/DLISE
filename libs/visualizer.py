#!/usr/bin/python3
# -*- coding: utf-8 -*-

import json
from logging import getLogger
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import torch

logger = getLogger('DLISE')

class Predictor(object):

    def __init__(self, config, save_dir):
        
        self.config = config
        self.save_dir = save_dir

        self.save_dir.mkdir(exist_ok=True, parents=True)


    def run(self, data_loader):

        logger.info('Begin visualization.')
