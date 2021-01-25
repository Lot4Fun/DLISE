#!/usr/bin/python3
# -*- coding: utf-8 -*-

import json
from logging import getLogger
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import netCDF4
import numpy as np
import pandas as pd
import torch

logger = getLogger('DLISE')

class Visualizer(object):

    def __init__(self, config, save_dir):
        
        self.config = config
        self.save_dir = save_dir

        self.save_dir.mkdir(exist_ok=True, parents=True)

    def load_netcdf(self, input_dir, date, data_type):

        input_file = input_dir.joinpath('predicted', date, data_type + '.nc')
        nc = netCDF4.Dataset(input_file)

        return nc

    def draw_map(self, netcdf, obj, data_type):

        # Mesh of latitude and longitude
        lat_min_idx = self.lat_deg2idx(obj.map.lat_min)
        lat_max_idx = self.lat_deg2idx(obj.map.lat_max)
        lon_min_idx = self.lon_deg2idx(obj.map.lon_min)
        lon_max_idx = self.lon_deg2idx(obj.map.lon_max)
        x, y = self.create_grid(obj.map.lat_min, obj.map.lat_max,
                                obj.map.lon_min, obj.map.lon_max, 0.25)

        # Crop
        if data_type == 'ssh':
            cropped = netcdf.variables['zos'][0,
                                              lat_min_idx:lat_max_idx+1,
                                              lon_min_idx:lon_max_idx+1]
        elif data_type == 'sst':
            cropped = netcdf.variables['thetao'][0, 0,
                                                 lat_min_idx:lat_max_idx+1,
                                                 lon_min_idx:lon_max_idx+1]
        elif data_type == 'bio':
            cropped = netcdf.variables['chl'][0, 0,
                                              lat_min_idx:lat_max_idx+1,
                                              lon_min_idx:lon_max_idx+1]

        # Fill missing value
        cropped[cropped.mask] = 0

        # Set output path
        lat_info = f'lat_{obj.map.lat_min:.2f}-{obj.map.lat_max:.2f}'
        lon_info = f'lon_{obj.map.lon_min:.2f}-{obj.map.lon_max:.2f}'
        save_path = self.save_dir.joinpath(obj.date, '_'.join([data_type, obj.date, lat_info, lon_info]) + '.png')

        self.save_dir.joinpath(obj.date).mkdir(exist_ok=True, parents=True)

        # Set title
        if data_type == 'ssh':
            title = 'Sea Surface Height'
        elif data_type == 'sst':
            title = 'Sea Surface Temperature'
        elif data_type == 'bio':
            title = 'Sea Surface Chlorophyll'

        self.draw_basemap(save_path, data_type, obj, title, x, y, cropped,
                          obj.map.lat_min, obj.map.lat_max, obj.map.lon_min, obj.map.lon_max, 0.25)

    def lat_deg2idx(self, degree):
        return int((degree  + 83) * 4)

    def lon_deg2idx(self, degree):
        return int(degree  * 4)

    def create_grid(self, lat_min, lat_max, lon_min, lon_max, grid_unit=0.25):

        lat_max += grid_unit
        lon_max += grid_unit

        lat = np.arange(lat_min, lat_max, grid_unit)
        lon = np.arange(lon_min, lon_max, grid_unit)

        x, y = np.meshgrid(lon, lat)

        return x, y

    def draw_basemap(self, save_path, data_type, obj, title, x_array, y_array, z_array,
                     lat_min, lat_max, lon_min, lon_max, grid_unit=0.25):

        bmap = Basemap(
            projection='merc',
            resolution='h',
            llcrnrlon=lon_min,
            llcrnrlat=lat_min,
            urcrnrlon=lon_max,
            urcrnrlat=lat_max)

        plt.figure(figsize=(20,20))

        x_array, y_array = bmap(x_array, y_array)

        if data_type == 'ssh':
            color_range = self.ssh_color_range(z_array, interval=0.1)
        elif data_type == 'sst':
            color_range = self.sst_color_range(z_array, interval=0.5)
        elif data_type == 'bio':
            color_range = self.chl_color_range(z_array, interval=0.05)

        bmap.contourf(x_array, y_array, z_array, levels=color_range, cmap='jet')
        bmap.contour(x_array, y_array, z_array, levels=color_range, linewidths=0.5, colors='black')
        bmap.fillcontinents(color='lightgray', lake_color='white')
        bmap.drawmeridians(np.arange(lon_min, lon_max, 1), labels=[0,0,0,1], fontsize=6, color='gray')
        bmap.drawparallels(np.arange(lat_min, lat_max, 1), labels=[1,0,0,0], fontsize=6, color='gray')

        if obj.draw_lines_on_map:
            ##### ToDo
            pass

        plt.title(title)
        plt.savefig(save_path, bbox_inches='tight')
        plt.clf()

    def ssh_color_range(self, z_array, interval=0.1):

        if z_array.min() < 0:
            color_min = (int(z_array.min() / interval) - 1) * interval
        else:
            color_min = (int(z_array * 10) + 1) * interval
        color_max = (int(z_array.max() / interval) + 1) * interval

        return np.arange(color_min, color_max+interval, interval)

    def sst_color_range(self, z_array, interval=1):

        color_min = int(z_array.min())
        color_max = int(z_array.max()) + 1

        return np.arange(color_min, color_max+interval, interval)

    def chl_color_range(self, z_array, interval=0.05):

        color_min = int(z_array.min())
        color_max = int(z_array.max()) + 1

        return np.arange(color_min, color_max+interval, interval)

    def prepare_section(self, sec_type, input_dir, db, date, sec_info, pressure_grid_unit=10, latlon_grid_unit=0.25):

        # Set fix axis
        if sec_type == 'zonal':
            latlon_fix = sec_info.lat
            latlon_min = sec_info.lon_min
            latlon_max = sec_info.lon_max
        elif sec_type == 'meridional':
            latlon_fix = sec_info.lon
            latlon_min = sec_info.lat_min
            latlon_max = sec_info.lat_max

        # Get X-Y grid data
        x, y = self.pre_latlon_grid(sec_info.pre_min, sec_info.pre_max,
                                    latlon_min, latlon_max, 
                                    pressure_grid_unit, latlon_grid_unit)

        # Extract vertical profile data
        if sec_type == 'zonal':
            fix_axis = 'latitude'
            other_axis = 'longitude'
        elif sec_type == 'meridional':
            fix_axis = 'longitude'
            other_axis = 'latitude'
        extracted_db = db[(db['date'] == date) &
                            (db[fix_axis] == latlon_fix) &
                            (db[other_axis] >= latlon_min) &
                            (db[other_axis] <= latlon_max)]
        extracted_db = extracted_db.sort_values(other_axis)

        # Load profiles
        profiles = []
        for data_id in extracted_db['data_id']:
            profile = np.load(input_dir.joinpath('predicted', date, 'profiles', data_id+'.npy'))
            profiles.append(profile)
        profiles = np.array(profiles).T

        return x, y, profiles

    def pre_latlon_grid(self, pre_min, pre_max, latlon_min, latlon_max, pre_grid_unit, latlon_grid_unit):

        pre_max += pre_grid_unit
        latlon_max += latlon_grid_unit

        pre = np.arange(pre_min, pre_max, pre_grid_unit)
        latlon = np.arange(latlon_min, latlon_max, latlon_grid_unit)

        x, y = np.meshgrid(latlon, pre)

        return x, y

    def draw_section(self, save_path, x, y, z):

        plt.figure(figsize=(20,15))
        plt.contourf(x, y, z, levels=self.sst_color_range(z, interval=1), cmap='jet')
        plt.contour(x, y, z, levels=self.sst_color_range(z, interval=1), linewidths=0.5, colors='black')
        
        ##### [ToDo] Modify hard-coding
        plt.title('Temperature')
        plt.ylim([1000, 10])

        plt.savefig(save_path, bbox_inches='tight')
        plt.clf()

