#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-

from pathlib import Path
import netCDF4
import numpy as np
import pandas as pd

from lib import visualize_utils


class Visualizer(object):

    def __init__(self):
        pass


    def horizontal_map(self, input_path, save_dir, data_type, min_lat, max_lat, min_lon, max_lon, grid_unit):

        # Mesh of latitude and longitude
        min_lat_idx = visualize_utils.lat_degree2index(min_lat)
        max_lat_idx = visualize_utils.lat_degree2index(max_lat)
        min_lon_idx = visualize_utils.lon_degree2index(min_lon)
        max_lon_idx = visualize_utils.lon_degree2index(max_lon)
        x, y = visualize_utils.lat_lon_grid(min_lat, max_lat, min_lon, max_lon, grid_unit)

        # Crop
        nc = netCDF4.Dataset(input_path, 'r')
        if data_type == 'ssh':
            cropped = nc.variables['zos'][0, min_lat_idx:max_lat_idx+1, min_lon_idx:max_lon_idx+1]
        elif data_type == 'sst':
            cropped = nc.variables['thetao'][0, 0, min_lat_idx:max_lat_idx+1, min_lon_idx:max_lon_idx+1]

        cropped[cropped.mask] = 0

        # Set output path
        date = Path(input_path).stem[-8:]
        lat_info = f'lat_{min_lat:.2f}-{max_lat:.2f}'
        lon_info = f'lon_{min_lon:.2f}-{max_lon:.2f}'
        save_path = Path(save_dir).joinpath('_'.join([data_type, date, lat_info, lon_info]) + '.png')

        # Set title
        if data_type == 'ssh':
            title = 'Sea Surface Height'
        elif data_type == 'sst':
            title = 'Sea Surface Temperature'
        else:
            title = 'Horizontal Map'

        visualize_utils.draw_basemap(save_path, title, x, y, cropped, min_lat, max_lat, min_lon, max_lon, grid_unit)


    def vertical_profile(self, input_path, lat, lon):
        pass


    def vertical_cross_section(self, input_path, save_dir, date, lats, lons, min_pre, max_pre):
        """
        Args:
          - date: YYYYMMDD
          - lat: is int or float -> 'lon' is tuple or list (min_lon, max_lon)
          - lon: is int or float -> 'lat' is tuple or list (min_lat, max_lat)
        """
        assert (len(lats) == 1 and len(lons) == 2) or (len(lats) == 2 and len(lons) == 1)

        # Constant value
        PRESSURE_GRID_UNIT = 10
        LATLON_GRID_UNIT = 0.25

        # Set fix axis
        if len(lats) == 1 and len(lons) == 2:
            fix_axis = 'lat'
            fix_latlon = lats[0]
            min_latlon, max_latlon = lons
        elif len(lats) == 2 and len(lons) == 1:
            fix_axis = 'lon'
            fix_latlon = lons[0]
            min_latlon, max_latlon = lats

        # Get X-Y grid data
        x, y = visualize_utils.pre_latlon_grid(min_pre, max_pre, min_latlon, max_latlon, PRESSURE_GRID_UNIT, LATLON_GRID_UNIT)

        # Extract temperature or salinity data
        data = pd.read_csv(input_path, dtype={'Date':str})
        data = data[(data['Date']  == date) & (min_pre <= data['Pressure']) & (data['Pressure'] <= max_pre)]
        x_lenght = len(np.arange(min_latlon, max_latlon+LATLON_GRID_UNIT, LATLON_GRID_UNIT))
        y_length = len(np.arange(min_pre, max_pre+PRESSURE_GRID_UNIT, PRESSURE_GRID_UNIT))            

        if fix_axis == 'lat':
            data = data[(data['Latitude'] == fix_latlon) & (min_latlon <= data['Longitude']) & (data['Longitude'] <= max_latlon)]
            data = data.sort_values(['Pressure', 'Longitude'])
        elif fix_axis == 'lon':
            data = data[(data['Longitude'] == fix_latlon) & (min_latlon <= data['Latitude']) & (data['Latitude'] <= max_latlon)]
            data = data.sort_values(['Pressure', 'Latitude'])

        data = np.array(data['Variable']).reshape(y_length, x_lenght)

        # Set output path
        if fix_axis == 'lat':
            lat_info = 'lat_' + str(fix_latlon) + '_' + str(fix_latlon)
            lon_info = 'lon_' + str(min_latlon) + '_' + str(max_latlon)
        elif fix_axis == 'lon':
            lat_info = 'lat_' + str(min_latlon) + '_' + str(max_latlon)
            lon_info = 'lon_' + str(fix_latlon) + '_' + str(fix_latlon)

        filename = '_'.join([date, lat_info, lon_info]) + '.png'
        save_path = Path(save_dir).joinpath(filename)

        print(x.shape, y.shape, data.shape)

        visualize_utils.draw_vertical_cross_section(save_path, x, y, data)


if __name__ == '__main__':

    visualizer = Visualizer()
    """
    # SSH/SST
    visualizer.horizontal_map(
        'C:/Users/stokes/Documents/80_Project/github/_data_storage/internal_structure/infer_sample/ssh/metoffice_coupled_orca025_GL4_SSH_b20180103_dm20180101.nc',
        'C:/Users/stokes/Documents/80_Project/github/internal_structure/tmp/figure',
        'ssh',
        min_lat=20,
        max_lat=40,
        min_lon=140,
        max_lon=180,
        grid_unit=0.25
    )
    """
    # Vertical Cross Section
    visualizer.vertical_cross_section(
        input_path='C:/Users/stokes/Documents/80_Project/github/internal_structure/tmp/0324-0947-2178/temperature.csv',
        save_dir='C:/Users/stokes/Documents/80_Project/github/internal_structure/tmp/figure',
        date='20180101',
        lats=(23, ),
        lons=(152,162),
        min_pre=10,
        max_pre=1000
    )

    # Horizontal Cross Section
