#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-

from pathlib import Path
import netCDF4

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

        visualize_utils.basemap(save_path, title, x, y, cropped, min_lat, max_lat, min_lon, max_lon, grid_unit)


    def vertical_profile(self, input_path, lat, lon):
        pass


    def vertical_cross_section(self, input_path, lat, lon):
        """
        Args:
          - 'lat' is int or float -> 'lon' is tuple or list
          - 'lat' is int or float -> 'lon' is tuple or list
        """
        assert len(lat) == 1 and len(lon) == 2
        assert len(lat) == 2 and len(lon) == 1


        pass


if __name__ == '__main__':

    visualizer = Visualizer()

    # Horizontal map
    visualizer.horizontal_map(
        'C:/Users/stokes/Documents/80_Project/github/_data_storage/internal_structure/infer_sample/ssh/metoffice_coupled_orca025_GL4_SSH_b20160403_dm20160401.nc',
        'C:/Users/stokes/Documents/80_Project/github/internal_structure/tmp/figure',
        'ssh',
        min_lat=20,
        max_lat=40,
        min_lon=140,
        max_lon=220,
        grid_unit=0.25
    )

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
 
n = 10
x = np.linspace(140, 180, n)
y = np.linspace(0, 40, n)
 
X, Y = np.meshgrid(x, y)
Z = np.sqrt(X**2 + Y**2)

bmap = Basemap(
    projection='merc',
    resolution='l',
    llcrnrlon=140,
    llcrnrlat=0,
    urcrnrlon=180,
    urcrnrlat=40)

X, Y = bmap(X, Y)

cs = bmap.contourf(X, Y, Z, linewidths=1.5)
bmap.fillcontinents(color='lightgray', lake_color='white')
bmap.drawcoastlines(color='black')

plt.show()
"""
