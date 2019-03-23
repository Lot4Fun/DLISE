#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-

import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def lat_degree2index(degree):
    return int((degree  + 83) * 4)


def lon_degree2index(degree):
    return int(degree  * 4)


def lat_lon_grid(min_lat, max_lat, min_lon, max_lon, grid_unit=0.25):
    max_lat += grid_unit
    max_lon += grid_unit
    #grid_lon, grid_lat = np.mgrid[min_lat:max_lat:grid_unit, min_lon:max_lon:grid_unit]
    lat = np.arange(min_lat, max_lat, grid_unit)
    lon = np.arange(min_lon, max_lon, grid_unit)
    x, y = np.meshgrid(lon, lat)

    #return grid_lon, grid_lat
    return x, y


def z_grid():
    pass


def basemap(save_path, title, x_array, y_array, z_array, min_lat, max_lat, min_lon, max_lon, grid_unit):

    bmap = Basemap(
        projection='merc',
        resolution='h',
        llcrnrlon=min_lon,
        llcrnrlat=min_lat,
        urcrnrlon=max_lon,
        urcrnrlat=max_lat)

    plt.figure(figsize=(20,20))

    x_array, y_array = bmap(x_array, y_array)

    bmap.contourf(x_array, y_array, z_array)
    bmap.fillcontinents(color='lightgray', lake_color='white')
    #bmap.drawcoastlines(color='black')
    bmap.drawmeridians(np.arange(0, 360, grid_unit), color='gray')
    bmap.drawparallels(np.arange(-90, 90, grid_unit), color='gray')

    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()

    """
    # 3D-diagram
    ax_3d = fig.add_subplot(122, projection='3d')
    ax_3d.set_title('surface')
    ax_3d.plot_surface(X, Y, Z)
    """


if __name__ == '__main__':
    """
    __main__ is for DEBUG.
    """
