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

    lat = np.arange(min_lat, max_lat, grid_unit)
    lon = np.arange(min_lon, max_lon, grid_unit)

    x, y = np.meshgrid(lon, lat)

    return x, y


def draw_basemap(save_path, title, x_array, y_array, z_array, min_lat, max_lat, min_lon, max_lon, grid_unit):

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
    bmap.drawmeridians(np.arange(min_lon, max_lon, 1), labels=[0,0,0,1], fontsize=4, color='gray')
    bmap.drawparallels(np.arange(min_lat, max_lat, 1), labels=[1,0,0,0], fontsize=4, color='gray')

    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()


def pre_latlon_grid(min_pre, max_pre, min_latlon, max_latlon, pre_grid_unit, latlon_grid_unit):
    max_pre += pre_grid_unit
    max_latlon += latlon_grid_unit

    pre = np.arange(min_pre, max_pre, pre_grid_unit)
    latlon = np.arange(min_latlon, max_latlon, latlon_grid_unit)

    x, y = np.meshgrid(latlon, pre)

    return x, y


def draw_vertical_cross_section(save_path, x, y, z):

    plt.figure(figsize=(20,15))
    plt.contourf(x, y, z)

    
    plt.ylim([1000, 10])

    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    """
    __main__ is for DEBUG.
    """
