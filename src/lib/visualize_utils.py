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

    color_range = ssh_color_range(z_array, interval=0.1)

    bmap.contourf(x_array, y_array, z_array, levels=color_range, cmap='jet')
    bmap.contour(x_array, y_array, z_array, levels=color_range, linewidths=0.5, colors='black')
    bmap.fillcontinents(color='lightgray', lake_color='white')
    bmap.drawmeridians(np.arange(min_lon, max_lon, 1), labels=[0,0,0,1], fontsize=6, color='gray')
    bmap.drawparallels(np.arange(min_lat, max_lat, 1), labels=[1,0,0,0], fontsize=6, color='gray')

    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()


def ssh_color_range(z_array, interval=0.1):

    if z_array.min() < 0:
        color_min = (int(z_array.min() / interval) - 1) * interval
    else:
        color_min = (int(z_array * 10) + 1) * interval
    color_max = (int(z_array.max() / interval) + 1) * interval

    return np.arange(color_min, color_max+interval, interval)


def tem_color_range(z_array, interval=1):

    color_min = int(z_array.min())
    color_max = int(z_array.max()) + 1

    return np.arange(color_min, color_max+interval, interval)


def pre_latlon_grid(min_pre, max_pre, min_latlon, max_latlon, pre_grid_unit, latlon_grid_unit):
    max_pre += pre_grid_unit
    max_latlon += latlon_grid_unit

    pre = np.arange(min_pre, max_pre, pre_grid_unit)
    latlon = np.arange(min_latlon, max_latlon, latlon_grid_unit)

    x, y = np.meshgrid(latlon, pre)

    return x, y


def draw_vertical_cross_section(save_path, x, y, z):

    plt.figure(figsize=(20,15))
    plt.contourf(x, y, z, levels=tem_color_range(z, interval=1), cmap='jet')
    plt.contour(x, y, z, levels=tem_color_range(z, interval=1), linewidths=0.5, colors='black')
    #plt.contour(x, y, z, levels=tem_color_range(z, interval=2), linewidths=1, colors='black')

    
    plt.ylim([1000, 10])

    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    """
    __main__ is for DEBUG.
    """
