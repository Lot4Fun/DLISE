#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-

import os

import numpy as np
import matplotlib.pyplot as plt


def lat_lon_grid(min_lat, max_lat, min_lon, max_lon, grid_unit=0.25):
    max_lat += grid_unit
    max_lon += grid_unit
    x, y = np.mgrid[min_lat:max_lat:grid_unit, min_lon:max_lon:grid_unit]
    return x, y


def z_grid()

def vertical_section(x_array, y_array, z_array):
    """
    """


if __name__ == '__main__':
    """
    __main__ is for DEBUG.
    """
