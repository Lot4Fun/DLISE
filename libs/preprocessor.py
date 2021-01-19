#!/usr/bin/python3
# -*- coding: utf-8 -*-

from logging import getLogger
import sys

from decimal import Decimal, ROUND_HALF_UP
import netCDF4
import pandas as pd
from scipy import interpolate

logger = getLogger('DLISE')

class Preprocessor(object):

    def __init__(self, config):
        
        self.config = config

    def parse_argo_header(self, header):

        wmo_id = header[8:15]
        argo_date = header[20:28]
        argo_lat = float(header[29:36])
        argo_lon = float(header[37:44])
        n_layer = int(header[44:48])

        return wmo_id, argo_date, argo_lat, argo_lon, n_layer

    def check_lat_and_lon(self, argo_lat, argo_lon):

        lat_min = self.config.preprocess.argo.lat_min
        lat_max = self.config.preprocess.argo.lat_max
        lon_min = self.config.preprocess.argo.lon_min
        lon_max = self.config.preprocess.argo.lon_max

        if (lat_min <= argo_lat <= lat_max) and (lon_min <= argo_lon <= lon_max):
            return True
        else:
            return False

    def check_period(self, current_date, date_min, date_max):

        date_min = pd.to_datetime(date_min)
        date_max = pd.to_datetime(date_max)

        if date_min <= pd.to_datetime(current_date) <= date_max:
            return True
        else:
            return False

    def check_file_existance(self, data_type, argo_date, files):

        if data_type == 'bio':
            for file in files:
                if argo_date in file.name:
                    return file
            return False
        else:
            for file in files:
                if 'dm' + argo_date in file.name:
                    return file
            return False

    def interpolate_by_akima(self, pre_profile, obj_profile, min_pressure, max_pressure, interval):

        func = interpolate.Akima1DInterpolator(pre_profile, obj_profile)
        return func(range(min_pressure, max_pressure+interval, interval))


    def crop_map(self, argo_lat, argo_lon, map_file, data_type='ssh'):

        # Round Argo's latitude and longitude to 0.25 units
        argo_lat = self.round_location_in_grid(argo_lat)
        argo_lon = self.round_location_in_grid(argo_lon)

        zonal_dist = int(self.config.preprocess.crop.zonal)
        meridional_dist = int(self.config.preprocess.crop.meridional)

        # Get min/max index of latitude and longitude
        lat_min_idx, lat_max_idx = self.get_minmax_index_from_degree(argo_lat, meridional_dist, 'latitude')
        lon_min_idx, lon_max_idx = self.get_minmax_index_from_degree(argo_lon, zonal_dist, 'longitude')

        # Load data
        map_nc = netCDF4.Dataset(map_file, 'r')

        # Crop
        if data_type == 'ssh':
            cropped = map_nc.variables['zos'][0, lat_min_idx:lat_max_idx+1, lon_min_idx:lon_max_idx+1]
        elif data_type == 'sst':
            cropped = map_nc.variables['thetao'][0, 0, lat_min_idx:lat_max_idx+1, lon_min_idx:lon_max_idx+1]
        elif data_type == 'bio':
            cropped = map_nc.variables['chl'][0, 0, lat_min_idx:lat_max_idx+1, lon_min_idx:lon_max_idx+1]
        else:
            logger.info('Map data type is not appropriate. Use default type (SSH)')
            cropped = map_nc.variables['zos'][0, lat_min_idx:lat_max_idx+1, lon_min_idx:lon_max_idx+1]

        # Scale factor and offset
        """
        Add a process if need to use scale_factor and add_offset
          - scale_factor = map_nc.variables['zos'].scale_factor
          - add_offset = map_nc.variables['thetao'].add_offset
        """

        # Fill missilng values
        cropped[cropped.mask] = 0.0

        return cropped

    def round_location_in_grid(self, in_degree):
        """
        Round 'in_degree' in 0.25 degree units
        """
        return float(Decimal(str(in_degree * 4)).quantize(Decimal('0'), rounding=ROUND_HALF_UP) / 4)

    def get_minmax_index_from_degree(self, argo_in_degree, distance_in_degree, data_type): 
        """
        Get min-max index of a small map based on the centerl latitude and latitude
        """
        # Latitude  : index=0 -> -83.0 degree, index=691 -> 89.75 degree
        # Longitude : index=0 -> XX degree, index=1439 -> XX degree
        if data_type == 'latitude':
            min_idx = int(((argo_in_degree - distance_in_degree / 2) + 83) * 4)
            max_idx = int(((argo_in_degree + distance_in_degree / 2) + 83) * 4)
        elif data_type == 'longitude':
            min_idx = int((argo_in_degree - distance_in_degree / 2) * 4)
            max_idx = int((argo_in_degree + distance_in_degree / 2) * 4)
        else:
            sys.exit('Error in "get_minmax_index_from_degree" function. Inappropriate "data_type"')
        
        return min_idx, max_idx
