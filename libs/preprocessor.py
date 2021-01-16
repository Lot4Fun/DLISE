#!/usr/bin/python3
# -*- coding: utf-8 -*-

from scipy import interpolate

class Preprocessor(object):

    def __init__(self, config):
        
        self.config = config


    def parse_argo_header(self, header):
        
        argo_date = header[20:28]
        argo_lat = float(header[29:36])
        argo_lon = float(header[37:44])
        n_layer = int(header[44:48])

        return argo_date, argo_lat, argo_lon, n_layer


    def interpolate_by_akima(self, pre_profile, obj_profile, min_pressure, max_pressure, interval):
        func = interpolate.Akima1DInterpolator(pre_profile, obj_profile)
        return func(range(min_pressure, max_pressure+interval, interval))


    def check_lat_and_lon(self, argo_lat, argo_lon):
        lat_min = self.hparams['argo_selection']['latitude']['min']
        lat_max = self.hparams['argo_selection']['latitude']['max']
        lon_min = self.hparams['argo_selection']['longitude']['min']
        lon_max = self.hparams['argo_selection']['longitude']['max']

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


    def check_file_existance(self, argo_date, files):
        for file in files:
            if 'dm' + argo_date in file.name:
                return file
        return False


    def crop_map(self, argo_lat, argo_lon, map_file, data_type='ssh'):
        # Round Argo's latitude and longitude to 0.25 units
        argo_lat = utils.round_location_in_grid(argo_lat)
        argo_lon = utils.round_location_in_grid(argo_lon)

        zonal_dist = int(self.hparams['preprocess']['crop']['zonal_distance_in_degree'])
        meridional_dist = int(self.hparams['preprocess']['crop']['meridional_distance_in_degree'])

        # Get min/max index of latitude and longitude
        lat_min_idx, lat_max_idx = utils.get_minmax_index_from_degree(argo_lat, meridional_dist, 'latitude')
        lon_min_idx, lon_max_idx = utils.get_minmax_index_from_degree(argo_lon, zonal_dist, 'longitude')

        # Load data
        map_nc = netCDF4.Dataset(map_file, 'r')

        # Crop
        if data_type == 'ssh':
            cropped = map_nc.variables['zos'][0, lat_min_idx:lat_max_idx+1, lon_min_idx:lon_max_idx+1]
        elif data_type == 'sst':
            cropped = map_nc.variables['thetao'][0, 0, lat_min_idx:lat_max_idx+1, lon_min_idx:lon_max_idx+1]
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