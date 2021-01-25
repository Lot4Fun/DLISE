#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import torch.utils.data as data


class CreateDataLoader(object):

    @classmethod
    def build_for_train(self, exec_type, config):

        data_home = config.train.input_dir

        date_list = []
        lat_list = []
        lon_list = []
        ssh_list = []
        sst_list = []
        bio_list = []
        tem_list = []
        sal_list = []

        with open(data_home + '/db.csv', 'r') as f:
            lines = f.readlines()
        
        for line in lines[1:]:

            data_id, _, date, _, _, lat, lon, data_split  = line.split(',')
            if data_split == 'test':
                continue

            date_list.append(date)
            lat_list.append(lat)
            lon_list.append(lon)
            tem_list.append(data_home + f'/temperature/{data_id}.npy')
            sal_list.append(data_home + f'/salinity/{data_id}.npy')
            ssh_list.append(data_home + f'/ssh/{data_id}.npy')
            sst_list.append(data_home + f'/sst/{data_id}.npy')
            bio_list.append(data_home + f'/bio/{data_id}.npy')

        train_date_list, valid_date_list, \
        train_lat_list, valid_lat_list, \
        train_lon_list, valid_lon_list, \
        train_tem_list, valid_tem_list, \
        train_sal_list, valid_sal_list, \
        train_ssh_list, valid_ssh_list, \
        train_sst_list, valid_sst_list, \
        train_bio_list, valid_bio_list = train_test_split(date_list, lat_list, lon_list,
                                                          tem_list, sal_list,
                                                          ssh_list, sst_list, bio_list,
                                                          random_state=config.train.split_random_seed)

        # Dataset
        train_dataset = BatchDataset(exec_type=exec_type,
                                     config=config,
                                     dates=train_date_list,
                                     lats=train_lat_list,
                                     lons=train_lon_list,
                                     sshs=train_ssh_list,
                                     ssts=train_sst_list,
                                     bios=train_bio_list,
                                     tems=train_tem_list,
                                     sals=train_sal_list)

        valid_dataset = BatchDataset(exec_type=exec_type,
                                     config=config,
                                     dates=valid_date_list,
                                     lats=valid_lat_list,
                                     lons=valid_lon_list,
                                     sshs=valid_ssh_list,
                                     ssts=valid_sst_list,
                                     bios=valid_bio_list,
                                     tems=valid_tem_list,
                                     sals=valid_sal_list)

        # Data loader
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=config.train.batch_size,
                                       shuffle=config.train.shuffle)
        valid_loader = data.DataLoader(valid_dataset,
                                       batch_size=config.train.batch_size,
                                       shuffle=False)

        return train_loader, valid_loader


    @classmethod
    def build_for_evaluate(self, exec_type, config):

        data_home = config.evaluate.input_dir

        date_list = []
        lat_list = []
        lon_list = []
        ssh_list = []
        sst_list = []
        bio_list = []
        pre_list = []
        tem_list = []
        sal_list = []

        with open(data_home + '/db.csv', 'r') as f:
            lines = f.readlines()
        
        for line in lines[1:]:

            data_id, _, date, _, _, lat, lon, data_split  = line.split(',')
            if data_split == 'train_val':
                continue

            date_list.append(date)
            lat_list.append(lat)
            lon_list.append(lon)
            pre_list.append(data_home + f'/pressure/{data_id}.npy')
            tem_list.append(data_home + f'/temperature/{data_id}.npy')
            sal_list.append(data_home + f'/salinity/{data_id}.npy')
            ssh_list.append(data_home + f'/ssh/{data_id}.npy')
            sst_list.append(data_home + f'/sst/{data_id}.npy')
            bio_list.append(data_home + f'/bio/{data_id}.npy')

        # Dataset
        eval_dataset = BatchDatasetWithPressure(exec_type=exec_type,
                                                config=config,
                                                dates=date_list,
                                                lats=lat_list,
                                                lons=lon_list,
                                                sshs=ssh_list,
                                                ssts=sst_list,
                                                bios=bio_list,
                                                pres=pre_list,
                                                tems=tem_list,
                                                sals=sal_list)

        # Data loader
        eval_loader = data.DataLoader(eval_dataset,
                                      batch_size=1,
                                      shuffle=False)

        return eval_loader


    @classmethod
    def build_for_predict(self, exec_type, config, dates, lats, lons, sshs, ssts, bios):

        dataset = BatchDataset(exec_type=exec_type,
                               config=config,
                               dates=dates,
                               lats=lats,
                               lons=lons,
                               sshs=sshs,
                               ssts=ssts,
                               bios=bios)

        data_loader = data.DataLoader(dataset,
                                      batch_size=1,
                                      shuffle=False)
        
        return data_loader


class BatchDataset(torch.utils.data.Dataset):

    def __init__(self, exec_type, config, dates, lats, lons, sshs, ssts, bios, tems=None, sals=None):

        self.exec_type = exec_type
        self.resize = config.model.input_size
        self.resize_method = config.train.resize_method
        self.objective = config.model.objective

        self.dats = dates
        self.lats = lats
        self.lons = lons
        self.sshs = sshs
        self.ssts = ssts
        self.bios = bios
        self.tems = tems
        self.sals = sals

    def __len__(self):

        return len(self.dats)

    def __getitem__(self, idx):

        if self.exec_type == 'train':
            dat = self.dats[idx]
            lat = torch.from_numpy(np.array([float(self.lats[idx])]).astype(np.float32)).clone()
            lon = torch.from_numpy(np.array([float(self.lons[idx])]).astype(np.float32)).clone()
            ssh = torch.from_numpy(np.load(self.sshs[idx], allow_pickle=True).data.astype(np.float32)).clone()
            sst = torch.from_numpy(np.load(self.ssts[idx], allow_pickle=True).data.astype(np.float32)).clone()
            bio = torch.from_numpy(np.load(self.bios[idx], allow_pickle=True).data.astype(np.float32)).clone()
            tem = torch.from_numpy(np.load(self.tems[idx]).astype(np.float32)).clone()
            sal = torch.from_numpy(np.load(self.sals[idx]).astype(np.float32)).clone()
        else:
            dat = self.dats[idx]
            lat = torch.from_numpy(np.array([float(self.lats[idx])]).astype(np.float32)).clone()
            lon = torch.from_numpy(np.array([float(self.lons[idx])]).astype(np.float32)).clone()
            ssh = torch.from_numpy(self.sshs[idx].data.astype(np.float32)).clone()
            sst = torch.from_numpy(self.ssts[idx].data.astype(np.float32)).clone()
            bio = torch.from_numpy(self.bios[idx].data.astype(np.float32)).clone()
            tem = torch.Tensor()
            sal = torch.Tensor()

        # Concatenate input and output
        in_map = torch.stack((ssh, sst, bio), dim=0).unsqueeze(0)

        # Resize ssh and sst
        in_map = F.interpolate(in_map, size=(self.resize, self.resize), mode=self.resize_method, align_corners=False).squeeze(0)

        if self.objective == 'salinity':
            return dat, lat, lon, in_map, sal
        else:
            return dat, lat, lon, in_map, tem
        
class BatchDatasetWithPressure(torch.utils.data.Dataset):

    def __init__(self, exec_type, config, dates, lats, lons, sshs, ssts, bios, pres=None, tems=None, sals=None):

        self.exec_type = exec_type
        self.resize = config.model.input_size
        self.resize_method = config.train.resize_method
        self.objective = config.model.objective

        self.dats = dates
        self.lats = lats
        self.lons = lons
        self.sshs = sshs
        self.ssts = ssts
        self.bios = bios
        self.pres = pres
        self.tems = tems
        self.sals = sals

    def __len__(self):

        return len(self.dats)

    def __getitem__(self, idx):

        dat = self.dats[idx]
        lat = torch.from_numpy(np.array([float(self.lats[idx])]).astype(np.float32)).clone()
        lon = torch.from_numpy(np.array([float(self.lons[idx])]).astype(np.float32)).clone()
        ssh = torch.from_numpy(np.load(self.sshs[idx], allow_pickle=True).data.astype(np.float32)).clone()
        sst = torch.from_numpy(np.load(self.ssts[idx], allow_pickle=True).data.astype(np.float32)).clone()
        bio = torch.from_numpy(np.load(self.bios[idx], allow_pickle=True).data.astype(np.float32)).clone()
        pre = torch.from_numpy(np.load(self.pres[idx]).astype(np.float32)).clone()
        tem = torch.from_numpy(np.load(self.tems[idx]).astype(np.float32)).clone()
        sal = torch.from_numpy(np.load(self.sals[idx]).astype(np.float32)).clone()

        # Concatenate input and output
        in_map = torch.stack((ssh, sst, bio), dim=0).unsqueeze(0)

        # Resize ssh and sst
        in_map = F.interpolate(in_map, size=(self.resize, self.resize), mode=self.resize_method, align_corners=False).squeeze(0)

        # Get filename
        filename = self.pres[idx].split('/')[-1].split('.')[0]

        if self.objective == 'salinity':
            return dat, lat, lon, in_map, pre, sal, filename
        else:
            return dat, lat, lon, in_map, pre, tem, filename
        


if __name__ == '__main__':
    pass
