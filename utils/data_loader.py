#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import torch.utils.data as data

def detection_collate(batch):

    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    
    imgs = torch.stack(imgs, dim=0)

    return imgs, targets


class CreateDataLoader(object):

    @classmethod
    def build_for_train(self, config):

        data_home = config.train.input_dir

        date_list = []
        lat_list = []
        lon_list = []
        tem_list = []
        sal_list = []
        ssh_list = []
        sst_list = []
        #bio_list = []

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
            #bio_list.append(data_home + f'/bio/{data_id}.npy')

        train_date_list, valid_date_list, \
        train_lat_list, valid_lat_list, \
        train_lon_list, valid_lon_list, \
        train_tem_list, valid_tem_list, \
        train_sal_list, valid_sal_list, \
        train_ssh_list, valid_ssh_list, \
        train_sst_list, valid_sst_list = train_test_split(date_list, lat_list, lon_list, tem_list, sal_list, ssh_list, sst_list,
                                                            random_state=config.train.split_random_seed)

        # Dataset
        train_dataset = BatchDataset(dates=train_date_list,
                                     lats=train_lat_list,
                                     lons=train_lon_list,
                                     tems=train_tem_list,
                                     sals=train_sal_list,
                                     sshs=train_ssh_list,
                                     ssts=train_sst_list,
                                     config=config)

        valid_dataset = BatchDataset(dates=valid_date_list,
                                     lats=valid_lat_list,
                                     lons=valid_lon_list,
                                     tems=valid_tem_list,
                                     sals=valid_sal_list,
                                     sshs=valid_ssh_list,
                                     ssts=valid_sst_list,
                                     config=config)

        # Data loader
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=config.train.batch_size,
                                       shuffle=config.train.shuffle)
                                       #collate_fn=detection_collate)
        valid_loader = data.DataLoader(valid_dataset,
                                       batch_size=config.train.batch_size,
                                       shuffle=False)
                                       #collate_fn=detection_collate)

        return train_loader, valid_loader


    @classmethod
    def build_for_predict(self, config, x_dir):

        date_list = []
        tem_list = []
        sal_list = []
        ssh_list = []
        sst_list = []
        #bio_list = []        



        """
        inputs = [img_path for img_path in Path(x_dir).glob('*') if re.fullmatch('.jpg|.jpeg|.png', img_path.suffix.lower())]

        dataset = BatchDataset(
            inputs=inputs,
            config=config,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        return data_loader
        """
        pass

class BatchDataset(torch.utils.data.Dataset):

    def __init__(self, dates, lats, lons, tems, sals, sshs, ssts, config):

        self.dats = dates
        self.lats = lats
        self.lons = lons
        self.tems = tems
        self.sals = sals
        self.sshs = sshs
        self.ssts = ssts
        self.resize = config.model.input_size
        self.resize_method = config.train.resize_method
        self.objective = config.model.objective

    def __len__(self):

        return len(self.tems)

    def __getitem__(self, idx):

        #dat = self.dats[idx]
        lat = torch.from_numpy(np.array([float(self.lats[idx])]).astype(np.float32)).clone()
        lon = torch.from_numpy(np.array([float(self.lons[idx])]).astype(np.float32)).clone()
        tem = torch.from_numpy(np.load(self.tems[idx]).astype(np.float32)).clone()
        sal = torch.from_numpy(np.load(self.sals[idx]).astype(np.float32)).clone()
        ssh = torch.from_numpy(np.load(self.sshs[idx], allow_pickle=True).data.astype(np.float32)).clone()
        sst = torch.from_numpy(np.load(self.ssts[idx], allow_pickle=True).data.astype(np.float32)).clone()

        # Concatenate input and output
        ##### Need to add BIO
        in_map = torch.stack((ssh, sst, sst), dim=0).unsqueeze(0)

        # Resize ssh and sst
        in_map = F.interpolate(in_map, size=(self.resize, self.resize), mode=self.resize_method, align_corners=False).squeeze(0)

        if self.objective == 'salinity':
            return lat, lon, in_map, sal
        else:
            return lat, lon, in_map, tem
        

if __name__ == '__main__':
    pass
