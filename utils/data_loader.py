#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from pathlib import Path
import re
import cv2
import numpy as np

import torch
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

        train_img_list, train_anno_list, val_img_list, val_anno_list = make_filepath_list(config.train.input_dir)

        # Dataset
        train_dataset = VOCDataset(train_img_list,
                                   train_anno_list,
                                   phase='train',
                                   transform=DataTransform(config.model.input_size, config.model.rgb_means),
                                   transform_anno=AnnoXML2List(config.model.classes))
        val_dataset = VOCDataset(val_img_list,
                                 val_anno_list,
                                 phase='val',
                                 transform=DataTransform(config.model.input_size, config.model.rgb_means),
                                 transform_anno=AnnoXML2List(config.model.classes))

        # Data loader
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=config.train.batch_size,
                                       shuffle=config.train.shuffle,
                                       collate_fn=detection_collate)
        val_loader = data.DataLoader(val_dataset,
                                     batch_size=config.train.batch_size,
                                     shuffle=False,
                                     collate_fn=detection_collate)

        return train_loader, val_loader


    @classmethod
    def build_for_detect(self, config, x_dir):

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


class BatchDataset(torch.utils.data.Dataset):

    def __init__(self, inputs, config):

        self.inputs = inputs
        self.resize = config.model.input_size
        self.rgb_means = config.model.rgb_means

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, idx):

        x = cv2.imread(str(self.inputs[idx]))
        img_h, img_w, _ = x.shape

        x = cv2.resize(x, (self.resize, self.resize)).astype(np.float32)
        x = x[:, :, ::-1].copy() # Reorder from BGR to RGB
        x -= self.rgb_means

        # [H, W, C] -> [C, H, W]
        x = torch.from_numpy(x).permute(2, 0, 1)

        return str(self.inputs[idx]), img_h, img_w, x


if __name__ == '__main__':
    pass
