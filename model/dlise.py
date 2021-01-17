#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import models

logger = getLogger('DLISE')

class DLISE(nn.Module):

    def __init__(self, exec_type, config):

        super(DLISE, self).__init__()

        self.exec_type = exec_type
        self.config = config

        self.backbone = nn.Sequential(*list(models.resnet50(pretrained=self.config.model.backbone_pretrained).children())[:-1])

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=512, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=100, bias=True)
        )

    def forward(self, lats, lons, maps):

        x = self.backbone(maps)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        return x

    def init_weights(self):

        self.backbone.apply(self.init_conv_layer)
        logger.info('Intialized backbone')

    def init_conv_layer(self, layer):

        if isinstance(layer, nn.Conv2d):
            self.xavier(layer.weight.data)
            #layer.bias.data.zero_()


    def xavier(self, param):

        init.xavier_uniform_(param)
        

    def load_weights(self, trained_weights):

        state_dict = torch.load(trained_weights, map_location=lambda storage, loc: storage)
        try:
            # Load weights trained by single GPU into single GPU
            self.load_state_dict(state_dict) 
        except:
            # Load weights trained by multi GPU into single GPU
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                if 'module' in k:
                    k = k.replace('module.', '')
                new_state_dict[k] = v

            self.load_state_dict(new_state_dict)        


if __name__ == '__main__':
    pass
