#!/usr/bin/python3
# -*- coding: utf-8 -*-

from torch import optim

class Optimizers(object):

    @classmethod
    def get_optimizer(self, config, params):

        return optim.SGD(params=params,
                         lr=config.lr,
                         momentum=config.momentum,
                         weight_decay=config.weight_decay)


if __name__ == '__main__':
    pass
