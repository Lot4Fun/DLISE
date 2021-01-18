#!/usr/bin/python3
# -*- coding: utf-8 -*-

from torch import optim

class Optimizers(object):

    @classmethod
    def get_optimizer(self, config, params):

        if config.optim_type == 'sgd':
            return optim.SGD(params=params, **(config.sgd))
        else:
            return optim.Adam(params=params, **(config.adam))


if __name__ == '__main__':
    pass
