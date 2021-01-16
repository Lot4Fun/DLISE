#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Import modules
import datetime
import json
from pathlib import Path
import torch

class CommonUtils(object):

    @classmethod
    def issue_id(self):
        return datetime.datetime.now().strftime('%Y-%m%d-%H%M-%S%f')[:-4]

    @classmethod
    def prepare(self, config, save_dir):
        save_dir.mkdir(exist_ok=True, parents=True)
        with open(str(save_dir.joinpath('config.json')), 'w') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

    @classmethod
    def save_weight(self, model, save_path):
        save_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    pass
