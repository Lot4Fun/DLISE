#!/usr/bin/python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from logging import getLogger
import cv2
import torch

logger = getLogger('DLISE')

class Detector(object):

    def __init__(self, model, device, config, save_dir):
        
        self.model = model
        self.device = device
        self.config = config
        self.save_dir = save_dir.joinpath('detected')

        self.save_dir.mkdir(exist_ok=True, parents=True)


    def run(self, data_loader):

        logger.info('Begin detection.')
        self.model.eval()
        with torch.no_grad():

            detected_list = [] if self.config.detect.save_results else None

            n_detected = 0
            for img_path, img_h, img_w, img in data_loader:

                # Convert tuple of length 1 to string
                img_path = img_path[0]

                if self.device.type == 'cuda':
                    img = img.to(self.device)

                detected = self.model(img).to('cpu')
                scale_factors = torch.Tensor([img_w, img_h, img_w, img_h])

                if self.config.detect.save_results:
                    detected_list.append(detected.tolist())

                if self.config.detect.visualize:
                    self._visualize(img_path, detected, scale_factors)
                
                n_detected += 1
                if not (n_detected % 100):
                    logger.info(f'Progress: [{n_detected:08}/{len(data_loader.dataset):08}]')

        if self.config.detect.save_results:
            with open(str(self.save_dir.parent.joinpath('detected.json')), 'w') as f:
                json.dump(detected_list, f, ensure_ascii=False, indent=4)

        logger.info('Detection has finished.')


    def _visualize(self, img_path, detected, scale_factors):

        # Set information
        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape
        conf_th = self.config.detect.conf_threshold
        labels = self.config.model.classes
        colors = self.config.detect.box_bgr_colors
        save_path = self.save_dir.joinpath(Path(img_path).name)

        # Set thicknesses
        thickness_ratio = 2. / 640 # Empirically decided
        line_thickness = max(int(max(img_h, img_w) * thickness_ratio), 1)
        char_thickness = max(line_thickness - 1, 1)

        for i in range(detected.size(1)):
            """
            detected[0, i, j, k]
              - 0 : Each 'detected' has only one batch
              - i : Class index
              - j : Index of top 200 boxes
              - k : conf, xmin, ymin, xmax or ymax
                    Format of 'detected[0, i, j]' is [conf, xmin, ymin, xmax, ymax]
            """
            j = 0
            while detected[0, i, j, 0] >= conf_th:
                score = detected[0, i, j, 0]
                label_name = labels[i-1]
                color = colors[i-1]
                display_txt = '%s: %.2f' % (label_name, score)
                pt = (detected[0, i, j, 1:] * scale_factors).cpu().numpy()
                cv2.rectangle(img, (int(pt[0]), int(pt[1])), (int(pt[2]+1), int(pt[3]+1)), color, line_thickness)
                cv2.putText(img, display_txt, (int(pt[0]), int(pt[1])-line_thickness*3), 0, 1e-3 * img_h, color, char_thickness)
                j += 1

        cv2.imwrite(str(save_path), img)
