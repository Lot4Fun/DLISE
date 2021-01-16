#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import utils.augmentations as aug

class DataTransform():

    def __init__(self, input_size=300, rgb_means=(104, 117, 123)):

        self.transform = {
            'train': aug.Compose([
                aug.ConvertFromInts(),
                aug.ToAbsoluteCoords(),
                aug.PhotometricDistort(),
                aug.Expand(rgb_means),
                aug.RandomSampleCrop(),
                aug.RandomMirror(),
                aug.ToPercentCoords(),
                aug.Resize(input_size),
                aug.SubtractMeans(rgb_means)
            ]),
            'val': aug.Compose([
                aug.ConvertFromInts(),
                aug.Resize(input_size),
                aug.SubtractMeans(rgb_means)
            ])
        }
    
    def __call__(self, img, phase, boxes, labels):

        return self.transform[phase](img, boxes, labels)


if __name__ == '__main__':
    pass
