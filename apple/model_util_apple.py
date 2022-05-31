# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))


class AppleDatasetConfig(object):
    def __init__(self):
        self.num_class = 17
        # for heading bins you can probably use the same as what votenet used on SUNRGBD
        self.num_heading_bin = 12
        self.num_size_cluster = 17

        self.type2class = {'cabinet': 0, 'refrigerator': 1, 'shelf': 2, 'stove': 3, 'bed': 4, 'sink': 5, 'washer': 6,
                           'toilet': 7, 'bathtub': 8, 'oven': 9, 'dishwasher': 10, 'fireplace': 11, 'stool': 12,
                           'chair': 13, 'table': 14, 'tv_monitor': 15, 'sofa': 16}
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.type2onehotclass = {'cabinet': 0, 'refrigerator': 1, 'shelf': 2, 'stove': 3, 'bed': 4, 'sink': 5,
                                 'washer': 6, 'toilet': 7, 'bathtub': 8, 'oven': 9, 'dishwasher': 10, 'fireplace': 11,
                                 'stool': 12, 'chair': 13, 'table': 14, 'tv_monitor': 15, 'sofa': 16}
        self.type_mean_size = {'cabinet': np.array([0.50535039, 0.99948107, 1.01409264]),
                               'refrigerator': np.array([0.68699782, 0.78236207, 1.69539735]),
                               'shelf': np.array([0.36185262, 0.96579244, 1.35701753]),
                               'stove': np.array([0.57206923, 0.74432994, 0.15969855]),
                               'bed': np.array([2.13424668, 1.5492419,  0.6371238]),
                               'sink': np.array([0.4487786,  0.49777492, 0.18768632]),
                               'washer': np.array([0.66299318, 0.61857649, 0.8749568]),
                               'toilet': np.array([0.62535566, 0.41170164, 0.72598087]),
                               'bathtub': np.array([0.78805841, 1.68880503, 0.58119007]),
                               'oven': np.array([0.61922381, 0.63303137, 0.71439738]),
                               'dishwasher': np.array([0.64387381, 0.58438052, 0.89909207]),
                               'fireplace': np.array([0.66883935, 1.48875462, 1.18658762]),
                               'stool': np.array([0.4108859,  0.61343751, 0.48144416]),
                               'chair': np.array([0.54869829, 0.49846076, 0.90221954]),
                               'table': np.array([0.59851039, 0.96247873, 0.61906196]),
                               'tv_monitor': np.array([0.05492941, 0.90722642, 0.5792501]),
                               'sofa': np.array([1.00251324, 1.60251208, 0.81406632])}

        self.mean_size_arr = np.zeros((self.num_size_cluster, 3))
        for i in range(self.num_size_cluster):
            self.mean_size_arr[i, :] = self.type_mean_size[self.class2type[i]]

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual

    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        mean_size = self.type_mean_size[self.class2type[pred_cls]]
        return mean_size + residual

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from  
            class center angle to current angle.

            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle
        '''
        num_class = self.num_heading_bin
        angle = angle % (2 * np.pi)
        assert (angle >= 0 and angle <= 2 * np.pi)
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - \
            (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class '''
        num_class = self.num_heading_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
        return angle

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle * -1
        return obb
