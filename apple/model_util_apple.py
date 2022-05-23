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
        self.num_heading_bin = 12 # for heading bins you can probably use the same as what votenet used on SUNRGBD
        self.num_size_cluster = 17 # for num_size_cluster you can use the number of semantic classes, same as on scannet

        self.type2class={'cabinet':0, 'refrigerator':1, 'shelf':2, 'stove':3, 'bed':4, 'sink':5, 'washer':6, 'toilet':7, 'bathtub':8, 'oven':9, 'dishwasher':10, 'fireplace':11, 'stool':12, 'chair':13, 'table':14, 'tv_monitor':15, 'sofa':16}
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        self.type2onehotclass={'cabinet':0, 'refrigerator':1, 'shelf':2, 'stove':3, 'bed':4, 'sink':5, 'washer':6, 'toilet':7, 'bathtub':8, 'oven':9, 'dishwasher':10, 'fireplace':11, 'stool':12, 'chair':13, 'table':14, 'tv_monitor':15, 'sofa':16}
        self.type_mean_size = {'cabinet': np.array([0.49929752, 0.98864983, 1.06902208]),
                          'refrigerator': np.array([0.69317901, 0.83925985, 1.78362792]),
                          'shelf': np.array([0.34451816, 1.06582312, 1.28442023]),
                          'stove': np.array([0.57476898, 0.78205954, 0.0930413 ]),
                          'bed': np.array([2.11906018, 1.62292264, 0.61575489]),
                          'sink': np.array([0.43687792, 0.52087483, 0.19492858]),
                          'washer': np.array([0.65188205, 0.6185208 , 0.87093902]),
                          'toilet': np.array([0.63613452, 0.42283165, 0.74882042]),
                          'bathtub': np.array([0.80622877, 1.76614899, 0.57439523]),
                          'oven': np.array([0.6073346 , 0.60937532, 0.75432479]),
                          'dishwasher': np.array([0.67502088, 0.62215706, 0.91262674]),
                          'fireplace': np.array([0.63435895, 1.56922212, 1.26271292]),
                          'stool': np.array([0.41311146, 0.63806829, 0.47056733]),
                          'chair': np.array([0.53437956, 0.47606608, 0.90516595]),
                          'table': np.array([0.5678956 , 0.949926  , 0.61854677]),
                          'tv_monitor': np.array([0.05225567, 0.90698287, 0.58970672]),
                          'sofa': np.array([0.98468159, 1.55693327, 0.81914572])}

        self.mean_size_arr = np.zeros((self.num_size_cluster, 3))
        for i in range(self.num_size_cluster):
            self.mean_size_arr[i,:] = self.type_mean_size[self.class2type[i]]

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
        angle = angle%(2*np.pi)
        assert(angle>=0 and angle<=2*np.pi)
        angle_per_class = 2*np.pi/float(num_class)
        shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
        class_id = int(shifted_angle/angle_per_class)
        residual_angle = shifted_angle - (class_id*angle_per_class+angle_per_class/2)
        return class_id, residual_angle
    
    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class '''
        num_class = self.num_heading_bin
        angle_per_class = 2*np.pi/float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle>np.pi:
            angle = angle - 2*np.pi
        return angle

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle*-1
        return obb


