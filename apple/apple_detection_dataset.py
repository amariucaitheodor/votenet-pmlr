# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Dataset for 3D object detection on Apple data (with support of vote supervision).
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from model_util_apple import AppleDatasetConfig
from apple_guys_utils import box_utils, rotation
import pc_util
import sunrgbd_utils
from box_util import get_3d_box
import numpy as np
from torch.utils.data import Dataset



DC = AppleDatasetConfig()  # dataset specific config
MAX_NUM_OBJ = 64  # maximum number of objects allowed per scene


class AppleDetectionVotesDataset(Dataset):
    def __init__(self, split_set='train', num_points=20000,
                 use_height=False, scan_idx_list=None):

        # assert(num_points <= 50000)
        self.data_path = os.path.join(
            ROOT_DIR, f'apple/{"Training" if split_set=="train" else "Validation"}/')

        self.scan_names = sorted(list(
            set([os.path.basename(x)[0:8] for x in os.listdir(self.data_path + "data/")])))
        if scan_idx_list is not None:
            self.scan_names = [self.scan_names[i] for i in scan_idx_list]
        self.num_points = num_points
        self.use_height = use_height

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            heading_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            heading_residual_label: (MAX_NUM_OBJ,)
            size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            vote_label: (N,9) with votes XYZ (3 votes: X1Y1Z1, X2Y2Z2, X3Y3Z3)
                if there is only one vote than X1==X2==X3 etc.
            vote_label_mask: (N,) with 0/1 with 1 indicating the point
                is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            max_gt_bboxes: unused
        """
        scan_name = self.scan_names[idx]
        point_cloud = np.load(os.path.join(self.data_path + "/data/", scan_name)+'_pc.npy') #['pc']  # Nx3, NOT Nx6 like sunrgbd
        label = np.load(os.path.join(self.data_path + "/label/", scan_name)+'_bbox.npy', allow_pickle=True).item()

        bboxes = label["bboxes"] # Kx7
        bboxes[:, 3:6] += 0.01 # make boxes a little bit larger to include more points
        gt_boxes = box_utils.boxes_to_corners_3d(bboxes) # K,8
        point_votes = get_votes(point_cloud, gt_boxes) # Nx4

        uids = label["types"] # Kx1
        sem_labels = np.array([DC.type2class[uid] for uid in uids]).reshape((-1, 1)) # Kx1

        bboxes = np.c_[bboxes, sem_labels]

        assert(len(point_cloud[:, 2]) > 0, f"point_cloud[:, 2] had shape {point_cloud[:, 2].shape} for scene {scan_name} - please consider removing it")

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate(
                [point_cloud, np.expand_dims(height, 1)], 1)  # (N,4) or (N,7)

        # ------------------------------- LABELS ------------------------------
        box3d_centers = np.zeros((MAX_NUM_OBJ, 3))
        box3d_sizes = np.zeros((MAX_NUM_OBJ, 3))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        label_mask = np.zeros((MAX_NUM_OBJ))
        label_mask[0:bboxes.shape[0]] = 1
        max_bboxes = np.zeros((MAX_NUM_OBJ, 8))
        max_bboxes[0:bboxes.shape[0], :] = bboxes

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            semantic_class = bbox[7]
            box3d_center = bbox[0:3]
            angle_class, angle_residual = DC.angle2class(bbox[6])
            # NOTE: The mean size stored in size2class is of full length of box edges,
            # while in sunrgbd_data.py data dumping we dumped *half* length l,w,h.. so have to time it by 2 here
            box3d_size = bbox[3:6] # * 2
            size_class, size_residual = DC.size2class(
                box3d_size, DC.class2type[semantic_class])
            box3d_centers[i, :] = box3d_center
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            size_classes[i] = size_class
            size_residuals[i] = size_residual
            box3d_sizes[i, :] = box3d_size

        target_bboxes_mask = label_mask
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            R = rotation.eulerAnglesToRotationMatrix([0, -1 * bbox[6], 0])
            corners_3d = box_utils.compute_box_3d(bbox[3:6], bbox[0:3], R)

            # compute axis aligned box
            xmin = np.min(corners_3d[:, 0])
            ymin = np.min(corners_3d[:, 1])
            zmin = np.min(corners_3d[:, 2])
            xmax = np.max(corners_3d[:, 0])
            ymax = np.max(corners_3d[:, 1])
            zmax = np.max(corners_3d[:, 2])
            target_bbox = np.array(
                [(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, xmax-xmin, ymax-ymin, zmax-zmin])
            target_bboxes[i, :] = target_bbox

        point_cloud, choices = pc_util.random_sampling(
            point_cloud, self.num_points, return_choices=True)

        # choices = pc_util.down_sample(point_cloud, voxel_sz=0.06)
        # point_cloud = point_cloud[choices, :]
        point_votes_mask = point_votes[choices, 0]
        point_votes = point_votes[choices, 1:]

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:, 0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0:bboxes.shape[0]] = bboxes[:, -1]  # from 0 to 9
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['max_gt_bboxes'] = max_bboxes
        return ret_dict


def viz_votes(pc, point_votes, point_votes_mask):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_votes_mask == 1)
    pc_obj = pc[inds, 0:3]
    pc_obj_voted1 = pc_obj + point_votes[inds, 0:3]
    # pc_obj_voted2 = pc_obj + point_votes[inds, 3:6]
    # pc_obj_voted3 = pc_obj + point_votes[inds, 6:9]
    pc_util.write_ply(pc_obj, 'pc_obj.ply')
    pc_util.write_ply(pc_obj_voted1, 'pc_obj_voted1.ply')
    # pc_util.write_ply(pc_obj_voted2, 'pc_obj_voted2.ply')
    # pc_util.write_ply(pc_obj_voted3, 'pc_obj_voted3.ply')


def viz_obb(pc, label, mask, angle_classes, angle_residuals,
            size_classes, size_residuals):
    """ Visualize oriented bounding box ground truth
    pc: (N,3)
    label: (K,3)  K == MAX_NUM_OBJ
    mask: (K,)
    angle_classes: (K,)
    angle_residuals: (K,)
    size_classes: (K,)
    size_residuals: (K,3)
    """
    oriented_boxes = []
    K = label.shape[0]
    for i in range(K):
        if mask[i] == 0:
            continue
        obb = np.zeros(7)
        obb[0:3] = label[i, 0:3]
        heading_angle = DC.class2angle(angle_classes[i], angle_residuals[i])
        box_size = DC.class2size(size_classes[i], size_residuals[i])
        obb[3:6] = box_size
        obb[6] = -1 * heading_angle
        print(obb)
        oriented_boxes.append(obb)
    pc_util.write_oriented_bbox(oriented_boxes, 'gt_obbs.ply')
    pc_util.write_ply(label[mask == 1, :], 'gt_centroids.ply')


def get_sem_cls_statistics():
    """ Compute number of objects for each semantic class """
    d = AppleDetectionVotesDataset(use_height=True)
    sem_cls_cnt = {}
    for i in range(len(d)):
        if i % 10 == 0:
            print(i)
        sample = d[i]
        pc = sample['point_clouds']
        sem_cls = sample['sem_cls_label']
        mask = sample['box_label_mask']
        for j in sem_cls:
            if mask[j] == 0:
                continue
            if sem_cls[j] not in sem_cls_cnt:
                sem_cls_cnt[sem_cls[j]] = 0
            sem_cls_cnt[sem_cls[j]] += 1
    print(sem_cls_cnt)


def get_votes(points, gt_boxes):
    """
    Args:
        points: (N, 3)
        boxes: (m, 8, 3)
    Returns:
        votes: (N, 4)
    """
    n_point = points.shape[0]
    point_votes = np.zeros((n_point, 4)).astype(np.float32)
    for obj_id in range(gt_boxes.shape[0]):
        tmp_box3d = np.expand_dims(gt_boxes[obj_id], 0)  # (8, 3)
        # (n_point, 1)
        mask_pts = box_utils.points_in_boxes(points[:, :3], tmp_box3d)
        mask_pts = mask_pts.reshape((-1,))
        point_votes[mask_pts, 0] = 1.0
        obj_center = np.mean(tmp_box3d, axis=1)  # (1, 3)

        # get votes
        pc_roi = points[mask_pts, :3]
        tmp_votes = obj_center - pc_roi
        point_votes[mask_pts, 1:4] = tmp_votes
    return point_votes

if __name__ == '__main__':
    d = AppleDetectionVotesDataset(use_height=True)
    sample = d[200]
    print(sample['vote_label'].shape, sample['vote_label_mask'].shape)
    pc_util.write_ply(sample['point_clouds'], 'pc.ply')
    viz_votes(sample['point_clouds'], sample['vote_label'],
              sample['vote_label_mask'])
    viz_obb(sample['point_clouds'], sample['center_label'], sample['box_label_mask'],
            sample['heading_class_label'], sample['heading_residual_label'],
            sample['size_class_label'], sample['size_residual_label'])
