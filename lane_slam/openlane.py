#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk

import os
import cv2
import numpy as np
import json
import glob
from misc.config import cfg


class OpenLane:

    def __init__(self, args, segment, dataset = None):
        self.dataset = dataset
        self.data_dir = cfg.dataset.label_dir
        self.dataset_dir = cfg.dataset.image_dir
        self.segment = segment
        self.timestamp_micros_list = self.load_data()

    def load_data(self):
        segment_dir = os.path.join(self.data_dir, 'training', self.segment)
        json_files = sorted(glob.glob(os.path.join(segment_dir, '*.json')))
        timestamp_micros_list = [int(os.path.basename(json_file).replace('00.json', '.json').split('.')[0]) for json_file in json_files]
        return timestamp_micros_list

    def __len__(self):
        return len(self.timestamp_micros_list)

    def fetch_gt_data(self, timestamp_micros, return_image = False):
        gt_json = os.path.join(self.data_dir, 'training', self.segment, '{:<018}.json'.format(timestamp_micros))
        with open(gt_json, 'r') as fp:
            gt_dict = json.load(fp)

        if return_image:
            image_path = os.path.join(self.dataset_dir, gt_dict['file_path'])
            img = cv2.imread(image_path)
        else:
            img = None

        vehicle_pose = np.array(gt_dict['pose'])
        ex0 = np.array(gt_dict['extrinsic'])
        cam0_pose = vehicle_pose @ ex0

        lane_all = []
        for lane in gt_dict['lane_lines']:
            xyz = np.asarray(lane['xyz']).reshape(3, -1).T
            if xyz.shape[0] == 0:
                continue
            category = np.asarray(lane['category']).reshape(-1, 1).repeat(xyz.shape[0], axis=0)
            visibility = np.asarray(lane['visibility']).reshape(-1, 1)
            track_id = np.asarray(lane['track_id']).reshape(-1, 1).repeat(xyz.shape[0], axis=0)
            points = np.concatenate([xyz, category, visibility, track_id], axis=1)
            lane_all.append(points)

        if len(lane_all) > 0:
            lane_points = np.vstack(lane_all)
        else:
            lane_points = None

        return img, lane_points, cam0_pose

    def fetch_data(self, timestamp):
        idx_json_file = os.path.join(self.data_dir, 'training', self.segment, '{:<018}.json'.format(timestamp))
        idx = self.dataset._label_list.index(idx_json_file)

        idx_json_file, image, seg_label, gt_anchor, gt_laneline_img, idx, gt_cam_height, \
        gt_cam_pitch, intrinsics, extrinsics, aug_mat, seg_name, seg_bev_map = self.dataset.WIP__getitem__(idx)

        data_dict = {
            'idx_json_file': idx_json_file,
            'image': image.unsqueeze(0),
            'seg_label': seg_label.unsqueeze(0),
            'gt_anchor': gt_anchor.unsqueeze(0),
            'gt_laneline_img': gt_laneline_img.unsqueeze(0),
            'idx': idx,
            'gt_cam_height': gt_cam_height.unsqueeze(0),
            'gt_cam_pitch': gt_cam_pitch.unsqueeze(0),
            'intrinsics': intrinsics.unsqueeze(0),
            'extrinsics': extrinsics.unsqueeze(0),
            'aug_mat': aug_mat.unsqueeze(0),
            'seg_name': seg_name,
            'seg_bev_map': seg_bev_map.unsqueeze(0)
        }

        return data_dict