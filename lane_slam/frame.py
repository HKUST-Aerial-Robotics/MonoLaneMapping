#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

from misc.curve.bspline_approx import CubicBSplineApproximator
from misc.pcd_utils import split_lane_by_id
from misc.plot_utils import visualize_points_list
from lane_slam.lane_feature import LaneFeature
from misc.config import cfg

class Frame:
    def __init__(self, frame_id, lanes, T_wc, timestamp):
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.T_wc = T_wc
        self.T_cw = self.inv_se3(self.T_wc)
        self.lane_features = self.load_lane_features(lanes)

    def inv_se3(self, T):
        T_inv = np.eye(4)
        T_inv[:3, :3] = T[:3, :3].T
        T_inv[:3, 3] = -np.dot(T[:3, :3].T, T[:3, 3])
        return T_inv

    def load_lane_features(self, lanes):
        lane_features = []
        for lane in lanes:
            instance_id = lane['track_id']
            lane_points_fit = lane['xyz']
            lane_features.append(LaneFeature(-1, lane_points_fit, lane['category']))
        return lane_features

    def get_lane_features(self):
        return self.lane_features

    def transform_to_world(self, lane_feature:LaneFeature):
        lane_pts_c = lane_feature.get_xyzs()
        lane_pts_w = lane_pts_c[:,:3] @ self.T_wc[:3, :3].T + self.T_wc[:3, 3].reshape(1, 3)
        lane_pts_w = np.concatenate([lane_pts_w, lane_pts_c[:, 3:]], axis=1)
        return LaneFeature(lane_feature.id, lane_pts_w, lane_feature.category)

    def lane_fitting(self, lane_points, step=0.1):
        #   lanes: [N, 6]
        approximator = CubicBSplineApproximator(max_iter = 20, res_delta_tld = 5e-2)
        bspline3 = approximator.approximate(lane_points, method="iterative")
        fitted_lane = bspline3.get_points_final(100)
        cumsum = np.cumsum(np.sqrt(np.sum(np.diff(fitted_lane, axis=0)**2, axis=1)))[-1]
        num = int(cumsum / step)
        fitted_lane = bspline3.get_points_final(num)

        return fitted_lane