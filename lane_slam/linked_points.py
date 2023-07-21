#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk
import numpy as np
from scipy.spatial import KDTree
from misc.config import cfg
from misc.curve.catmull_rom import parameterization
from time import perf_counter

class Node(object):
    def __init__(self, item: np.ndarray, id: int):
        # item存放数据元素
        self.item = item
        self.id = id
        self.gtsam_key = None
        self.lane_id = None
    
    def set_key(self, key):
        assert self.gtsam_key is None, "key has been set"
        self.gtsam_key = key

    def set_lane_id(self, lane_id):
        self.lane_id = lane_id

    def get_lane_id(self):
        return self.lane_id

    def get_key(self):
        return self.gtsam_key

    def set_item(self, item):
        self.item = item

class LinkedPoints(object):

    def __init__(self):
        self.kdtree = None
        self.max_node_id = -1
        self.items = []
        self.head_height = None
        self.tail_height = None
        self.alpha = cfg.lane_mapping.z_filter_alpha

    def get_nearest_ctrlpts4(self, point):
        # find 4 nearest control points
        point = point[np.newaxis, :3]
        dist, idx = self.kdtree.query(point, k=2)
        near_enough = dist[0][0] < cfg.lane_mapping.ctrl_points_chord \
                        and dist[0][1] < cfg.lane_mapping.ctrl_points_chord
        sec_idx = min(idx[0])
        third_idx = max(idx[0])
        second_ctrlpts = self.items[sec_idx]
        third_ctrlpts = self.items[third_idx]
        is_first = sec_idx == 0
        is_last = third_idx == (len(self.items) - 1)
        if near_enough and not is_first and not is_last:
            first_ctrlpts = self.items[sec_idx - 1]
            fourth_ctrlpts = self.items[third_idx + 1]
            ctrlpts = [first_ctrlpts, second_ctrlpts, third_ctrlpts, fourth_ctrlpts]
            return ctrlpts
        else:
            return None
    def find_footpoint(self, point):
        point = point[np.newaxis, :3]
        dist, idx = self.kdtree.query(point, k=2)
        if (dist > cfg.lane_mapping.ctrl_points_chord).any():
            return None, None, None
        if (idx == 0).any() or (idx == (self.size() - 1)).any():
            return None, None, None
        if abs(idx[0][0] - idx[0][1]) > 1:
            return None, None, None
        idx1 = min(idx[0])
        idx2 = max(idx[0])
        ctrl_pts = [self.get_node(idx1-1), self.get_node(idx1),
                   self.get_node(idx2), self.get_node(idx2+1)]
        ctrl_pts_np = np.array([node.item for node in ctrl_pts])
        u, error = parameterization(point, ctrl_pts_np)
        if u is None:
            return None, None, None
        return ctrl_pts, u, error

    def get_nodes(self):
        return self.items
    def get_node(self, idx):
        return self.items[idx]
    def get_xyzs(self):
        return [node.item for node in self.items]
    def get_xyz(self, idx):
        return self.items[idx].item

    def reverse(self):
        self.items.reverse()
        tmp = self.head_height
        self.head_height = self.tail_height
        self.tail_height = tmp

    def add(self, item):
        self.max_node_id += 1
        if self.alpha > 0:
            if self.head_height is None:
                self.head_height = item[2]
            else:
                self.head_height = self.alpha * self.head_height + (1 - self.alpha) * item[2]
            item[2] = self.head_height
        node = Node(item, self.max_node_id)
        self.items.insert(0, node)

    def append(self, item):
        self.max_node_id += 1
        if self.alpha > 0:
            if self.tail_height is None:
                self.tail_height = item[2]
            else:
                self.tail_height = self.alpha * self.tail_height + (1 - self.alpha) * item[2]
            item[2] = self.tail_height
        node = Node(item, self.max_node_id)
        self.items.append(node)

    def insert(self, index, item):
        self.max_node_id += 1
        node = Node(item, self.max_node_id)
        self.items.insert(index, node)

    def size(self):
       return len(self.items)

    def update_kdtree(self):
        points = self.get_xyzs()
        points = np.array(points)
        # ordered
        self.kdtree = KDTree(points)