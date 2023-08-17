#!/usr/bin/env python
#!/bin/sh
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk
from time import perf_counter
from copy import deepcopy
import tqdm
from lane_slam.system.lane_opt import LaneOptimizer
from lane_slam.lane_utils import drop_lane_by_p
from misc.config import cfg
import numpy as np

class LaneMapping(LaneOptimizer):
    def __init__(self, bag_file):
        super(LaneMapping, self).__init__(bag_file)
        self.load_data()
        self.tracking_init()
        self.graph_init()
        self.debug_init()

    def process(self):
        for frame_id, frame_data in enumerate(tqdm.tqdm(self.frames_data, leave=False, dynamic_ncols=True)):
            pose_wc = frame_data['gt_pose']
            timestamp = frame_data['timestamp']
            lane_pts_c = drop_lane_by_p(deepcopy(frame_data['lanes_predict']), p=cfg.preprocess.drop_prob)
            self.save_pred_to_json(lane_pts_c, timestamp)
            self.time_stamp.append(timestamp)

            # 1. odometry
            t0 = perf_counter()
            self.odometry(lane_pts_c, pose_wc, timestamp)
            self.odo_timer.update(perf_counter() - t0)

            # 2. lane association
            t1 = perf_counter()
            self.lane_association()
            self.assoc_timer.update(perf_counter() - t1)

            # 3. update lanes in map
            self.map_update()
            self.prev_frame = self.cur_frame
            self.whole_timer.update(perf_counter() - t0)

            self.save_lanes_to_json(self.cur_frame)
            # self.save_for_visualization(self.cur_frame)
            self.lane_nms(self.cur_frame)

            if self.merge_lane:
                self.post_merge_lane()

        # print('whole time: %.3fms, odo: %.3fms, assoc: %.3fms, graph: %.3fms, isam: %.3fms' % (
        #     self.whole_timer.avg * 1000, self.odo_timer.avg * 1000, self.assoc_timer.avg * 1000,
        #     self.graph_build_timer.avg * 1000, self.opt_timer.avg * 1000))
        # print('Max whole time: %.3fms, odo: %.3fms, assoc: %.3fms, graph: %.3fms, isam: %.3fms' % (
        #     self.whole_timer.max * 1000, self.odo_timer.max * 1000, self.assoc_timer.max * 1000,
        #     self.graph_build_timer.max * 1000, self.opt_timer.max * 1000))
        stats = self.evaluate_pose()
        stats.update({"map_size": self.map_size()})
        stats.update({"graph": self.graph_build_timer.avg * 1000})
        stats.update({"isam": self.opt_timer.avg * 1000})
        if self.eval_pose_only:
            return stats
        self.post_merge_lane()
        self.save_map()
        if self.visualization:
            # stats.update(self.eval_single_segment())
            self.visualize_map()

        return stats

    def lane_nms(self, frame):
        for lane in frame.get_lane_features():
            if lane.id == -1:
                continue
            landmark = self.lanes_in_map[lane.id]
            landmark.add_obs_frame_id(frame.frame_id)

        min_obs_num = 4
        ids_to_remove = []
        for lm in self.lanes_in_map.values():
            if frame.frame_id - lm.obs_first_frame_id < min_obs_num+2:
                continue
            if lm.obs_num < min_obs_num:
                ids_to_remove.append(lm.id)
        for id in ids_to_remove:
            self.lanes_in_map.pop(id)

    def post_merge_lane(self):

        overlap_id = []
        for lm in self.lanes_in_map.values():
            score = lm.obs_num / (lm.obs_last_frame_id - lm.obs_first_frame_id + 1)
            # print(lm.id, lm.obs_num, score)
            if score < 0.5 and lm.obs_num > 2:
                overlap_id.append(lm.id)

        for lane_id_a, lane_feature_a in self.lanes_in_map.items():
            for lane_id_b, lane_feature_b in self.lanes_in_map.items():
                if lane_id_a == lane_id_b:
                    continue
                overlap_a = lane_feature_a.overlap_ratio(lane_feature_b)
                overlap_b = lane_feature_b.overlap_ratio(lane_feature_a)
                if overlap_a > 0.7 and lane_feature_a.size() < lane_feature_b.size():
                    overlap_id.append(lane_id_a)
                if overlap_b > 0.7 and lane_feature_b.size() < lane_feature_a.size():
                    overlap_id.append(lane_id_b)

        overlap_id = list(set(overlap_id))
        for lane_id in overlap_id:
            self.lanes_in_map.pop(lane_id)
