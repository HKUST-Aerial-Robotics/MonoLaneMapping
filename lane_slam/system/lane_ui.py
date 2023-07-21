#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk

import open3d as o3d
import numpy as np
from misc.ros_utils.msg_utils import posemsg_to_np, lanemsg_to_list
from misc.config import cfg
from lane_slam.lane_utils import lane_denoise, prune_3d_lane_by_range, points_downsample
from misc.plot_utils import visualize_points_list
from misc.pcd_utils import transform_points, make_open3d_point_cloud
from misc.lie_utils import inv_se3
from lane_slam.assoc_utils import draw_correspondences, draw_lines
from misc.plot_utils import pointcloud_to_spheres, text_3d
from lane_slam.lane_feature import LaneFeature
import rosbag
from misc.curve.catmull_rom import CatmullRomSpline, CatmullRomSplineList
import tqdm
from misc.utils import AverageMeter, mkdir_if_missing
from lane_slam.persformer_utils import transform_points_from_cam_to_ground, transform_points_from_ground_to_camera
import os
import json
from lane_slam.frame import Frame
from evaluation.eval_3D_lane import eval_lane3d

class LaneUI:
    def __init__(self, bag_file):
        self.pp_ds = cfg.preprocess.downsample # downsample size in pre-processing
        self.bag = rosbag.Bag(bag_file)
        self.segment = bag_file.split('/')[-1].split('.')[0]
        self.eval_pose_only = cfg.eval_pose
        self.add_odo_noise = cfg.pose_update.add_odo_noise
        self.merge_lane = cfg.lane_mapping.merge_lane

    def debug_init(self):
        # debug
        self.ctrl_pts_num_last = 0
        self.ctrl_pts_num = 0
        cfg.debug_flag = False
        self.visualization = cfg.visualization

        self.output_dir = os.path.join(cfg.output_dir, 'results', self.segment)
        self.det_output_dir = os.path.join(cfg.output_dir, 'results_det', self.segment)
        mkdir_if_missing(self.output_dir)
        mkdir_if_missing(self.det_output_dir)
        self.odo_timer = AverageMeter()
        self.assoc_timer = AverageMeter()
        self.graph_build_timer = AverageMeter()
        self.opt_timer = AverageMeter()
        self.whole_timer = AverageMeter()

    def visualize_association(self, lane_lm, lane_det, pose_ab, A=None, Agt=None):
        if len(lane_lm) == 0 or len(lane_det) == 0:
            return None
        lane_lm_points = {}
        lane_lm_xyz = []
        text_pcd_list = []
        for i, lane in enumerate(lane_lm):
            xyz = lane.points
            lane_lm_xyz.append(xyz)
            text_pos = xyz[xyz.shape[0]//2, :] + np.array([1.5, -1.5, 0])
            text_pcd = text_3d(str(i), text_pos, font_size=100, degree=-90.0)
            text_pcd_list.append(text_pcd)
            category = lane.category
            if category not in lane_lm_points:
                lane_lm_points[category] = []
            lane_lm_points[category].append(xyz)
        lm_points_list = []
        for key, value in lane_lm_points.items():
            lm_points_list.append(np.concatenate(value, axis=0))
        lane_det_points = {}
        lane_det_xyz = []
        for i, lane in enumerate(lane_det):
            xyz = lane.points
            category = lane.category
            xyz = np.dot(pose_ab, np.concatenate([xyz, np.ones([len(xyz), 1])], axis=1).T).T[:, :3]
            xyz = xyz + np.asarray([0, 0, 20]).reshape(1, 3)
            text_pos = xyz[xyz.shape[0]//2, :] + np.array([1.5, -1.5, 0])
            text_pcd = text_3d(str(i), text_pos, font_size=100, degree=-90.0)
            text_pcd_list.append(text_pcd)
            lane_det_xyz.append(xyz)
            if category not in lane_det_points:
                lane_det_points[category] = []
            lane_det_points[category].append(xyz)
        det_points_list = []
        for key, value in lane_det_points.items():
            det_points_list.append(np.concatenate(value, axis=0))

        # draw line
        correspondences = None
        if A is not None:
            correspondences = draw_correspondences(lane_lm_xyz, lane_det_xyz, A, Agt)

        lm_points_list.extend(det_points_list)
        text_pcd_list.extend(correspondences)
        visualize_points_list(lm_points_list, extra_pcd=text_pcd_list)
    def visualize_optimization(self, lanes_prev, lanes_opted, lanes_meas, pts_cp):
        extra_pcds = []
        lanes_prev_points = []
        for lane in lanes_prev:
            ctrl_pts = lane.ctrl_pts.get_xyzs()
            pcds = pointcloud_to_spheres(ctrl_pts, color=(1, 0, 0))
            lanes_prev_points.append(lane.points[:, :3])
            extra_pcds.append(pcds)
        lanes_prev_points = np.concatenate(lanes_prev_points, axis=0)

        lanes_opted_points = []
        for lane in lanes_opted:
            ctrl_pts = lane.ctrl_pts.get_xyzs()
            pcds = pointcloud_to_spheres(ctrl_pts, color=(0, 1, 0))
            lanes_opted_points.append(lane.points[:, :3])
            extra_pcds.append(pcds)
            for i in range(len(ctrl_pts)):
                text_pos = np.array(ctrl_pts[i]) + np.array([1.5, -1.5, -1])
                text_pcd = text_3d(str(i), text_pos, font_size=100, degree=-90.0)
                extra_pcds.append(text_pcd)

        lanes_opted_points = np.concatenate(lanes_opted_points, axis=0)
        lanes_meas_points = []
        for lane in lanes_meas:
            lanes_meas_points.append(lane.points[:, :3])
        lanes_meas_points = np.concatenate(lanes_meas_points, axis=0) if len(lanes_meas_points) > 0 else np.array([])

        pts_meas = []
        pts_model = []
        for pt_w, pt_c, ctrl_pts, lm_id, frame_id, noise, factor_id in pts_cp:
            pts_meas.append(pt_w[:3])
            catmull_spline = CatmullRomSpline(ctrl_pts)
            pts_model.append(catmull_spline.get_point(pt_w[3]))
        pts_meas = np.concatenate(pts_meas).reshape(-1, 3)
        pts_model = np.concatenate(pts_model).reshape(-1, 3)
        line_set = draw_lines(pts_meas, pts_model)
        pts_meas = pointcloud_to_spheres(pts_meas, color=[1, 0.706, 0], sphere_size=0.01)
        pts_model = pointcloud_to_spheres(pts_model, color=[0, 0.651, 0.929], sphere_size=0.01)
        extra_pcds.extend([line_set, pts_meas, pts_model])

        visualize_points_list([lanes_prev_points, lanes_opted_points, lanes_meas_points], extra_pcd=extra_pcds, axis_marker=1)
    def visualize_map(self, show_id=False):

        vis_pcds = []
        vis_points = []
        for lane_id, lane_feature in self.lanes_in_map.items():
            lane_pts = lane_feature.raw_points
            ctrl_pts = lane_feature.get_ctrl_xyz()

            pcd = make_open3d_point_cloud(lane_pts, color=[0.4, 0.4, 0.4])
            ctrl_pcds = pointcloud_to_spheres(ctrl_pts, sphere_size=0.4)
            curve = CatmullRomSplineList(np.array(ctrl_pts))
            fitted_pts = curve.get_points(30)
            vis_points.append(fitted_pts)

            lane_text = [pcd, ctrl_pcds]
            if show_id:
                for i in range(len(ctrl_pts)):
                    text_pos = ctrl_pts[i] + np.array([1.5, -1.5, -1])
                    text_pcd = text_3d(str(i), text_pos, font_size=100, degree=-90.0)
                    lane_text.append(text_pcd)

            # text_pos = np.mean(ctrl_pts, axis=0) + np.array([0.5, -0.5, 2])
            # text_pcd = text_3d(str(lane_id), text_pos, font_size=100, degree=-90.0)
            # lane_text.append(text_pcd)

            # visualize_points_list([fitted_pts], extra_pcd=lane_text, title='lane {}'.format(lane_id))
            vis_pcds.extend(lane_text)

        visualize_points_list(vis_points, extra_pcd=vis_pcds, title = self.segment)
    def visualize_mapping_and_gt(self, lanes_gt, lanes_pred, frame):
        lane_mapping = []
        for lane in frame.get_lane_features():
            if lane.id == -1:
                continue
            lane_lm = self.lanes_in_map[lane.id]
            xyz_w = lane_lm.points[:,:3].copy()
            xyz_c = transform_points(xyz_w, frame.T_cw)
            lane_mapping.append(xyz_c)
        lane_mapping = np.concatenate(lane_mapping, axis=0) if len(lane_mapping) > 0 else np.array([])
        lane_mapping = prune_3d_lane_by_range(lane_mapping, cfg.preprocess.range_area)

        lane_gt = []
        for lane in lanes_gt:
            lane_gt.append(lane['xyz'])
        lane_gt = np.concatenate(lane_gt, axis=0) if len(lane_gt) > 0 else np.array([])
        lane_gt = prune_3d_lane_by_range(lane_gt, cfg.preprocess.range_area)

        lane_pred = []
        for lane in lanes_pred:
            lane_pred.append(lane['xyz'])
        lane_pred = np.concatenate(lane_pred, axis=0) if len(lane_pred) > 0 else np.array([])
        lane_pred = prune_3d_lane_by_range(lane_pred, cfg.preprocess.range_area)

        visualize_points_list([lane_mapping, lane_gt, lane_pred], axis_marker=1)
    def load_data(self):
        self.eval_area = cfg.evaluation.eval_area
        frames_data = []
        gt_pose_msg = None
        lanes_gt_msg = None
        lanes_predict_msg = None
        for topic, msg, timestamp in self.bag.read_messages():
            if topic == '/gt_pose_wc':
                gt_pose_msg = msg
            elif topic == '/lanes_gt':
                lanes_gt_msg = msg
            elif topic == '/lanes_predict':
                lanes_predict_msg = msg
            else :
                raise ValueError('Unknown topic: {}'.format(topic))
            if gt_pose_msg is not None and lanes_gt_msg is not None and lanes_predict_msg is not None \
                    and gt_pose_msg.header.stamp == lanes_gt_msg.header.stamp \
                    and gt_pose_msg.header.stamp == lanes_predict_msg.header.stamp:
                gt_pose_wc = posemsg_to_np(gt_pose_msg)
                lanes_gt = lanemsg_to_list(lanes_gt_msg)
                lanes_predict = lanemsg_to_list(lanes_predict_msg)
                lanes_gt = self.preprocess_lanes(lanes_gt)
                lanes_predict = self.preprocess_lanes(lanes_predict)
                frames_data.append({
                    'timestamp': gt_pose_msg.header.stamp.to_sec(),
                    'gt_pose': gt_pose_wc,
                    'lanes_gt': lanes_gt,
                    'lanes_predict': lanes_predict
                })
                gt_pose_msg = None
                lanes_gt_msg = None
                lanes_predict_msg = None

        self.frames_data = frames_data
    def preprocess_lanes(self, lanes):
        for lane in lanes:
            xyz = lane['xyz']
            xyz = points_downsample(xyz, self.pp_ds)
            xyz = lane_denoise(xyz, smooth=True, interval=self.pp_ds)
            # xyz = lane_denoise(xyz)
            # rearrange points by distance to the origin
            xyz = xyz[np.argsort(np.linalg.norm(xyz, axis=1))]
            lane['xyz'] = xyz
            lane['visibility'] = np.ones(len(xyz))
        return lanes
    def get_lane_in_range(self, lanes):
        #  lane_points: [N, c]
        lanes_new = []
        for lane in lanes:
            xyz = lane['xyz']
            xyz = prune_3d_lane_by_range(xyz, cfg.preprocess.range_area)
            lane['xyz'] = xyz
            lane['visibility'] = [1] * len(lane['xyz'])
            if len(lane['xyz']) >= 4:
                lanes_new.append(lane)
        return lanes_new

    def save_pred_to_json(self, lane_pts_c, timestamp):
        if self.eval_pose_only:
            return
        timestamp_str = "{:<018}".format(int(timestamp*1e6))
        result = {'file_path': "validation/{}/{}.jpg".format(self.segment, timestamp_str), 'lane_lines': []}
        for lane_id, lane in enumerate(lane_pts_c):
            xyz = lane['xyz']
            extrinsic = self.get_extrinsic(timestamp)
            xyz = transform_points_from_cam_to_ground(xyz, extrinsic)
            result['lane_lines'].append({'xyz': xyz.tolist(), 'category': lane['category']})
        json_file = os.path.join(self.det_output_dir, '{}.json'.format(timestamp_str))
        with open(json_file, 'w') as result_file:
            json.dump(result, result_file)
        return True

    def get_extrinsic(self, timestamp):
        timestamp_str = "{:<018}".format(int(timestamp*1e6))
        annotation_dir = cfg.dataset.annotation_dir
        gt_json = os.path.join(annotation_dir, self.segment, '{}.json'.format(timestamp_str))
        with open(gt_json, 'r') as fp:
            gt_dict = json.load(fp)
        extrinsic = np.array(gt_dict['extrinsic'])
        return extrinsic

    def save_lanes_to_json(self, frame: Frame):
        if self.eval_pose_only:
            return
        timestamp_str = "{:<018}".format(int(frame.timestamp*1e6))
        result = {'file_path': "validation/{}/{}.jpg".format(self.segment, timestamp_str), 'lane_lines': []}
        saved_lanes = []
        for lane in frame.get_lane_features():
            if lane.id == -1:
                continue
            lane_lm = self.lanes_in_map[lane.id]
            xyz_w = lane_lm.points[:,:3].copy()
            xyz_c = transform_points(xyz_w, frame.T_cw)
            extrinsic = self.get_extrinsic(frame.timestamp)
            xyz = transform_points_from_cam_to_ground(xyz_c, extrinsic)
            xyz = prune_3d_lane_by_range(xyz, self.eval_area)
            if xyz.shape[0] < 2:
                continue
            saved_lanes.append(transform_points_from_ground_to_camera(xyz, extrinsic))
            result['lane_lines'].append({'xyz': xyz.tolist(), 'category': lane_lm.category})
        json_file = os.path.join(self.output_dir, '{}.json'.format(timestamp_str))
        with open(json_file, 'w') as result_file:
            json.dump(result, result_file)
        return saved_lanes

    def eval_single_segment(self):
        dataset_dir = cfg.dataset.annotation_dir + "/"
        pred_dir = os.path.abspath(os.path.join(cfg.output_dir, "results/")) + "/"
        output_dir = os.path.abspath(os.path.join(cfg.output_dir, "eval_results/segments/{}".format(self.segment))) + '/'
        mkdir_if_missing(output_dir)
        test_list = os.path.abspath(os.path.join(cfg.output_dir, "eval_results/segments/{}/{}.txt".format(self.segment, "test_list")))
        with open(test_list, 'w') as f:
            for timestamp in self.time_stamp:
                timestamp_str = "{:<018}".format(int(timestamp*1e6))
                item_str = "{}/{}.jpg".format(self.segment, timestamp_str)
                f.write(item_str + '\n')
        f1_smooth = eval_lane3d(dataset_dir, pred_dir, test_list, output_dir, "smooth")
        pred_dir = os.path.abspath(os.path.join(cfg.output_dir, "results_det/")) + "/"
        f1_pers = eval_lane3d(dataset_dir, pred_dir, test_list, output_dir, "PersFormer")
        return {
            "f1_smooth": f1_smooth,
            "f1_pers": f1_pers
        }

    def save_for_visualization(self, frame):
        if self.eval_pose_only:
            return
        timestamp_str = "{:<018}".format(int(frame.timestamp*1e6))
        result = {'file_path': "validation/{}/{}.jpg".format(self.segment, timestamp_str),
                  'pose': frame.T_wc,
                  'pose_raw': self.raw_pose[-1],
                  'pose_gt': self.gt_pose[-1],
                  'timestamp': frame.timestamp}
        lanes_in_frame = []
        for lane in frame.get_lane_features():
            lane_dict = {'id': lane.id, 'category': lane.category, 'xyz': lane.points}
            lanes_in_frame.append(lane_dict)
        result['lanes_in_frame'] = lanes_in_frame

        lanes_in_map = {}
        for lane in self.lanes_in_map.values():
            lane_dict = {'category': lane.category,
                         'xyz': lane.points, 'xyz_raw': lane.raw_points,
                         'ctrl_pts': np.array([node.item for node in lane.ctrl_pts.get_nodes()])}
            lanes_in_map[lane.id] = lane_dict
        result['lanes_in_map'] = lanes_in_map

        output_dir = os.path.join(cfg.output_dir, 'visualization', self.segment)
        mkdir_if_missing(output_dir)
        npy_file = os.path.join(output_dir, '{}.npy'.format(timestamp_str))
        np.save(npy_file, result)

    def save_map(self):

        result = {}
        lanes_in_map = {}
        for lane in self.lanes_in_map.values():
            lane_dict = {'category': lane.category,
                         'xyz': lane.points, 'xyz_raw': lane.raw_points,
                         'ctrl_pts': np.array([node.item for node in lane.ctrl_pts.get_nodes()])}
            lanes_in_map[lane.id] = lane_dict
        result['lanes_in_map'] = lanes_in_map

        output_dir = os.path.join(cfg.output_dir, 'visualization', self.segment)
        mkdir_if_missing(output_dir)
        npy_file = os.path.join(output_dir, '{}.npy'.format("map"))
        np.save(npy_file, result)

    def dataset_inspect(self):
        stats = {}
        path_length = 0
        period = 0
        for frame_id, (frame_data_prev, frame_data_cur) in enumerate(tqdm.tqdm(zip(self.frames_data[:-1], self.frames_data[1:]), leave=False, dynamic_ncols=True)):
            pose_wc_cur = frame_data_cur['gt_pose']
            timestamp_cur = frame_data_cur['timestamp']
            pose_wc_prev = frame_data_prev['gt_pose']
            timestamp_prev = frame_data_prev['timestamp']
            path_length += np.linalg.norm(pose_wc_cur[:3, 3] - pose_wc_prev[:3, 3])
            period += timestamp_cur - timestamp_prev
        stats['path_length'] = path_length
        stats['path_gap'] = path_length / (len(self.frames_data)-1)
        stats['num_frames'] = len(self.frames_data)
        stats['period'] = period / (len(self.frames_data)-1)
        return stats

    def map_size(self):
        total_size = 0
        for lane in self.lanes_in_map.values():
            total_size += lane.size()
        return total_size