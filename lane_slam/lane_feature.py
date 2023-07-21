#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from misc.config import cfg
from lane_slam.linked_points import LinkedPoints
from misc.curve.catmull_rom import CatmullRomSplineList
from lane_slam.lane_utils import points_downsample, lane_denoise, robust_poly1d
from misc.plot_utils import visualize_points_list, pointcloud_to_spheres, text_3d
from time import perf_counter
from scipy.spatial.transform import Rotation as R
from misc.pcd_utils import compute_plane, make_open3d_point_cloud

class LaneFeature:
    # x, y, z, category, visibility, track_id
    def __init__(self, id, points, category):
        self.id = id
        self.points = points
        self.noise = self.compute_noise(points)
        self.category = category
        self.kdtree = None
        self.ctrl_pts = LinkedPoints()
        self.initialized = False
        self.ctrl_points_chord = cfg.lane_mapping.ctrl_points_chord
        self.downsample = cfg.preprocess.downsample
        self.polyline = {
            'f_yx': None,
            'f_zx': None,
            'rot': None,
        }
        self.raw_points = points
        self.vis_frame_ids = []

        self.obs_num = 0
        self.obs_first_frame_id = None
        self.obs_last_frame_id = None

    def add_obs_frame_id(self, frame_id):
        self.obs_num += 1
        if self.obs_first_frame_id is None:
            self.obs_first_frame_id = frame_id
        self.obs_last_frame_id = frame_id

    def compute_noise(self, xyz):
        x_min, x_max, y_min, y_max = cfg.preprocess.range_area
        lower_bound, upper_bound = cfg.lane_mapping.lane_meas_noise
        max_range2 = x_max**2 + y_max**2
        dist2 = xyz[:, 0]**2 + xyz[:, 1]**2
        ratio = np.clip(np.sqrt(dist2 / max_range2), 0, 1.0)
        noise = lower_bound + (upper_bound - lower_bound) * ratio
        return noise

    def update_raw_pts(self, lane_feature, frame_id):
        # use KDTree to update lane with downsampling new lane points
        if frame_id not in self.vis_frame_ids:
            self.vis_frame_ids.append(frame_id)
            points_to_add = lane_feature.get_xyzs()
            self.raw_points = np.vstack((self.raw_points, points_to_add))
            self.raw_points = points_downsample(self.raw_points, cfg.vis_downsample)

    def smooth(self):
        ctrl_pts = self.get_ctrl_xyz()
        if len(ctrl_pts) >= 2:
            curve = CatmullRomSplineList(np.array(ctrl_pts))
            num = int(self.ctrl_points_chord / self.downsample) + 1
            fitted_pts = curve.get_points(num)
            self.points = fitted_pts
        self.kdtree = KDTree(self.points[:, :cfg.preprocess.dim])

    def init_ctrl_pts(self, lane_w, cur_pose_cw):
        # the use of inputing cur_pose_cw is to make sure the first ctrl point is cloest to the car
        # It is based on the assumption that the car is heading to the forward direction
        # initialize the ctrl points
        self.get_skeleton(lane_w.get_xyzs(), polyline = lane_w.polyline)

        head = self.ctrl_pts.get_xyz(0)
        tail = self.ctrl_pts.get_xyz(-1)
        head = cur_pose_cw[:3, :3].dot(head) + cur_pose_cw[:3, 3]
        tail = cur_pose_cw[:3, :3].dot(tail) + cur_pose_cw[:3, 3]
        if np.linalg.norm(head) > np.linalg.norm(tail):
            self.ctrl_pts.reverse()
        self.initialized = True

    def get_pts_to_add(self, points):
        # points: N x 3
        ctrl_pts_size = self.ctrl_pts.size()
        if ctrl_pts_size < 2:
            return points
        else:
            normal_a = (self.ctrl_pts.get_xyz(0) - self.ctrl_pts.get_xyz(1))
            normal_b = (self.ctrl_pts.get_xyz(-1) - self.ctrl_pts.get_xyz(-2))
        pts_to_add = []
        for pt in points:
            d_a = pt - self.ctrl_pts.get_xyz(0)
            d_b = pt - self.ctrl_pts.get_xyz(-1)
            cos_a = np.dot(d_a, normal_a) / (np.linalg.norm(d_a) * np.linalg.norm(normal_a))
            cos_b = np.dot(d_b, normal_b) / (np.linalg.norm(d_b) * np.linalg.norm(normal_b))
            thd = np.cos(np.deg2rad(cfg.lane_mapping.skeleton_angle_thd))
            if cos_a > thd or cos_b > thd:
                pts_to_add.append(pt)
        pts_to_add = np.array(pts_to_add)
        return pts_to_add

    def update_ctrl_pts(self, lane_w):
        # create squential points
        # 用离控制点比较远的那些点来更新控制点
        lane_w_points = lane_w.get_xyzs()
        succ = self.get_skeleton(lane_w_points, self.ctrl_pts.get_xyz(-1), polyline = lane_w.polyline)
        return succ

    def fitting(self):
        # fitting the lane
        if self.points.shape[0] == 1:
            return
        # determine the order of the polynomial
        order = 3
        if self.points.shape[0] <= 3:
            order = self.points.shape[0] - 1
        principal_axis = self.points[-1, :2] - self.points[0, :2]
        expected_axis = np.array([1, 0])
        angle = np.arccos(np.dot(principal_axis, expected_axis) / (np.linalg.norm(principal_axis) * np.linalg.norm(expected_axis)))
        if np.cross(principal_axis, expected_axis) < 0:
            angle = -angle
        rot = R.from_rotvec(angle * np.array([0, 0, 1]))
        xyz = rot.apply(self.points)
        x_g, y_g, z_g = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        f_yx = robust_poly1d(x_g, y_g, order)
        f_zx = robust_poly1d(x_g, z_g, order)
        self.polyline = {
            'f_yx': f_yx,
            'f_zx': f_zx,
            'rot': rot
        }

    def get_polyline_points(self):
        points = self.polyline['rot'].apply(self.points)
        max_x = np.max(points[:, 0])
        min_x = np.min(points[:, 0])
        x = np.linspace(min_x, max_x, 100)
        y = self.polyline['f_yx'](x)
        z = self.polyline['f_zx'](x)
        xyz = np.vstack((x, y, z)).T
        xyz = self.polyline['rot'].inv().apply(xyz)
        return xyz

    def dist2ctrlpts(self, pt):
        dist = []
        last_id = -1
        for ctrl_pt in self.ctrl_pts.items():
            last_id += 1
            dist.append(np.linalg.norm(pt - ctrl_pt))
        min_id = np.argmin(dist)
        min_dist = dist[min_id]
        return min_dist, min_id==last_id or min_id==0

    def find_border_point_kdtree(self, query, points, no_assigned):
        kdtree = KDTree(points)
        upper_bound = cfg.lane_mapping.ctrl_points_chord
        num = kdtree.query_ball_point(query, r=upper_bound, return_length=True)
        dist, idx = kdtree.query(query, k=num+1)
        inner_border = None if num == 0 else points[idx[-2]]
        if num == 0:
            outer_border = None if dist == np.inf else points[idx]
            no_assigned.remove(idx)
        else:
            outer_border = None if dist[-1] == np.inf else points[idx[-1]]
            for i in idx[:-1]:
                no_assigned.remove(i)
        return inner_border, outer_border

    def find_border_point(self, query, points, no_assigned):
        min_dist = np.inf
        outer_border = None
        max_dist = 0
        inner_border = None
        for i in range(points.shape[0]):
            dist = np.linalg.norm(points[i] - query)
            if dist < self.ctrl_points_chord:
                no_assigned.remove(i)
                if dist > max_dist:
                    max_dist = dist
                    inner_border = points[i]
            else:
                if dist < min_dist:
                    min_dist = dist
                    outer_border = points[i]
        return inner_border, outer_border

    def get_skeleton(self, origin_points:np.ndarray, inital_point=None, polyline=None):
        # input: N*3 points
        if inital_point is None:
            # rand_id = np.random.choice(no_assigned)
            # inital_point = origin_points[rand_id]
            inital_point = origin_points[0]
            self.ctrl_pts.add(inital_point)
        num = 0
        origin_points_debug = origin_points.copy()
        # origin_points = self.get_pts_to_add(origin_points)
        while True:
            num += 1
            # if self.id == 2 and self.ctrl_pts.size() >= 43:
            #     self.id = self.id
            origin_points = self.get_pts_to_add(origin_points)
            if origin_points.shape[0] == 0:
                return True
            no_assigned = np.arange(origin_points.shape[0]).tolist()
            # find the farthest point in a radius
            if origin_points.shape[0] <= 15:
                inner_border, outer_border = self.find_border_point(inital_point, origin_points, no_assigned)
            else:
                inner_border, outer_border = self.find_border_point_kdtree(inital_point, origin_points, no_assigned)
            if outer_border is None: # this is equal to the case that no_assigned is empty
                d_head = np.linalg.norm(self.ctrl_pts.get_xyz(0) - inner_border)
                d_tile = np.linalg.norm(self.ctrl_pts.get_xyz(-1) - inner_border)
                if d_head <= d_tile: # head is closer to the farthest point
                    center = self.ctrl_pts.get_xyz(0)
                    if self.ctrl_pts.size()>=2:
                        inner_border = self.get_query(self.ctrl_pts.get_xyz(1), self.ctrl_pts.get_xyz(0), polyline)
                    next_initial = self.get_next_node(inner_border, self.ctrl_pts.get_xyz(0), self.ctrl_points_chord, polyline)
                    self.ctrl_pts.add(next_initial)
                else:#if tile is closer or equal
                    center = self.ctrl_pts.get_xyz(-1)
                    if self.ctrl_pts.size()>=2:
                        inner_border = self.get_query(self.ctrl_pts.get_xyz(-2), self.ctrl_pts.get_xyz(-1), polyline)
                    next_initial = self.get_next_node(inner_border, self.ctrl_pts.get_xyz(-1), self.ctrl_points_chord, polyline)
                    self.ctrl_pts.append(next_initial)

                # if self.id == 2 and self.ctrl_pts.size() == 106:
                    # self.plot(origin_points)
                    # self.plot_debug(points=origin_points, query=inner_border, center=center, next_initial=next_initial)
                return True
            else:
                d_head = np.linalg.norm(self.ctrl_pts.get_xyz(0) - outer_border)
                d_tile = np.linalg.norm(self.ctrl_pts.get_xyz(-1) - outer_border)
                # if self.id == 5 and self.ctrl_pts.size() == 105:
                #     self.id = self.id
                if d_head <= d_tile: # head is closer to the farthest point
                    center = self.ctrl_pts.get_xyz(0)
                    if self.ctrl_pts.size()>=2:
                        outer_border = self.get_query(self.ctrl_pts.get_xyz(1), self.ctrl_pts.get_xyz(0), polyline)
                    next_initial = self.get_next_node(outer_border, self.ctrl_pts.get_xyz(0), self.ctrl_points_chord, polyline)
                    self.ctrl_pts.add(next_initial)
                    # print("add head ", d_head, " ", d_tile)
                else:#if tile is closer or equal
                    center = self.ctrl_pts.get_xyz(-1)
                    if self.ctrl_pts.size()>=2:
                        outer_border = self.get_query(self.ctrl_pts.get_xyz(-2), self.ctrl_pts.get_xyz(-1), polyline)
                    next_initial = self.get_next_node(outer_border, self.ctrl_pts.get_xyz(-1), self.ctrl_points_chord, polyline, points_debug = origin_points_debug)
                    self.ctrl_pts.append(next_initial)

                # if self.id == 2 and self.ctrl_pts.size() >= 43:
                #     dist = np.linalg.norm(next_initial - center)
                #     self.plot_debug(points=origin_points_debug, query=outer_border, center=center, next_initial=next_initial, polyline = polyline)

                    # print("add tail")
                inital_point = next_initial
                origin_points = origin_points[no_assigned]

    def get_query(self, start_pt, end_pt, polyline):
        start_pt_new = polyline['rot'].apply(start_pt)
        end_pt_new = polyline['rot'].apply(end_pt)
        direction = end_pt_new[0] - start_pt_new[0]
        if direction > 0:
            direction = 1
        else:
            direction = -1
        x_new = end_pt_new[0] + direction * 10
        query = np.array([x_new, polyline['f_yx'](x_new), polyline['f_zx'](x_new)])
        query = polyline['rot'].inv().apply(query)
        return query
    
    def get_nearest_on_circle(self, query, center, radius):
        query_new = query - center
        phi = np.arctan2(query_new[1], query_new[0])
        theta = np.arctan2(np.linalg.norm(query_new[:2]), query_new[2])
        delta = np.array([radius * np.cos(phi) * np.sin(theta),
                                      radius * np.sin(phi) * np.sin(theta),
                                      radius * np.cos(theta)])
        nearest_on_circle = center + delta
        return nearest_on_circle
    
    def get_next_node(self, query, center, radius, polyline, points_debug=None):
        query_new = polyline['rot'].apply(query)
        center_new = polyline['rot'].apply(center)
        last_result = np.asarray([0, 0, 0])
        for i in range(10):
            # query_new_debug = polyline['rot'].inv().apply(query_new)
            nearest_on_circle = self.get_nearest_on_circle(query_new, center_new, radius)

            x = nearest_on_circle[0]
            y_on_polyline = polyline['f_yx'](x)
            z_on_polyline = polyline['f_zx'](x)
            query_new = np.array([x, y_on_polyline, z_on_polyline])
            # query_new_next = polyline['rot'].inv().apply(query_new)
            delta = np.linalg.norm(query_new - last_result)

            # if self.id == 2 and self.ctrl_pts.size() == 105:
            #     self.plot_debug(points=points_debug, query=query_new_debug, center=center, next_initial=query_new_next, polyline = polyline)
            if delta < 1e-2:
                break
            last_result = query_new

        node_on_polyline = polyline['rot'].inv().apply(query_new)
        nearest_on_circle = self.get_nearest_on_circle(node_on_polyline, center, radius)
        return nearest_on_circle

    def self_check(self):
        # check if the length of the polyline is shorter than 2.0 m
        if len(self.points) * self.downsample < 2.0:
            return False
        else:
            return True

    def get_xyzs(self):
        return self.points[:, :3]

    def get_points(self):
        return self.points

    def get_ctrl_nodes(self):
        return self.ctrl_pts.get_nodes()

    def get_ctrl_node(self, idx):
        return self.ctrl_pts.get_node(idx)

    def get_ctrl_xyz(self):
        return self.ctrl_pts.get_xyzs()

    def get_ctrl_pt_idx(self, node):
        return self.ctrl_pts.items.index(node)

    def size(self):
        return self.ctrl_pts.size()

    def overlap_ratio(self, other):
        if self.kdtree is None or other.kdtree is None:
            return 0
        other_tree = other.kdtree
        dist_thd=cfg.lane_asso.lane_width / 2
        idx = self.kdtree.query_ball_tree(other_tree, r=dist_thd)
        num = 0
        for i in range(len(idx)):
            num += len(idx[i]) > 0
        overlap = num / self.points.shape[0]
        return overlap

    def plot(self, extra_pts=None):
        vis_pcds = self.plot_ctrl_pts()
        points_list = [self.points]
        if extra_pts is not None:
            points_list.append(extra_pts)
        visualize_points_list(points_list, extra_pcd=vis_pcds, axis_marker=0.5)

    def plot_ctrl_pts(self):
        ctrl_pts = self.ctrl_pts.get_points()
        ctrl_pcds = pointcloud_to_spheres(ctrl_pts)
        vis_pcds = [ctrl_pcds]
        for i in range(len(ctrl_pts)):
            text_pos = ctrl_pts[i] + np.array([1.5, -1.5, -1])
            text_pcd = text_3d(str(i), text_pos, font_size=100, degree=-90.0)
            vis_pcds.append(text_pcd)
        return vis_pcds
    # visualize the skeleton generating process
    def plot_debug(self, points = None, query=None, center=None, next_initial=None, polyline = None):
        if polyline is None:
            self_points = self.points[:, :3]
            points_vis = points
            query_vis = query
            center_vis = center
            next_initial_vis = next_initial
        else:
            self_points = polyline['rot'].apply(self.points[:, :3])
            points_vis = polyline['rot'].apply(points)
            query_vis = polyline['rot'].apply(query)
            center_vis = polyline['rot'].apply(center)
            next_initial_vis = polyline['rot'].apply(next_initial)
        vis_points = [self_points, points_vis]
        vis_pcds = self.plot_ctrl_pts() if polyline is None else []
        if query is not None:
            query_pcd = pointcloud_to_spheres(query_vis.reshape(1,3), color = (1, 0.706, 0))
            text_pcd = text_3d("Q", query_vis + np.array([1.5, -1.5, -1]), font_size=100, degree=-90.0)
            vis_pcds.extend([query_pcd, text_pcd])
        if center is not None:
            center_pcd = pointcloud_to_spheres(center_vis.reshape(1,3), color = (0, 1, 0))
            # draw a circle
            circle_pts = []
            for i in range(100):
                phi = i * 2 * np.pi / 100
                circle_pts.append([np.cos(phi), np.sin(phi), 0])
            circle_pts = np.array(circle_pts) * self.ctrl_points_chord + center_vis
            vis_points.append(circle_pts)
            text_pcd = text_3d("C", center_vis + np.array([1.5, -1.5, -1]), font_size=100, degree=-90.0)
            vis_pcds.extend([center_pcd, text_pcd])
        if next_initial is not None:
            next_initial_pcd = pointcloud_to_spheres(next_initial_vis.reshape(1,3), color = (0, 0, 1))
            text_pcd = text_3d("N", next_initial_vis + np.array([1.5, -1.5, -1]), font_size=100, degree=-90.0)
            vis_pcds.extend([next_initial_pcd, text_pcd])
        visualize_points_list(vis_points, extra_pcd=vis_pcds, axis_marker=10)