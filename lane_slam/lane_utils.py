import numpy as np
from lane_slam.linked_points import LinkedPoints
from misc.plot_utils import visualize_points_list, pointcloud_to_spheres
import math
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import open3d as o3d
from misc.config import cfg
from lane_slam.persformer_utils import transform_points_from_cam_to_ground, transform_points_from_ground_to_camera
from typing import List, Optional

def drop_lane_by_p(lane_pts:List, p=0.0):
    if p == 0.0:
        return lane_pts
    r = np.random.random()
    if r < p and len(lane_pts) > 1:
        drop_id = np.random.randint(0, len(lane_pts))
        lane_pts.pop(drop_id)
    return lane_pts


def robust_poly1d(x, y, order):
    fyx_order = order
    if x.shape[0] <= 3:
        fyx_order = x.shape[0] - 1
    if np.max(y) - np.min(y) < 0.1:
        fyx_order = 0
    elif np.max(y) - np.min(y) < 1:
        fyx_order = 2
        if x.shape[0] <= 2:
            fyx_order = x.shape[0] - 1
    f_yx = np.poly1d(np.polyfit(x, y, fyx_order))
    return f_yx

def prune_3d_lane_by_range(lane_3d, range_area):
    if len(range_area) == 4:
        x_min, x_max, y_min, y_max = range_area
    elif len(range_area) == 2:
        x_min, x_max = range_area
        y_min, y_max = 0, 103
    idx = np.logical_and(lane_3d[:, 0] > x_min, lane_3d[:, 0] < x_max)
    idx = np.logical_and(idx, lane_3d[:, 1] > y_min)
    idx = np.logical_and(idx, lane_3d[:, 1] < y_max)
    lane_3d = lane_3d[idx, ...]
    return lane_3d

def vis_lanes_dict(lane_gt, lane_pred, saved_lanes = None):
    lane_gt_list = []
    lane_pred_list = []
    extrinsic = np.asarray(cfg.dataset.extrinsic).astype(np.float32).reshape(4, 4)
    for lane in lane_gt:
        xyz = transform_points_from_cam_to_ground(lane['xyz'], extrinsic)
        xyz = prune_3d_lane_by_range(xyz, cfg.evaluation.eval_area)
        lane_gt_list.append(xyz)
    for lane in lane_pred:
        xyz = transform_points_from_cam_to_ground(lane['xyz'], extrinsic)
        xyz = prune_3d_lane_by_range(xyz, cfg.evaluation.eval_area)
        lane_pred_list.append(xyz)
    lane_gt_list = np.concatenate(lane_gt_list, axis=0)
    lane_pred_list = np.concatenate(lane_pred_list, axis=0)
    if saved_lanes is not None:
        lane_saved_list = []
        for lane in saved_lanes:
            xyz = transform_points_from_cam_to_ground(lane, extrinsic)
            xyz = prune_3d_lane_by_range(xyz, cfg.evaluation.eval_area)
            lane_saved_list.append(xyz)
        lane_saved_list = np.concatenate(lane_saved_list, axis=0)
        visualize_points_list([lane_gt_list, lane_pred_list, lane_saved_list])
    else:
        visualize_points_list([lane_gt_list, lane_pred_list])

# 不用open3d的原因是其只保留体素的中心点，而不是原始点
def points_downsample(points, size):
    max_x, max_y, max_z = np.max(points, axis=0)
    min_x, min_y, min_z = np.min(points, axis=0)
    downsampled_points = []
    has_points = np.zeros((math.ceil((max_x - min_x) / size) + 1,
                           math.ceil((max_y - min_y) / size) + 1,
                           math.ceil((max_z - min_z) / size) + 1))
    for p in points:
        x, y, z = p
        x_id = math.floor((x - min_x) / size)
        y_id = math.floor((y - min_y) / size)
        z_id = math.floor((z - min_z) / size)
        if has_points[x_id, y_id, z_id] == 0:
            downsampled_points.append(p)
            has_points[x_id, y_id, z_id] = 1
    return np.asarray(downsampled_points)

def linear_interp(xyz, interval):
    assert xyz.shape[0] == 2
    start, end = xyz[0, :], xyz[1, :]
    dist = np.linalg.norm(end - start)
    num = int(dist / interval)
    if num == 0:
        return xyz
    else:
        pts = []
        for i in range(num):
            pts.append(start + (end - start) * i / num)
        pts.append(end)
        return np.asarray(pts)

def lane_denoise(xyz:np.ndarray, order = 3, smooth = False, interval = None):
    if xyz.shape[0] < 2:
        return xyz
    elif xyz.shape[0] < 4:
        order = xyz.shape[0] - 1

    principal_axis = xyz[-1, :2] - xyz[0, :2]
    expected_axis = np.array([1, 0])
    angle = np.arccos(np.dot(principal_axis, expected_axis) / (np.linalg.norm(principal_axis) * np.linalg.norm(expected_axis)))
    if np.cross(principal_axis, expected_axis) < 0:
        angle = -angle
    rot = R.from_rotvec(angle * np.array([0, 0, 1]))
    xyz = rot.apply(xyz)
    x_g, y_g, z_g = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    f_yx = np.poly1d(np.polyfit(x_g, y_g, order))
    f_zx = np.poly1d(np.polyfit(x_g, z_g, order))
    y_fit = f_yx(x_g)
    z_fit = f_zx(x_g)
    xyz_fit = np.concatenate((x_g.reshape(-1,1), y_fit.reshape(-1,1), z_fit.reshape(-1,1)), axis=1)

    residual = np.linalg.norm(xyz_fit - xyz, axis=1)
    idx = np.where(residual > np.mean(residual) + 2 * np.std(residual))[0]
    xyz = np.delete(xyz, idx, axis=0)

    if smooth and interval is not None and xyz.shape[0] > 1:
        x_g, y_g, z_g = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        f_yx = np.poly1d(np.polyfit(x_g, y_g, order))
        f_zx = np.poly1d(np.polyfit(x_g, z_g, order))
        x_new = np.arange(np.min(x_g), np.max(x_g), interval/np.sqrt(2))
        y_fit = f_yx(x_new)
        z_fit = f_zx(x_new)
        xyz = np.concatenate((x_new.reshape(-1,1), y_fit.reshape(-1,1), z_fit.reshape(-1,1)), axis=1)

    # recover the original coordinate
    xyz = rot.inv().apply(xyz)

    return xyz