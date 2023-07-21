#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk
import numpy as np
from scipy.spatial.transform import Rotation as R
import jaxlie

def make_noisy_pose2d(pose, yaw_std, trans_std):
    # pose: 4x4
    ypr = [np.random.normal(0, yaw_std), 0, 0]
    pose_noisy = np.eye(4)
    pose_noisy[:3, :3] = R.from_euler('zyx', ypr, degrees=True).as_matrix()
    pose_noisy[:2, 3] += np.random.normal(0, trans_std, 2)
    pose_add_noise = pose @ pose_noisy
    return pose_add_noise

def get_pose2d_noise(yaw_std, trans_std):
    # pose: 4x4
    ypr = [np.random.normal(0, yaw_std), 0, 0]
    pose_noisy = np.eye(4)
    pose_noisy[:3, :3] = R.from_euler('zyx', ypr, degrees=True).as_matrix()
    pose_noisy[:2, 3] += np.random.normal(0, trans_std, 2)
    return pose_noisy

def se3_log(transform):
    # transform: (4, 4)
    # points: (N, 3)
    SE3 = jaxlie.SE3.from_matrix(transform)
    se3 = SE3.log()
    return np.asarray(se3)

def se3_exp(se3):
    # se3: (6, )
    SE3 = jaxlie.SE3.exp(se3)
    transform = SE3.as_matrix()
    return transform

def inv_se3(transform):
    return np.linalg.inv(transform)
    # transform: (4, 4)
    # points: (N, 3)
    inv_transform = np.eye(4)
    inv_transform[:3, :3] = transform[:3, :3].T
    inv_transform[:3, 3] = -np.dot(transform[:3, :3].T, transform[:3, 3])
    return inv_transform

def so3_to_quat(rot):
    r = R.from_matrix(rot)
    return r.as_quat()

def so3_to_rotvec(rot) -> np.ndarray:
    r = R.from_matrix(rot)
    rotvec = r.as_rotvec()
    return rotvec

def rad2deg(rad):
    return rad * 180.0 / np.pi

def se3_to_euler_xyz(transform):
    # transform: (4, 4)
    # points: (N, 3)
    r = R.from_matrix(transform[:3, :3])
    euler = r.as_euler('zyx', degrees=True)
    t = transform[:3, 3]
    return np.concatenate([euler, t]).tolist()

def compute_rpe(pose_est_i, pose_est_j, pose_gt_i, pose_gt_j):
    gt_ij = inv_se3(pose_gt_j) @ pose_gt_i
    est_ij = inv_se3(pose_est_j) @ pose_est_i
    delta_ij = inv_se3(est_ij) @ gt_ij
    delta_deg = rot_to_angle(delta_ij[:3, :3], deg=True)
    delta_xyz = np.linalg.norm(delta_ij[:3, 3])
    return delta_deg, delta_xyz

def rot_to_angle(rot, deg=True):
    rotvec = so3_to_rotvec(rot)
    angle = np.linalg.norm(rotvec)
    if deg:
        angle = rad2deg(angle)
    return angle