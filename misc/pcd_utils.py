#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def knn(query, points, k=1):
    # query: (3, )
    # points: (N, 3)
    # return: (k, 3), (k, )
    dist = np.sum((points - query) ** 2, axis=1)
    idx = np.argsort(dist)
    return dist[idx[:k]], idx[:k]

def compute_plane(normal, point_on_plane):
    # normal: (3, )
    # point_on_plane: (3, )
    # return: ax + by + cz + d = 0
    d = -np.dot(normal, point_on_plane)
    return np.concatenate((normal, [d]))

def dbscan_cluster(pcd, eps=0.02, min_points=10):
    # np.darray, (N, ), each element is the cluster label(0~max), if -1, it is noise
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

    id = np.unique(labels)
    clusters = []
    for i in id:
        if i == -1:
            continue
        cluster = pcd.select_by_index(np.where(labels == i)[0])
        clusters.append(cluster)
    return clusters

def make_open3d_point_cloud(xyz, color=None):
    # pcd.paint_uniform_color([1, 0.706, 0])
    # pcd.paint_uniform_color([0, 0.651, 0.929])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, :3])
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd

def split_lane_by_id(lane_points):
    lane_points = lane_points[lane_points[:, -1] >= 0]
    track_ids = np.unique(np.round(lane_points[:, -1]))
    # print('track_ids', track_ids)
    # category = np.unique(lane_points[:, -3])
    # print('category', category)
    lane_points_split = []
    for track_id in sorted(track_ids):
        lane_points_split.append(lane_points[np.round(lane_points[:, -1]) == track_id, :])
    return lane_points_split

def split_lane_by_category(lane_points):
    place = 3
    lane_points = lane_points[lane_points[:, place] > 0]
    track_ids = np.unique(np.round(lane_points[:, place]))
    lane_points_split = []
    for track_id in track_ids:
        lane_points_split.append(lane_points[np.round(lane_points[:, place]) == track_id, :])
    return lane_points_split

def transform_points(points, transform):
    # transform: (4, 4)
    # points: (N, 3)
    points_3d = points[:, :3]
    points_3d = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    points_3d = np.dot(transform, points_3d.T).T
    points[:, :3] = points_3d[:, :3]

    return points