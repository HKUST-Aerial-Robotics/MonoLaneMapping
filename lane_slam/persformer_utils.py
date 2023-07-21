import numpy as np
from copy import deepcopy

def transform_points_from_cam_to_ground(points, extrinsic):
    # points: [N, 3]

    lane = np.vstack((points[:,:3].T, np.ones((1, points.shape[0]))))
    transform = get_transform_from_cam_to_ground(extrinsic)
    lane = np.matmul(transform, lane)
    lane = lane[0:3, :].T

    return lane

def transform_points_from_ground_to_camera(points, extrinsic):
    # points: [N, 3]
    if points.shape[0] == 0:
        return points
    assert points.shape[1] == 3
    lane = np.vstack((points[:,:3].T, np.ones((1, points.shape[0]))))
    transform = get_transform_from_cam_to_ground(extrinsic)
    transform = np.linalg.inv(transform)
    lane = np.matmul(transform, lane)
    lane = lane[0:3, :].T

    return lane


def get_transform_from_cam_to_ground(extrinsics):
    # input: extrinsic: [3, 4]
    waymo_extrinsics = deepcopy(extrinsics)
    # Re-calculate extrinsic matrix based on ground coordinate
    R_vg = np.array([[0, 1, 0],
                     [-1, 0, 0],
                     [0, 0, 1]], dtype=float)
    R_gc = np.array([[1, 0, 0],
                     [0, 0, 1],
                     [0, -1, 0]], dtype=float)
    waymo_extrinsics[:3, :3] = np.matmul(np.matmul(
        np.matmul(np.linalg.inv(R_vg), waymo_extrinsics[:3, :3]),
        R_vg), R_gc)

    waymo_extrinsics[0:2, 3] = 0.0

    cam_representation = np.linalg.inv(
        np.array([[0, 0, 1, 0],
                  [-1, 0, 0, 0],
                  [0, -1, 0, 0],
                  [0, 0, 0, 1]], dtype=float))
    transform = np.matmul(waymo_extrinsics, cam_representation)

    return transform