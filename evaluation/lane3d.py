import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from lane_slam.lane_utils import robust_poly1d, prune_3d_lane_by_range

class Lane3D:
    def __init__(self, xyz, category):
        self.xyz = xyz
        self.category = category
        self.kdtree = KDTree(xyz)
        self.size = self.xyz.shape[0]

    def similarity(self, other, ratio_thd, dist_thd):
        # check if two lanes are similar
        dist, idx = self.kdtree.query(other.xyz, k=1)
        num = 0
        # visualize_lanes(self.xyz, other.xyz)
        dists = []
        for i in range(other.size):
            if dist[i] < dist_thd:
                num += 1
                dists.append(dist[i])
        if num / other.size > ratio_thd:
            return True, dists
        else:
            return False, dists


def fitting(xyz):
    # fitting the lane
    if xyz.shape[0] == 1:
        return xyz
    # determine the order of the polynomial
    order = 3
    if xyz.shape[0] <= 3:
        order = xyz.shape[0] - 1
    principal_axis = xyz[-1, :2] - xyz[0, :2]
    expected_axis = np.array([1, 0])
    angle = np.arccos(
        np.dot(principal_axis, expected_axis) / (np.linalg.norm(principal_axis) * np.linalg.norm(expected_axis)))
    if np.cross(principal_axis, expected_axis) < 0:
        angle = -angle
    rot = R.from_rotvec(angle * np.array([0, 0, 1]))
    xyz = rot.apply(xyz)
    x_g, y_g, z_g = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    f_yx = robust_poly1d(x_g, y_g, order)
    f_zx = robust_poly1d(x_g, z_g, order)
    num = int((x_g.max() - x_g.min()) / 0.1)
    x = np.linspace(x_g.min(), x_g.max(), num)
    y = f_yx(x)
    z = f_zx(x)
    xyz = np.stack([x, y, z], axis=1)
    xyz = rot.inv().apply(xyz)
    return xyz

def visualize_lanes(xyz_pred, xyz_gt):
    # visualize the lanes
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(xyz_pred)
    pcd_pred.paint_uniform_color([0, 1, 0])
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(xyz_gt)
    pcd_gt.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([pcd_pred, pcd_gt])