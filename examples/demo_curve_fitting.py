#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk

import numpy as np
import open3d as o3d
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(ROOT_DIR)
from misc.curve.bspline_approx import CubicBSplineApproximator, BSplineGridApproximator
from misc.curve.catmull_rom import CatmullRomSplineList, CentripetalCatmullRomSpline
import matplotlib.pyplot as plt

def main():
    lane_pcd= o3d.io.read_point_cloud(os.path.join(ROOT_DIR, "examples/data/lane.pcd"))
    lane_pcd = lane_pcd.voxel_down_sample(voxel_size=0.01)
    lane_points = np.asarray(lane_pcd.points)
    lane_key_points = np.asarray(o3d.io.read_point_cloud(os.path.join(ROOT_DIR, "examples/data/lane_selected.pcd")).points)

    # bspline approximation
    bspline_approximation(lane_points)

    # bspline grid approximation
    print("BSpline approximation using grid")
    b_grid_approx = BSplineGridApproximator(resolution=5)
    fitted_points = b_grid_approx.global_optimize(lane_points)
    b_grid_approx.plot_result(lane_points, fitted_points)

    # catmull-rom spline interpolation
    catmull_rom_interpolate(lane_points, lane_key_points)

    # ctrlpts_sphere = pointcloud_to_spheres(lane_key_points, sphere_size=0.5, color=[0.7, 0.1, 0.1])
    # o3d.visualization.draw_geometries([ctrlpts_sphere, lane_pcd])

def catmull_rom_interpolate(lane_points, lane_key_points):
    spline = CatmullRomSplineList(lane_key_points, tau=0.5)
    points_cr_0_5 = spline.get_points(20)
    # catmull-rom (seems to be sharper than centripetal)
    spline = CatmullRomSplineList(lane_key_points, tau=0.1)
    points_cr_0_1 = spline.get_points(20)

    spline = CentripetalCatmullRomSpline(lane_key_points, alpha=0.5)
    points_ccr_0_5 = spline.get_points(20)

    plt.figure(figsize=(15, 10))
    plt.plot(lane_points[:, 0], lane_points[:, 1], 'y.', label="lane points")
    plt.plot(points_cr_0_5[:, 0], points_cr_0_5[:, 1], 'b.', label="catmull-rom tau=0.5")
    plt.plot(points_cr_0_1[:, 0], points_cr_0_1[:, 1], 'g.', label="catmull-rom tau=0.1")
    plt.plot(points_ccr_0_5[:, 0], points_ccr_0_5[:, 1], 'c.', label="centripetal catmull-rom alpha=0.5")
    # Plot the control points
    plt.plot(lane_key_points[:, 0], lane_key_points[:, 1], 'or', label="control points")
    plt.title('Interpolating lane points')
    plt.legend(prop = {'size':14})
    plt.show()
    # plt.savefig("lane_fit.png")


def bspline_approximation(lane_points):
    # bspline approximation using the lane points
    print("BSpline approximation using the lane points")
    approximator = CubicBSplineApproximator(max_iter = 20, res_delta_tld = 5e-2)
    print("knots association using cumulative distance method")
    bspline3 = approximator.approximate(lane_points, method="chord_length")
    bspline3.plot(num=100, raw_points=lane_points, sphere_size=5)
    print("knots association using xyz_norm method")
    bspline3 = approximator.approximate(lane_points, method="xyz_norm")
    bspline3.plot(num=100, raw_points=lane_points, sphere_size=5)
    print("knots association using iterative method")
    bspline3 = approximator.approximate(lane_points, method="iterative")
    bspline3.plot(num=100, raw_points=lane_points, sphere_size=5)

if __name__ == '__main__':

    main()