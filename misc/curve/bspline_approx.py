#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk
import numpy as np
from misc.curve.bspline import CubicBSplineCurve
from scipy.spatial import KDTree
import gtsam
from gtsam.symbol_shorthand import P
import gtsam.utils.plot as gtsam_plot
from typing import List, Optional
from functools import partial
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

class CubicBSplineApproximator:
    # default four control points
    def __init__(self, max_iter = 20, res_delta_tld = 5e-2):
        self.degree = 3
        self.max_iter = max_iter
        self.res_delta_tld = res_delta_tld

    def approximate(self, points, method)->CubicBSplineCurve:
        # points: (N, 3) np.darray
        points = points[:, :3]
        if method == "iterative":
            u = self.get_u_xyz_norm(points)
            curve_init = self.ls_fitting(u, points)
            curve = self.approx_iter(points, curve_init)
        elif method == "chord_length":
            u = self.get_u_chord(points)
            curve = self.ls_fitting(u, points)
        elif method == "xyz_norm":
            u = self.get_u_xyz_norm(points)
            curve = self.ls_fitting(u, points)
        else:
            raise NotImplementedError

        return curve

    def ls_fitting(self, u, points, weights = None):
        # u: (N,1) np.darray
        weights = np.eye(points.shape[0]) if weights is None else weights
        coeff_mat = self.get_coeff_mat(u) # (N, 4)
        # (N, 4) @ (4, 3) = (N, 3), points: (N, 3)
        ctrlpts = np.linalg.inv(coeff_mat.T @ weights @ coeff_mat) @ (weights @ coeff_mat).T @ points
        curve = CubicBSplineCurve(ctrlpts, u)
        return curve

    # 随着迭代次数增加，控制点反而远离真实点，并且knots往中间靠拢
    def approx_iter(self, points, curve:CubicBSplineCurve):
        residuals = np.linalg.norm(points - curve.get_self_points(), axis=1)
        weights = np.diag(residuals / np.max(residuals))
        last_res_mean = i = 0
        while i < self.max_iter:
            knots = self.get_u_nearest(points, curve)
            curve = self.ls_fitting(knots, points, weights)
            residuals = np.linalg.norm(points - curve.get_self_points(), axis=1)
            weights = np.diag(residuals / np.max(residuals))
            # print("iter: {}, residuals: {}".format(i, np.mean(residuals) - last_res_mean))
            if abs(np.mean(residuals) - last_res_mean) < self.res_delta_tld:
                break
            last_res_mean = np.mean(residuals)
            i += 1

        return curve

    def get_u_xyz_norm(self, points):
        # sorted by distance from zero
        distance = np.linalg.norm(points, axis = 1)
        min_dist = np.min(distance)
        max_dist = np.max(distance)
        knots = (distance - min_dist) / (max_dist - min_dist)
        return knots.reshape(-1, 1)

    def get_u_chord(self, points):
        distance = np.linalg.norm(points, axis = 1)
        points = points[distance.argsort()]
        # compute the distance between each two points
        dists = np.linalg.norm(points[1:] - points[:-1], axis = 1)
        # compute the cumulative distance
        cum_dists = np.cumsum(dists)
        # normalize the cumulative distance
        cum_dists /= cum_dists[-1]
        cum_dists = np.vstack([np.zeros((1,1)), cum_dists.reshape(-1, 1)])
        return cum_dists

    def get_u_nearest(self, points, curve):
        knots = np.linspace(0, 1, 10000).reshape(-1, 1)
        points_on_curve = curve.get_points(knots)
        #find nearest
        tree = KDTree(points_on_curve)
        dist, ind = tree.query(points, k=1)
        knots = knots[ind]
        return knots

    def get_coeff_mat(self, knots):
        # knots: (N,1) np.darray
        # return: (N, 4) np.darray
        knot_vector = np.hstack([knots ** 3, knots ** 2, knots, np.ones_like(knots)])
        M_b = np.asarray([[-1, 3, -3, 1],
                          [3, -6, 3, 0],
                          [-3, 0, 3, 0],
                          [1, 4, 1, 0]]) / 6 # non-singular
        coeff_mat = np.matmul(knot_vector, M_b) # (N, 4) @ (4, 4) = (N, 4)
        return coeff_mat


def error_cubic_bspline(measurement: np.ndarray, this: gtsam.CustomFactor,
                        values: gtsam.Values,
                        jacobians: Optional[List[np.ndarray]]) -> float:
    """GPS Factor error function
    :param measurement: GPS measurement, to be filled with `partial`
    :param this: gtsam.CustomFactor handle
    :param values: gtsam.Values
    :param jacobians: Optional list of Jacobians
    :return: the unwhitened error
    """

    ctrl_pt1 = values.atVector(this.keys()[0]).reshape(1,3) #np.ndarray
    ctrl_pt2 = values.atVector(this.keys()[1]).reshape(1,3)
    ctrl_pt3 = values.atVector(this.keys()[2]).reshape(1,3)
    ctrl_pt4 = values.atVector(this.keys()[3]).reshape(1,3)
    ctrl_pts = np.vstack((ctrl_pt1, ctrl_pt2, ctrl_pt3, ctrl_pt4))

    u = measurement[3].item()
    knots_cat = np.asarray([u**3, u**2, u, 1]).reshape(1,4)
    M_b = np.asarray([[-1, 3, -3, 1],
                      [3, -6, 3, 0],
                      [-3, 0, 3, 0],
                      [1, 4, 1, 0]]).reshape(4, 4) / 6
    coff = np.matmul(knots_cat, M_b)
    est_pt = np.matmul(coff, ctrl_pts).reshape(3,1)
    sample_pt = measurement[:3].reshape(3,1)
    error = est_pt - sample_pt

    if jacobians is not None:
        jacobians[0] = np.eye(3) * coff[0,0]
        jacobians[1] = np.eye(3) * coff[0,1]
        jacobians[2] = np.eye(3) * coff[0,2]
        jacobians[3] = np.eye(3) * coff[0,3]

    return error

class BSplineGridApproximator:
    def __init__(self, resolution = 5):
        self.all_nodes = {}
        self.x_res = self.y_res = resolution
        self.pt_noise_sigma = 0.2
        self.grid_info = {}
        self.is_u_norm = True

    def global_optimize(self, measured_points):

        assigned_pts = self.assign_points_to_grid(measured_points)

        mean_h = np.mean(assigned_pts[:,2])
        initial_estimate = gtsam.Values()
        for exist_grid, grid_dict in self.grid_info.items():
            x, y = exist_grid
            dx, dy = grid_dict['principal_x'], 1 - grid_dict['principal_x']
            for i in range(4):
                insert_x, insert_y = x - dx * i, y - dy * i
                symbol, is_exist = self.get_gtsam_symbol(insert_x, insert_y)
                if symbol in initial_estimate.keys():
                    continue
                if (insert_x, insert_y) in self.grid_info:
                    initial_estimate.insert(symbol, gtsam.Point3(self.grid_info[(insert_x, insert_y)]['mean_pt']))
                else:
                    initial_estimate.insert(symbol, gtsam.Point3(insert_x, insert_y, mean_h))

        factor_graph = gtsam.NonlinearFactorGraph()
        pt_noise_model = gtsam.noiseModel.Isotropic.Sigma(3, self.pt_noise_sigma)
        # pt_noise_model = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber(0.5), pt_noise_model)
        for k in range(assigned_pts.shape[0]):
            x, y, z, u, principal_x, grid_x, grid_y = assigned_pts[k]
            dx, dy = principal_x, 1 - principal_x
            ctrl_pt_key1, is_exist1 = self.get_gtsam_symbol(grid_x - 3 * dx, grid_y - 3 * dy)
            ctrl_pt_key2, is_exist2 = self.get_gtsam_symbol(grid_x - 2 * dx, grid_y - 2 * dy)
            ctrl_pt_key3, is_exist3 = self.get_gtsam_symbol(grid_x - dx, grid_y - dy)
            ctrl_pt_key4, is_exist4 = self.get_gtsam_symbol(grid_x, grid_y)
            assert is_exist1 and is_exist2 and is_exist3 and is_exist4
            meas_pt = np.asarray([x, y, z, u])
            gf = gtsam.CustomFactor(pt_noise_model, [ctrl_pt_key1, ctrl_pt_key2, ctrl_pt_key3, ctrl_pt_key4],
                                    partial(error_cubic_bspline, meas_pt))
            factor_graph.add(gf)

        # Initialize optimizer
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(factor_graph, initial_estimate, params)
        # Optimize the factor graph
        result = optimizer.optimize()

        print("Initial error: ", factor_graph.error(initial_estimate))
        print("Final error: ", factor_graph.error(result))

        marginals = gtsam.Marginals(factor_graph, result)
        for key, value in self.all_nodes.items():
            symbol = value['symbol']
            self.all_nodes[key]['xyz'] = result.atPoint3(symbol)
            self.all_nodes[key]['cov'] = marginals.marginalCovariance(symbol)

        all_fitted_points = []
        for exist_grid, grid_dict in self.grid_info.items():
            x, y = exist_grid
            dx, dy = grid_dict['principal_x'], 1 - grid_dict['principal_x']
            ctrl_pts = []
            for i in reversed(range(4)):
                x_, y_ = x - dx * i, y - dy * i
                ctrl_pts.append(self.all_nodes[(x_, y_)]['xyz'])
            ctrl_pts = np.asarray(ctrl_pts)
            bspline = CubicBSplineCurve(ctrl_pts)
            knots = np.linspace(grid_dict['min_u'], grid_dict['max_u'], 10)
            points_part = bspline.get_points(knots)
            all_fitted_points.append(points_part)
        all_fitted_points = np.concatenate(all_fitted_points, axis=0)
        return all_fitted_points

    def plot_result(self, meas_points, result_points):

        min_x = round(np.min(meas_points[:,0]) / self.x_res) * self.x_res - self.x_res / 2
        max_x = round(np.max(meas_points[:,0]) / self.x_res) * self.x_res + self.x_res / 2
        min_y = round(np.min(meas_points[:,1]) / self.y_res) * self.y_res - self.y_res / 2
        max_y = round(np.max(meas_points[:,1]) / self.y_res) * self.y_res + self.y_res / 2
        fig = plt.figure(figsize=(30, 30 * (max_y - min_y) / (max_x - min_x)))
        ax = fig.add_subplot(111)
        ax.scatter(meas_points[:,0], meas_points[:,1], c='r')
        ax.scatter(result_points[:,0], result_points[:,1], c='b')
        for key, value in self.all_nodes.items():
            gtsam_plot.plot_point2_on_axes(ax, value['xyz'][:2], 0.5, value['cov'][:2,:2])

        major_ticks_x = np.arange(min_x, max_x, self.x_res)
        ax.set_xticks(major_ticks_x)
        major_ticks_y = np.arange(min_y, max_y, self.y_res)
        ax.set_yticks(major_ticks_y)
        ax.grid(which='both', alpha=0.5)
        ax.grid(which='major', alpha=0.5)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        plt.show()


    def assign_points_to_grid(self, points):
        grid_dict = {}

        for i in range(points.shape[0]):
            x, y, z = points[i, :3]
            grid_x, grid_y, u_x, u_y = self.get_grid_info(x, y)
            if (grid_x, grid_y) not in grid_dict:
                grid_dict[(grid_x, grid_y)] = []
            grid_dict[(grid_x, grid_y)].append([x, y, z, u_x, u_y])

        #pricipal direction
        points_list = []
        for grid, points_xyz_uxy in grid_dict.items():
            if(len(points_xyz_uxy) < 5):
                continue
            # if grid != (0, 0):
            #     continue
            points_xyz_uxy = np.array(points_xyz_uxy)
            mean_pt = np.mean(points_xyz_uxy[:,:3], axis=0)
            xy_n = points_xyz_uxy[:,:2] - mean_pt[:2].reshape(1,2)
            u, s, vh = np.linalg.svd(xy_n.T.dot(xy_n), full_matrices=True)
            grid_x, grid_y = grid
            principal_x = abs(u[0, 0]) > abs(u[1, 0])
            # 法一：记录最大最小值，法二：归一化
            xyz = points_xyz_uxy[:,:3]
            u = points_xyz_uxy[:,3] if principal_x else points_xyz_uxy[:,4]
            if self.is_u_norm:
                u = (u - np.min(u)) / (np.max(u) - np.min(u))
            principal_vec = np.ones_like(u) if principal_x else np.zeros_like(u)
            grid_xy = np.asarray([grid_x, grid_y] * xyz.shape[0]).reshape(xyz.shape[0], 2)
            new_points = np.concatenate([xyz, u.reshape(-1,1), principal_vec.reshape(-1,1), grid_xy], axis=1)
            points_list.append(new_points)
            self.grid_info[grid] = {
                "principal_x": principal_x,
                "mean_pt": mean_pt,
                "max_u": np.max(u),
                "min_u": np.min(u)
            }
        return np.vstack(points_list)

    def get_grid_info(self, x, y):
        grid_x = round(x / self.x_res)
        grid_y = round(y / self.y_res)
        grid_center = np.array([grid_x * self.x_res, grid_y * self.y_res])
        u_xy = (np.array([x, y]) - grid_center) / np.array([self.x_res, self.y_res]) + 0.5
        return grid_x, grid_y, u_xy[0], u_xy[1]

    def get_gtsam_symbol(self, x, y):
        is_exist = True
        if (x, y) not in self.all_nodes:
            is_exist = False
            idx = len(self.all_nodes)
            self.all_nodes[(x, y)] = {
                "symbol": P(idx),
                "value": []
            }
        return self.all_nodes[(x, y)]['symbol'], is_exist

def bspline_approx_demo():
    ctrlpts = np.array([[1, 2, 0], [2, 1, 2], [3, 1, 1], [4, 3, 2]]) * 5
    bsplne3 = CubicBSplineCurve(ctrlpts)
    points = bsplne3.get_points(np.linspace(0, 1, 100).reshape(-1, 1))

    points = np.random.rand(100, 3) + points
    bspline_approx = CubicBSplineApproximator(points)
    bspline_approx.approximate()
    # bspline_approx.curve.plot(points)
    # print("ctrlpts: ", ctrlpts)
    # print("fitted ctrlpts: ", bspline_approx.curve.ctrlpts)
    # print("ctrlpts residual: ", np.linalg.norm(bspline_approx.curve.ctrlpts - ctrlpts))