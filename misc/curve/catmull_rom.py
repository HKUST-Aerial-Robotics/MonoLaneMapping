#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk
import numpy as np
from misc.pcd_utils import knn
from misc.config import cfg

def parameterization(pt_w, ctrl_pts_np, tau = 0.5, N = 4):
    # pt_w: 1x3, ctrl_pts: list of linked_points.Node
    # min N = 2
    spline = CatmullRomSpline(ctrl_pts_np, tau)
    anchors = spline.get_points(N, return_knots = True)

    # find the two anchors that pt_w is between
    dist, idx = knn(pt_w, anchors[:, :3], k = 2)
    pt1 = anchors[min(idx)]
    pt2 = anchors[max(idx)]
    v1 = pt2[:3] - pt1[:3]
    v2 = pt_w[:3] - pt1[:3]
    ratio = np.dot(v1, v2.T) / np.sum(v1**2) #除以v1的模，再除以v1的模得到0~1之间的比例
    u = pt1[3] + ratio * (pt2[3] - pt1[3])
    if u < 0 or u > 1:
        return None, None
    u = u.item()
    est_pt = spline.get_point(u)
    error = np.linalg.norm(pt_w[:3] - est_pt)

    return u, error.item()

class CatmullRomSpline:
    def __init__(self, ctrl_pts, tau = 0.5):
        # lower tau -> more sharp at control points
        try:
            self.tau = cfg.lane_mapping.tau
        except:
            self.tau = tau
        self.ctrl_pts = ctrl_pts
        self.num_ctrl_pts = ctrl_pts.shape[0]
        assert self.num_ctrl_pts == 4, "CatmullRomSpline only supports 4 control points"
        self.M = np.array([[0, 1, 0, 0],
                           [-tau, 0, tau, 0],
                           [2*tau, tau-3, 3-2*tau, -tau],
                           [-tau, 2-tau, tau-2, tau]])

    def get_M(self):
        return self.M

    def get_points(self, num, return_knots = False):
        # four_ctrl_pts: 4xc
        u = np.linspace(0, 1, num)
        u = u.reshape((num, 1))
        u_vec = np.hstack((np.ones((num, 1)), u, u**2, u**3)) # (num, 4)
        points = np.dot(u_vec, np.dot(self.M, self.ctrl_pts))
        if return_knots:
            points = np.hstack([points, u])
        return points

    def get_point(self, u, return_coeff = False):
        u_vec = np.array([1, u, u**2, u**3])
        if not return_coeff:
            point = np.dot(u_vec, np.dot(self.M, self.ctrl_pts))
            return point
        else:
            coeff = np.dot(u_vec, self.M)
            point = np.dot(coeff, self.ctrl_pts)
            return point, coeff

    def get_derivative(self, u):
        u_vec = np.array([0, 1, 2*u, 3*u**2])
        derivative = np.dot(u_vec, np.dot(self.M, self.ctrl_pts))
        derivative = derivative / np.linalg.norm(derivative)
        return derivative

class CatmullRomSplineList:
    def __init__(self, ctrl_pts, tau = 0.5):
        # lower tau -> more sharp at control points
        if ctrl_pts.shape[0] < 4:
            ctrl_pts = self.padding(ctrl_pts)
        self.tau = tau
        self.ctrl_pts = ctrl_pts
        self.num_ctrl_pts = ctrl_pts.shape[0]
        self.M = np.array([[0, 1, 0, 0],
                            [-tau, 0, tau, 0],
                            [2*tau, tau-3, 3-2*tau, -tau],
                            [-tau, 2-tau, tau-2, tau]])

    def padding(self, ctrl_pts):
        if ctrl_pts.shape[0] == 3:
            last_pt = ctrl_pts[2, :] + (ctrl_pts[2, :] - ctrl_pts[1, :])
            ctrl_pts = np.vstack((ctrl_pts, last_pt))
        elif ctrl_pts.shape[0] == 2:
            last_pt = ctrl_pts[1, :] + (ctrl_pts[1, :] - ctrl_pts[0, :])
            first_pt = ctrl_pts[0, :] - (ctrl_pts[1, :] - ctrl_pts[0, :])
            ctrl_pts = np.vstack((first_pt, ctrl_pts, last_pt))
        else:
            raise ValueError("ctrl_pts should have at least 2 points")
        return ctrl_pts

    def get_M(self):
        return self.M

    def get_points(self, num_points):
        points = []
        for i in range(self.num_ctrl_pts-3):
            four_ctrl_pts = self.ctrl_pts[i:i+4, :]
            spline = CatmullRomSpline(four_ctrl_pts, self.tau)
            if i != self.num_ctrl_pts-4:
                points.append(spline.get_points(num_points)[:-1])
            else:
                points.append(spline.get_points(num_points))
        return np.concatenate(points, axis=0)

class CentripetalCatmullRomSpline:
    def __init__(self, ctrl_pts, alpha = 0.5):
        # ctrl_pts: N x c
        # lower alpha -> more sharp at control points
        # if alpha = 0.0, it is equivalent to CatmullRomSpline with tau = 0.5
        self.ctrl_pts = ctrl_pts
        self.num_ctrl_pts = ctrl_pts.shape[0]
        self.alpha = alpha

    def get_points(self, n_points):
        points = []
        for i in range(self.num_ctrl_pts - 3):
            points.append(self.get_points_part(self.ctrl_pts[i:i+4], n_points))
        return np.concatenate(points, axis=0)

    def get_points_part(self, four_ctrl_pts, n_points):
        t0 = 0
        t1 = self.tj(t0, four_ctrl_pts[0], four_ctrl_pts[1])
        t2 = self.tj(t1, four_ctrl_pts[1], four_ctrl_pts[2])
        t3 = self.tj(t2, four_ctrl_pts[2], four_ctrl_pts[3])
        t = np.linspace(t1, t2, n_points)
        t = t.reshape(-1, 1)
        A1 = (t1 - t) / (t1 - t0) * four_ctrl_pts[0] + (t - t0) / (t1 - t0) * four_ctrl_pts[1]
        A2 = (t2 - t) / (t2 - t1) * four_ctrl_pts[1] + (t - t1) / (t2 - t1) * four_ctrl_pts[2]
        A3 = (t3 - t) / (t3 - t2) * four_ctrl_pts[2] + (t - t2) / (t3 - t2) * four_ctrl_pts[3]
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        points = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2
        return points

    def tj(self, ti, p_i, p_j):
        return (np.linalg.norm(p_i - p_j) ** self.alpha) + ti
