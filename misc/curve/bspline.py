#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk
import numpy as np
from misc.plot_utils import visualize_points_list, pointcloud_to_spheres

class CubicBSplineCurve:
    # 三次B样条曲线, 控制点对应的knots为0,1,2,3,4,5,6,7
    def __init__(self, ctrlpts = None, knots = None):
        # ctrlpts: (4, 3) np.darray
        self.ctrlpts = ctrlpts
        self.knots = knots

    def plot(self, num = 10, raw_points=None, sphere_size=0.1):
        points = self.get_self_points(num)
        ctrlpts_sphere = pointcloud_to_spheres(self.ctrlpts, sphere_size=sphere_size, color=[0.7, 0.3, 0.3])
        if raw_points is not None:
            visualize_points_list([points, raw_points], extra_pcd=[ctrlpts_sphere])
        else:
            visualize_points_list([points], extra_pcd=[ctrlpts_sphere])

    def get_points_final(self, num = 10, return_knots=False):
        if self.knots is None:
            knots = np.linspace(0, 1, num).reshape(-1, 1)
        else:
            knots = np.linspace(self.knots[0], self.knots[-1], num).reshape(-1, 1)
        return self.get_points(knots, return_knots=return_knots)

    def get_self_points(self, num = 10, return_knots=False):
        if self.knots is None:
            knots = np.linspace(0, 1, num).reshape(-1, 1)
        else:
            knots = self.knots
        return self.get_points(knots, return_knots=return_knots)

    def get_points(self, knots, return_knots=False):
        # knots: (N,1) np.darray
        if knots.ndim == 1:
            knots = knots.reshape(-1,1)

        u3 = knots ** 3
        u2 = knots ** 2
        u1 = knots
        u0 = np.ones_like(knots)
        knots_cat = np.hstack([u3, u2, u1, u0])

        M_b = np.asarray([[-1, 3, -3, 1],
                            [3, -6, 3, 0],
                            [-3, 0, 3, 0],
                            [1, 4, 1, 0]]) / 6

        points = np.matmul(knots_cat, np.matmul(M_b, self.ctrlpts))
        if return_knots:
            points = np.hstack([points, knots])

        return points

class BSplineCurve:
    def __init__(self, ctrlpts, degree=3, type="uniform"):
        self.ctrlpts = ctrlpts
        self.degree = degree
        self.num_ctrlpts = ctrlpts.shape[0]
        self.num_knots = self.num_ctrlpts + degree + 1
        self.degree = degree
        self.type = type
        if self.type == "uniform":
            self.knots = np.arange(self.num_knots)
        elif self.type == "clamped":
            self.knots = [0] * (degree + 1) + [i for i in range(1, ctrlpts.shape[0] - degree)] + [ctrlpts.shape[0] - degree] * (degree + 1)
        elif self.type == "nonuniform":
            self.knots = np.asarray([0, 1])
            self.knots = np.hstack([self.knots, np.random.uniform(0.01, 0.99, self.num_knots - 2)])
            self.knots = np.sort(self.knots)
        else:
            raise ValueError("Unknown type: {}".format(type))
        self.knots = np.asarray(self.knots, dtype=np.float32)

    def get_ubs_coeff(self, u):
        # u: (N, 1) np.darray
        # return: (N, 4) np.darray
        knot_vector = np.hstack([u ** 3, u ** 2, u, np.ones_like(u)])
        M_b = np.asarray([[-1, 3, -3, 1],
                          [3, -6, 3, 0],
                          [-3, 0, 3, 0],
                          [1, 4, 1, 0]]) / 6
        coeff_mat = np.matmul(knot_vector, M_b) # (N, 4) @ (4, 4) = (N, 4)
        return coeff_mat


    def print_infos(self):
        print("Print BSplineCurve infos:")
        print("type: ", self.type)
        np.set_printoptions(precision=3)
        print("ctrlpts: ", end="")
        for ctrlpt in self.ctrlpts:
            print(ctrlpt, end=" ")
        print()
        print("knots: ", self.knots)
        mean_knots = np.diff(self.knots) / 2.0 + self.knots[:-1]
        for idx, u in enumerate(mean_knots):
            weight = np.zeros(self.num_ctrlpts)
            for i in range(self.num_ctrlpts):
                weight[i] = self.basis_function(i, self.degree, u)
            if self.knots[idx] != self.knots[idx + 1]:
                new_u = np.asarray([(u - self.knots[idx]) / (self.knots[idx + 1] - self.knots[idx])]).reshape(-1, 1)
                coeff = self.get_ubs_coeff(new_u).reshape(-1)
                print("u: ", u, ", weight: ", weight, ", sum: {:.3f}".format(np.sum(weight)),
                      ", coeff: ", coeff, ", sum: {:.3f}".format(np.sum(coeff)))
            else:
                print("u: ", u, ", weight: ", weight, ", sum: {:.3f}".format(np.sum(weight)))

    def basis_function(self, i, p, u):
        if p == 0:
            if self.knots[i] <= u < self.knots[i + 1]:
                return 1
            else:
                return 0
        else:
            if self.knots[i + p] == self.knots[i]:
                c1 = 0
            else:
                c1 = (u - self.knots[i]) / (self.knots[i + p] - self.knots[i]) * self.basis_function(i, p - 1, u)
            if self.knots[i + p + 1] == self.knots[i + 1]:
                c2 = 0
            else:
                c2 = (self.knots[i + p + 1] - u) / (self.knots[i + p + 1] - self.knots[i + 1]) * self.basis_function(i + 1, p - 1, u)
            return c1 + c2

def bspline_demo():
    ctrlpts = np.array([[1, 2, 0], [2, 1, 2], [3, 1, 1], [4, 3, 2]]) * 5
    # bsplne3 = CubicBSplineCurve(ctrlpts)
    # bsplne3.draw()
    bspline3 = BSplineCurve(ctrlpts, 3)
    bspline3.draw()

