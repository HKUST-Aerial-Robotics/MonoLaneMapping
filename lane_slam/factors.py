import gtsam
from typing import List, Optional
import numpy as np
from misc.curve.catmull_rom import CatmullRomSpline
from misc.config import cfg

def error_catmull_rom(measurement: np.ndarray, this: gtsam.CustomFactor,
                      values: gtsam.Values,
                      jacobians: Optional[List[np.ndarray]]) -> float:
    """catmull_rom Factor error function
    :param this: gtsam.CustomFactor handle
    :param values: gtsam.Values
    :param jacobians: Optional list of Jacobians
    :return: the unwhitened error
    """

    ctrl_pts = [values.atPoint3(this.keys()[i]) for i in range(4)]
    ctrl_pts = np.concatenate(ctrl_pts, axis=0).reshape(4,3)
    spline = CatmullRomSpline(ctrl_pts)
    u = measurement[3]

    est_pt, coeff = spline.get_point(u, return_coeff=True)

    error = (est_pt - measurement[:3]).reshape(3,1)

    if jacobians is not None:
        for i in range(4):
            jacobians[i] = np.eye(3) * coeff[i]

    return error

def PoseCurveTangentFactor(measurement: Optional[List[np.ndarray]], this: gtsam.CustomFactor,
                           values: gtsam.Values,
                           jacobians: Optional[List[np.ndarray]]) -> float:
    pt = measurement[0]
    pose:gtsam.Pose3 = values.atPose3(this.keys()[4])
    pt_est = pose.transformFrom(pt)

    u = measurement[1]
    ctrl_pts = [values.atPoint3(this.keys()[i]) for i in range(4)]
    ctrl_pts = np.concatenate(ctrl_pts, axis=0).reshape(4,3)
    spline = CatmullRomSpline(ctrl_pts)
    pt_meas, coeff = spline.get_point(u, return_coeff=True)

    p2p = (pt_est - pt_meas).reshape(3,1)

    proj = np.eye(3)
    di = spline.get_derivative(u).reshape(3, 1)
    proj = np.eye(3) - di @ di.T

    if jacobians is not None:
        for i in range(4):
            jacobians[i] = proj @ np.eye(3) * coeff[i]
        jacobians[4] = np.zeros((3,6))
        jacobians[4][:3,:3] = - pose.rotation().matrix() @ skew(pt)
        jacobians[4][:3,3:] = pose.rotation().matrix()
        jacobians[4] = proj @ jacobians[4]

    return proj @ p2p

def p2tan_catmull_rom(measurement: np.ndarray, this: gtsam.CustomFactor,
                      values: gtsam.Values,
                      jacobians: Optional[List[np.ndarray]]) -> float:
    """catmull_rom Factor error function
    :param this: gtsam.CustomFactor handle
    :param values: gtsam.Values
    :param jacobians: Optional list of Jacobians
    :return: the unwhitened error
    """

    ctrl_pts = [values.atPoint3(this.keys()[i]) for i in range(4)]
    ctrl_pts = np.concatenate(ctrl_pts, axis=0).reshape(4,3)

    spline = CatmullRomSpline(ctrl_pts)
    u = measurement[3]

    est_pt, coeff = spline.get_point(u, return_coeff=True)
    di = spline.get_derivative(u).reshape(3, 1)
    p2p = (measurement[:3] - est_pt).reshape(3,1)
    proj = np.eye(3) - di @ di.T
    error = proj @ p2p

    if jacobians is not None:
        for i in range(4):
            jacobians[i] = -proj * coeff[i]
            # print(i, jacobians[i])

    return error

def p2tan_factor(measurement: List, this: gtsam.CustomFactor,
                      values: gtsam.Values,
                      jacobians: Optional[List[np.ndarray]]) -> float:

    ctrl_pts = [measurement[1]]
    ctrl_pts.extend([values.atPoint3(this.keys()[i]) for i in range(2)])
    ctrl_pts.append(measurement[2])
    ctrl_pts = np.concatenate(ctrl_pts, axis=0).reshape(4,3)

    spline = CatmullRomSpline(ctrl_pts)
    pt_w = measurement[0]
    u = pt_w[3]

    est_pt, coeff = spline.get_point(u, return_coeff=True)
    di = spline.get_derivative(u).reshape(3, 1)
    p2p = (pt_w[:3] - est_pt).reshape(3,1)
    proj = np.eye(3) - di @ di.T
    error = proj @ p2p
    error_norm = np.linalg.norm(error)
    if error_norm > 5:
        print('error_norm', error_norm, 'key: ', this.keys()[0], this.keys()[1])

    if jacobians is not None:
        for i in range(2):
            jacobians[i] = -proj * coeff[i+1]
            # print(i, jacobians[i])

    return error

def p2tan_factor3(measurement: List, this: gtsam.CustomFactor,
                 values: gtsam.Values,
                 jacobians: Optional[List[np.ndarray]]) -> float:

    ctrl_pts = [measurement[1]]
    ctrl_pts.extend([values.atPoint3(this.keys()[i]) for i in range(3)])
    ctrl_pts = np.concatenate(ctrl_pts, axis=0).reshape(4,3)

    spline = CatmullRomSpline(ctrl_pts)
    pt_w = measurement[0]
    u = pt_w[3]

    est_pt, coeff = spline.get_point(u, return_coeff=True)
    di = spline.get_derivative(u).reshape(3, 1)
    p2p = (pt_w[:3] - est_pt).reshape(3,1)
    proj = np.eye(3) - di @ di.T
    error = proj @ p2p
    error_norm = np.linalg.norm(error)
    if error_norm > 5:
        print('error_norm', error_norm, 'key: ', this.keys()[0], this.keys()[1])

    if jacobians is not None:
        for i in range(3):
            jacobians[i] = -proj * coeff[i+1]
            # print(i, jacobians[i])

    return error

def chord_factor(measurement: float, this: gtsam.CustomFactor,
                      values: gtsam.Values,
                      jacobians: Optional[List[np.ndarray]]) -> float:

    X_i = values.atPoint3(this.keys()[0])
    X_j = values.atPoint3(this.keys()[1])
    D_ij = np.linalg.norm(X_i - X_j)

    error = (D_ij - measurement).reshape(1,1)

    if jacobians is not None:
        jacobians[0] = ((X_i - X_j) / D_ij).reshape(1,3)
        jacobians[1] = ((X_j - X_i) / D_ij).reshape(1,3)

    return error

def chord_factor2(measurement: List, this: gtsam.CustomFactor,
                 values: gtsam.Values,
                 jacobians: Optional[List[np.ndarray]]) -> float:
    X_i = values.atPoint3(this.keys()[0])
    X_j = values.atPoint3(measurement[1])
    D_ij = np.linalg.norm(X_i - X_j)

    error = (D_ij - measurement[0]).reshape(1,1)

    if jacobians is not None:
        jacobians[0] = ((X_i - X_j) / D_ij).reshape(1,3)

    return error

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def PosePointFactor(measurement: Optional[List], this: gtsam.CustomFactor,
                  values: gtsam.Values,
                  jacobians: Optional[List[np.ndarray]]) -> float:

    pt = measurement[0]
    pose:gtsam.Pose3 = values.atPose3(this.keys()[0])
    pt_est = pose.transformFrom(pt)

    u = measurement[1]
    spline = CatmullRomSpline(measurement[2])
    pt_meas, coeff = spline.get_point(u, return_coeff=True)
    error = (pt_est - pt_meas).reshape(3,1)

    if jacobians is not None:
        tmp = np.zeros((3,6))
        tmp[:3,:3] = - pose.rotation().matrix() @ skew(pt)
        tmp[:3,3:] = pose.rotation().matrix()
        jacobians[0] = tmp

    return error

def PosePointTangentFactor(measurement: Optional[List[np.ndarray]], this: gtsam.CustomFactor,
                    values: gtsam.Values,
                    jacobians: Optional[List[np.ndarray]]) -> float:
    pt = measurement[0]
    pose:gtsam.Pose3 = values.atPose3(this.keys()[0])
    pt_est = pose.transformFrom(pt)

    u = measurement[1]
    ctrl_pts = measurement[2]
    spline = CatmullRomSpline(ctrl_pts)
    pt_meas, coeff = spline.get_point(u, return_coeff=True)
    p2p = (pt_est - pt_meas).reshape(3,1)
    # p2p_2 = pose.inverse().transformFrom(pt_meas) - pt
    di = spline.get_derivative(u).reshape(3, 1)
    proj = np.eye(3) - di @ di.T
    error = proj @ p2p
    if cfg.pose_update.reproject_error:
        proj = np.eye(3) #proj residual but not jacobian
    if jacobians is not None:
        tmp = np.zeros((3,6))
        tmp[:3,:3] = - proj @ pose.rotation().matrix() @ skew(pt)
        tmp[:3,3:] = proj @ pose.rotation().matrix()
        jacobians[0] =  tmp
        # np.set_printoptions(precision=3)
        # print(jacobians[0])

    return error