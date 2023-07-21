import json
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
from scipy.interpolate import interp1d

def load_json(pred_lines, pred_dir):
    json_pred = []
    # print("Loading pred json ...")
    for pred_file_path in tqdm(pred_lines, dynamic_ncols=True, leave=False, desc='Loading json'):
        pred_lines = pred_dir + pred_file_path.strip('\n').replace('jpg', 'json')

        with open(pred_lines, 'r') as fp:
            json_pred.append(json.load(fp))
    return json_pred
def mp_load_json(pred_lines, pred_dir, num_workers=40, mp=True):
    if mp is False:
        return load_json(pred_lines, pred_dir)
    pool = Pool(num_workers)
    results = []
    for i in range(num_workers):
        start_idx = i * len(pred_lines) // num_workers
        end_idx = (i + 1) * len(pred_lines) // num_workers
        if i == num_workers - 1:
            end_idx = len(pred_lines)
        results.append(pool.apply_async(load_json, (pred_lines[start_idx:end_idx], pred_dir, )))
    pool.close()
    pool.join()
    result = []
    for res in results:
        result.extend(res.get())
    return result


def prune_3d_lane_by_visibility(lane_3d, visibility):
    lane_3d = lane_3d[visibility > 0, ...]
    return lane_3d

def prune_3d_lane_by_range(lane_3d, x_min, x_max):
    # TODO: solve hard coded range later
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 1] > 0, lane_3d[:, 1] < 200), ...]

    # remove lane points out of x range
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 0] > x_min,
                                     lane_3d[:, 0] < x_max), ...]
    return lane_3d

def prune_3d_lane_by_range(lane_3d, x_min, x_max):
    # TODO: solve hard coded range later
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 1] > 0, lane_3d[:, 1] < 200), ...]

    # remove lane points out of x range
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 0] > x_min,
                                     lane_3d[:, 0] < x_max), ...]
    return lane_3d

def prune_3d_lane_by_box(lane_3d, x_min, x_max, y_min, y_max):
    idx = np.logical_and(lane_3d[:, 0] > x_min, lane_3d[:, 0] < x_max)
    idx = np.logical_and(idx, lane_3d[:, 1] > y_min)
    idx = np.logical_and(idx, lane_3d[:, 1] < y_max)
    lane_3d = lane_3d[idx, ...]
    return lane_3d

def resample_laneline_in_y(input_lane, y_steps, out_vis=False):
    """
        Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
    :param y_steps: a vector of steps in y
    :param out_vis: whether to output visibility indicator which only depends on input y range
    :return:
    """

    # at least two points are included
    assert(input_lane.shape[0] >= 2)

    y_min = np.min(input_lane[:, 1])-5
    y_max = np.max(input_lane[:, 1])+5

    if input_lane.shape[1] < 3:
        input_lane = np.concatenate([input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)], axis=1)

    f_x = interp1d(input_lane[:, 1], input_lane[:, 0], fill_value="extrapolate")
    f_z = interp1d(input_lane[:, 1], input_lane[:, 2], fill_value="extrapolate")

    x_values = f_x(y_steps)
    z_values = f_z(y_steps)

    if out_vis:
        output_visibility = np.logical_and(y_steps >= y_min, y_steps <= y_max)
        return x_values, z_values, output_visibility.astype(np.float32) + 1e-9
    return x_values, z_values


def projection_g2im_extrinsic(E, K):
    E_inv = np.linalg.inv(E)[0:3, :]
    P_g2im = np.matmul(K, E_inv)
    return P_g2im

