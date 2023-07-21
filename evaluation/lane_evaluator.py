from evaluation.eval_utils import mp_load_json
import json
from tqdm import tqdm
from evaluation.lane3d import Lane3D, fitting
from lane_slam.lane_utils import prune_3d_lane_by_range
import numpy as np
from misc.config import cfg
class LaneEval(object):
    def __init__(self, dataset_dir, pred_dir):
        self.dataset_dir = dataset_dir
        self.pred_dir = pred_dir
        self.dataset_name = "openlane"
        # self.top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
        self.top_view_region = np.array([[-10, 50], [10, 50], [-10, 3], [10, 3]])
        self.x_min = self.top_view_region[0, 0]
        self.x_max = self.top_view_region[1, 0]
        self.y_min = self.top_view_region[2, 1]
        self.y_max = self.top_view_region[0, 1]
        self.y_samples = np.linspace(self.y_min, self.y_max, num=100, endpoint=False)
        self.close_range = 40

    def bench_one_submit(self, pred_dir, gt_dir, test_txt, prob_th=0.5, vis=False):
        pred_lines = open(test_txt).readlines()
        gt_lines = pred_lines

        json_pred = mp_load_json(pred_lines, pred_dir, mp=True)
        json_gt = mp_load_json(gt_lines, gt_dir, mp=True)
        gts = {l['file_path']: l for l in json_gt}

        laneline_stats = []
        laneline_error = []
        for _, pred in enumerate(tqdm(json_pred, dynamic_ncols=True, leave=False, desc='Evaluating')):
            if 'file_path' not in pred or 'lane_lines' not in pred:
                raise Exception('file_path or lane_lines not in some predictions.')
            raw_file = pred['file_path']
            pred_lanelines = pred['lane_lines']
            pred_lanes = [np.array(lane['xyz']) for _, lane in enumerate(pred_lanelines)]
            pred_category = [int(lane['category']) for _, lane in enumerate(pred_lanelines)]

            if raw_file not in gts:
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]

            # evaluate lanelines
            cam_extrinsics = np.array(gt['extrinsic'])
            # Re-calculate extrinsic matrix based on ground coordinate
            R_vg = np.array([[0, 1, 0],
                             [-1, 0, 0],
                             [0, 0, 1]], dtype=float)
            R_gc = np.array([[1, 0, 0],
                             [0, 0, 1],
                             [0, -1, 0]], dtype=float)
            cam_extrinsics[:3, :3] = np.matmul(np.matmul(
                np.matmul(np.linalg.inv(R_vg), cam_extrinsics[:3, :3]),
                R_vg), R_gc)

            cam_extrinsics[0:2, 3] = 0.0

            cam_intrinsics = gt['intrinsic']

            try:
                gt_lanes_packed = gt['lane_lines']
            except:
                print("error 'lane_lines' in gt: ", gt['file_path'])

            gt_lanes, gt_visibility, gt_category = [], [], []
            for j, gt_lane_packed in enumerate(gt_lanes_packed):
                # A GT lane can be either 2D or 3D
                # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
                lane = np.array(gt_lane_packed['xyz'])
                lane_visibility = np.array(gt_lane_packed['visibility'])

                lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
                cam_representation = np.linalg.inv(
                    np.array([[0, 0, 1, 0],
                              [-1, 0, 0, 0],
                              [0, -1, 0, 0],
                              [0, 0, 0, 1]], dtype=float))
                lane = np.matmul(cam_extrinsics, np.matmul(cam_representation, lane))
                lane = lane[0:3, :].T

                gt_lanes.append(lane)
                gt_visibility.append(lane_visibility)
                gt_category.append(gt_lane_packed['category'])

            # N to N matching of lanelines
            tp, c_lane, cnt_gt, cnt_pred, xyz_error = self.bench(pred_lanes,
                                                                 pred_category,
                                                                 gt_lanes,
                                                                 gt_category)
            laneline_stats.append(np.array([tp, c_lane, cnt_gt, cnt_pred]))
            # consider x_error z_error only for the matched lanes
            laneline_error.extend(xyz_error)

        output_stats = []
        laneline_stats = np.array(laneline_stats)
        laneline_error = np.array(laneline_error)

        if np.sum(laneline_stats[:, 2]) != 0:
            R_lane = np.sum(laneline_stats[:, 0]) / (np.sum(laneline_stats[:, 2]))
        else:
            R_lane = np.sum(laneline_stats[:, 0]) / (np.sum(laneline_stats[:, 2]) + 1e-6)  # recall = TP / (TP+FN)
        if np.sum(laneline_stats[:, 3]) != 0:
            P_lane = np.sum(laneline_stats[:, 0]) / (np.sum(laneline_stats[:, 3]))
        else:
            P_lane = np.sum(laneline_stats[:, 0]) / (np.sum(laneline_stats[:, 3]) + 1e-6)  # precision = TP / (TP+FP)
        if np.sum(laneline_stats[:, 0]) != 0:
            C_lane = np.sum(laneline_stats[:, 1]) / (np.sum(laneline_stats[:, 0]))
        else:
            C_lane = np.sum(laneline_stats[:, 1]) / (np.sum(laneline_stats[:, 0]) + 1e-6)  # category_accuracy
        if R_lane + P_lane != 0:
            F_lane = 2 * R_lane * P_lane / (R_lane + P_lane)
        else:
            F_lane = 2 * R_lane * P_lane / (R_lane + P_lane + 1e-6)
        if laneline_error.shape[0] != 0:
            xyz_error = np.average(laneline_error, axis=0)
        else:
            xyz_error = 0.0

        output_stats.append(F_lane)
        output_stats.append(R_lane)
        output_stats.append(P_lane)
        output_stats.append(C_lane)
        output_stats.append(xyz_error)

        return output_stats

    def bench(self, pred_lanes, pred_category, gt_lanes, gt_category):
        # change this properly
        lanes3d_gt = []
        for lane, category in zip(gt_lanes, gt_category):
            xyz = prune_3d_lane_by_range(lane, (self.x_min, self.x_max, self.y_min, self.y_max))
            if xyz.shape[0] > 0:
                xyz = fitting(xyz)
                lanes3d_gt.append(Lane3D(xyz, category))
        lanes3d_pred = []
        for lane, category in zip(pred_lanes, pred_category):
            xyz = prune_3d_lane_by_range(lane, (self.x_min - 10, self.x_max + 10, self.y_min, self.y_max + 20))
            if xyz.shape[0] > 0:
                xyz = fitting(xyz)
                xyz = prune_3d_lane_by_range(xyz, (self.x_min, self.x_max, self.y_min, self.y_max))
            if xyz.shape[0] > 0:
                lanes3d_pred.append(Lane3D(xyz, category))

        TP = 0
        dists = []
        category_match = 0
        for lane3d_pred in lanes3d_pred:
            for lane3d_gt in lanes3d_gt:
                succ, error_list = lane3d_pred.similarity(lane3d_gt, cfg.evaluation.overlap_thd, cfg.evaluation.dist_thd)
                if succ:
                    TP += 1
                    dists.extend(error_list)
                    if lane3d_pred.category == lane3d_gt.category or (lane3d_pred.category==20 and lane3d_gt.category==21):
                        category_match += 1
                    break
        return TP, category_match, len(lanes3d_gt), len(lanes3d_pred), dists