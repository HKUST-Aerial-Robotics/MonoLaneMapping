from misc.plot_utils import visualize_points_list
from misc.ros_utils.msg_utils import posemsg_to_np, lanemsg_to_list
from lane_slam.assoc_utils import get_lane_assoc_gt, make_noisy_lane, get_precision_recall, ShellAssoc, \
    KnnASSOC
# from lane_slam.assoc_utils import ClipperAssoc
import rosbag
from misc.lie_utils import get_pose2d_noise
from copy import deepcopy
import numpy as np
from lane_slam.lane_utils import lane_denoise, points_downsample
from misc.utils import AverageMeter
from time import perf_counter
from misc.config import cfg
import open3d as o3d
from lane_slam.lane_feature import LaneFeature

class EvalOneSegment:
    def __init__(self, bag_file):
        self.bag_file = bag_file
        self.bag = rosbag.Bag(bag_file)
        self.pp_ds = cfg.preprocess.downsample # downsample size in pre-processing
        self.frames_data = self.load_data()
        self.step = cfg.lane_asso.fstep_bm

        self.yaw_std = cfg.lane_asso.yaw_std
        self.xyz_std = cfg.lane_mapping.ctrl_noise[0]
        self.trans_std = cfg.lane_asso.trans_std

    def eval_assoc(self):
        # evaluate the association accuracy
        eval_results = []
        for i, (a, b) in enumerate(zip(self.frames_data[:-self.step], self.frames_data[self.step:])):
            # check if the association is correct
            pose_ab = np.dot(np.linalg.inv(a['gt_pose']), b['gt_pose'])
            lane_lm = deepcopy(a['lanes_gt'])
            lane_det = deepcopy(b['lanes_gt'])
            Agt = get_lane_assoc_gt(lane_lm, lane_det)
            lane_det_noisy = make_noisy_lane(lane_det, self.xyz_std)
            pose_ab_noisy = pose_ab @ get_pose2d_noise(self.yaw_std, self.trans_std)
            # if i != 28:
            #     continue
            t1 = perf_counter()
            if cfg.lane_asso.method == 'knn':
                A, stats = self.lane_assoc_knn(lane_lm, lane_det_noisy, pose_ab_noisy)
            elif cfg.lane_asso.method == 'shell':
                A, stats = self.lane_assoc_by_shell(lane_lm, lane_det_noisy, pose_ab_noisy)
            # elif cfg.lane_asso.method == 'clipper':
            #     A, stats = self.lane_assoc_clipper(lane_lm, lane_det_noisy, pose_ab_noisy)
            else:
                raise NotImplementedError
            f1, precision, recall = get_precision_recall(A, Agt)
            # if precision < 0.9:
            #     print('precision: {}, i: {}'.format(precision, i))
            #     self.visualize(lane_lm, lane_det_noisy, pose_ab_noisy, A, Agt, show_correspondences=False)
            #     self.visualize(lane_lm, lane_det_noisy, pose_ab_noisy, A, Agt)
            eval_results.append([f1, precision, recall, perf_counter() - t1])
        return np.array(eval_results)

    def visualize(self, lane_lm, lane_det_noisy, pose_ab_noisy, A=None, Agt=None, show_correspondences=True):
        lane_lm_points = {}
        lane_lm_xyz = []
        for lane in lane_lm:
            xyz = lane['xyz']
            lane_lm_xyz.append(xyz)
            category = lane['category']
            if category not in lane_lm_points:
                lane_lm_points[category] = []
            lane_lm_points[category].append(xyz)
        lm_points_list = []
        for key, value in lane_lm_points.items():
            lm_points_list.append(np.concatenate(value, axis=0))
        lane_det_points = {}
        lane_det_xyz = []
        for lane in lane_det_noisy:
            xyz = lane['xyz']
            category = lane['category']
            xyz = np.dot(pose_ab_noisy, np.concatenate([xyz, np.ones([len(xyz), 1])], axis=1).T).T[:, :3]
            if show_correspondences:
                xyz = xyz + np.asarray([0, 30, 0]).reshape(1, 3)
            lane_det_xyz.append(xyz)
            if category not in lane_det_points:
                lane_det_points[category] = []
            lane_det_points[category].append(xyz)
        det_points_list = []
        for key, value in lane_det_points.items():
            det_points_list.append(np.concatenate(value, axis=0))

        # draw line
        correspondences = None
        if A is not None and show_correspondences:
            correspondences = self.draw_correspondences(lane_lm_xyz, lane_det_xyz, A, Agt)
        if show_correspondences is False:
            lanes_list = []
            lm_points_list = np.concatenate(lm_points_list, axis=0)
            det_points_list = np.concatenate(det_points_list, axis=0)
            lanes_list.append(lm_points_list)
            lanes_list.append(det_points_list)
            visualize_points_list(lanes_list, extra_pcd=correspondences)
        else:
            lm_points_list.extend(det_points_list)
            visualize_points_list(lm_points_list, extra_pcd=correspondences)

    def draw_correspondences(self, lane_lm, lane_det, A, Agt):
        # A: N x 3, Agt: N x 3
        lm_centers = np.zeros([len(lane_lm), 3])
        det_centers = np.zeros([len(lane_det), 3])
        for i, lane in enumerate(lane_lm):
            lm_centers[i] = lane[lane.shape[0]//2, :]
        for i, lane in enumerate(lane_det):
            det_centers[i] = lane[lane.shape[0]//2, :]
        tp = []
        fp = []
        tp_lines = []
        fp_lines = []
        for i in range(len(A)):
            if A[i] in Agt:
                tp.extend([lm_centers[A[i][0]].reshape(1, 3), det_centers[A[i][1]].reshape(1, 3)])
                tp_lines.append([len(tp) - 2, len(tp) - 1])
            else:
                fp.extend([lm_centers[A[i][0]].reshape(1, 3), det_centers[A[i][1]].reshape(1, 3)])
                fp_lines.append([len(fp) - 2, len(fp) - 1])
        tp = np.concatenate(tp, axis=0)
        colors = [[0, 1, 0] for i in range(len(tp_lines))]  # lines are shown in green
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(tp),
            lines=o3d.utility.Vector2iVector(tp_lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        if len(fp) == 0:
            return [line_set]
        fp = np.concatenate(fp, axis=0)
        colors = [[1, 0, 0] for i in range(len(fp_lines))]  # lines are shown in red
        line_set2 = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(fp),
            lines=o3d.utility.Vector2iVector(fp_lines),
        )
        line_set2.colors = o3d.utility.Vector3dVector(colors)
        return [line_set, line_set2]

    def lane_assoc_by_shell(self, lane_lm, lane_det, pose_ab):

        shell_assoc = ShellAssoc()
        shell_assoc.set_ref(lane_lm)
        A = shell_assoc.get_assoc(lane_det, pose_ab)
        return A, None

    # def lane_assoc_clipper(self, lanes_lm, lanes_det, pose_ab):
    #     clipper_assoc = ClipperAssoc()
    #     clipper_assoc.set_landmark(lanes_lm)
    #     clipper_assoc.set_deteciton(lanes_det, pose_ab)
    #     A = clipper_assoc.association()
    #     A_ = []
    #     for i, j in A:
    #         A_.append([i, j])
    #     return A_, None

    def lane_assoc_knn(self, lanes_lm, lanes_det, pose_ab):
        knn_assoc = KnnASSOC()
        lm_features = []
        for lane in lanes_lm:
            lm_features.append(LaneFeature(lane['track_id'], lane['xyz'], lane['category']))
        knn_assoc.set_landmark(lm_features)
        det_features = []
        for lane in lanes_det:
            det_features.append(LaneFeature(lane['track_id'], lane['xyz'], lane['category']))
        knn_assoc.set_deteciton(det_features, pose_ab)
        A, stats = knn_assoc.association()
        return A, stats
        
    def load_data(self):
        frames_data = []
        gt_pose_msg = None
        lanes_gt_msg = None
        lanes_predict_msg = None
        for topic, msg, timestamp in self.bag.read_messages():
            if topic == '/gt_pose_wc':
                gt_pose_msg = msg
            elif topic == '/lanes_gt':
                lanes_gt_msg = msg
            elif topic == '/lanes_predict':
                lanes_predict_msg = msg
            else :
                raise ValueError('Unknown topic: {}'.format(topic))
            if gt_pose_msg is not None and lanes_gt_msg is not None and lanes_predict_msg is not None \
                    and gt_pose_msg.header.stamp == lanes_gt_msg.header.stamp \
                    and gt_pose_msg.header.stamp == lanes_predict_msg.header.stamp:
                gt_pose_wc = posemsg_to_np(gt_pose_msg)
                lanes_gt = lanemsg_to_list(lanes_gt_msg)
                lanes_predict = lanemsg_to_list(lanes_predict_msg)
                lanes_gt = self.preprocess_lanes(lanes_gt)
                lanes_predict = self.preprocess_lanes(lanes_predict)
                frames_data.append({
                    'timestamp': gt_pose_msg.header.stamp.to_sec(),
                    'gt_pose': gt_pose_wc,
                    'lanes_gt': lanes_gt,
                    'lanes_predict': lanes_predict
                })
                gt_pose_msg = None
                lanes_gt_msg = None
                lanes_predict_msg = None

        return frames_data

    def preprocess_lanes(self, lanes):
        for lane in lanes:
            xyz = lane['xyz']
            xyz = points_downsample(xyz, self.pp_ds)
            xyz = lane_denoise(xyz)
            # rearrange points by distance to the origin
            xyz = xyz[np.argsort(np.linalg.norm(xyz, axis=1))]
            lane['xyz'] = xyz
            lane['visibility'] = np.ones(len(xyz))
        return lanes

    def check_prediction(self):
        # check if the prediction's quality is good

        gt_map = []
        predict_map = []
        for frame_data in self.frames_data[::10]:
            lanes_gt = frame_data['lanes_gt']
            lanes_predict = frame_data['lanes_predict']
            lanes_xyz_gt = np.concatenate([lane['xyz'] for lane in lanes_gt], axis=0)
            lanes_xyz_predict = np.concatenate([lane['xyz'] for lane in lanes_predict], axis=0)

            pose = frame_data['gt_pose']
            lanes_xyz_gt = points_downsample(lanes_xyz_gt, 0.5)
            lanes_xyz_gt = pose[:3, :3].dot(lanes_xyz_gt.T).T + pose[:3, 3]
            lanes_xyz_predict = pose[:3, :3].dot(lanes_xyz_predict.T).T + pose[:3, 3]
            gt_map.append(lanes_xyz_gt)
            predict_map.append(lanes_xyz_predict)

        gt_map = points_downsample(np.concatenate(gt_map, axis=0), 0.5)
        predict_map = points_downsample(np.concatenate(predict_map, axis=0), 0.5)
        visualize_points_list([gt_map, predict_map])

        return True