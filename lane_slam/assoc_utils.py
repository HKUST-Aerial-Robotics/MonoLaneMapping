import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
import math
from lane_slam.km_matcher import get_km_match
# import clipperpy
from misc.config import cfg
from misc.plot_utils import visualize_points_list
import open3d as o3d

def get_precision_recall(A, Agt):
    if len(Agt) == 0 and len(A) == 0:
        return 1, 1, 1
    if len(A) == 0 or len(Agt) == 0:
        return 0, 0, 0
    num_tp = 0
    for i in range(len(A)):
        if A[i] in Agt:
            num_tp += 1
    precision = num_tp / len(A)
    recall = num_tp / len(Agt)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall

def get_lane_assoc_gt(lane_lm, lane_det):
    Agt = []
    for i in range(len(lane_lm)):
        track_id = lane_lm[i]['track_id']
        for j in range(len(lane_det)):
            if lane_det[j]['track_id'] == track_id:
                Agt.append([i, j])
    return Agt

# waymo camera coordinate system: x forward, y left, z up
def make_noisy_lane(lanes_list, xyz_std):
    # lane: dict
    for lane_dict in lanes_list:
        xyz = lane_dict['xyz']
        xyz_noisy = xyz + np.random.normal(0, xyz_std, xyz.shape)
        lane_dict['xyz'] = xyz_noisy
    return lanes_list

def affinity2assoc(affinity, method='default'):
    # if no association, the affinity matrix is all zeros
    n, m = affinity.shape
    if method == 'km':
        if n > m:
            affinity = np.concatenate([affinity, np.zeros((n, n - m))], axis=1)
        A = get_km_match(affinity)
    else:
        A = []
        # for i in range(n):
        #     j = np.argmax(affinity[i, :])
        #     if affinity[i, j] > 0:
        #         A.append([i, j])
        for det_id in range(m):
            lane_id = np.argmax(affinity[:, det_id])
            if affinity[lane_id, det_id] > 0:
                A.append([lane_id, det_id])
    return A

def get_dist_thd(xyz, yaw_std, trans_std, xyz_std, dim):
    # xyz: np.array (N, 3) or (N, 2)
    # pt_range = np.max(np.linalg.norm(xyz, axis=1)) # a little bit better
    pt_range = np.linalg.norm(xyz, axis=1)
    yaw_thd = 2 * yaw_std #degree
    upper_thd_R = 2 * pt_range * np.sin(yaw_thd / 180 * np.pi / 2)
    upper_thd_t = trans_std * 2 # assume the range error
    upper_thd_xyz = xyz_std * 2 # assume the range error
    upper_thd = upper_thd_R + upper_thd_t + upper_thd_xyz
    upper_thd = np.maximum(upper_thd, 1.0)
    return upper_thd

def road_direction(lanes):
    points = []
    for lane in lanes:
        points.append(lane)
    points = np.concatenate(points, axis=0)
    ds_ratio = points.shape[0] // 100
    if ds_ratio > 1:
        points = points[::ds_ratio, :]
    else:
        if points.shape[0] < 10:
            return None
        points = points
    #principal direction
    cov = np.cov(points, rowvar=False)
    eig_val, eig_vec = np.linalg.eig(cov)
    idx = np.argsort(eig_val)
    eig_vec = eig_vec[:, idx]
    return eig_vec[:, -1]

def get_left_right_id(xyz_list):
    road_d = road_direction(xyz_list)
    if road_d is None:
        return None, None
    if road_d.shape[0] == 2:
        road_d = np.hstack([road_d, np.array([0])])
    expected_axis = np.array([1, 0, 0])
    angle = np.arccos(np.dot(road_d, expected_axis) / (np.linalg.norm(road_d) * np.linalg.norm(expected_axis)))
    rot = R.from_rotvec(angle * np.array([0, 0, 1]))
    visualize_points_list(xyz_list)
    xyz_list_new = [rot.apply(xyz) for xyz in xyz_list]
    visualize_points_list(xyz_list_new)
    road_d_new = road_direction(xyz_list_new)
    median = []
    for i in range(len(xyz_list)):
        mid = xyz_list[i][xyz_list[i].shape[0]//2, :]
        if mid.shape[0] == 2:
            mid = np.hstack([mid, np.array([0])])
        mid = rot.apply(mid)
        median.append(mid[1])
    id = [i for i in range(len(median))]
    argsort_ = np.argsort(median)
    for i, idx in enumerate(argsort_):
        id[idx] = i

    return id, median

def draw_lines(pts1, pts2, color = [0, 1, 0]):
    # pts1: N x 3, pts2: N x 3
    line_cp = []
    for i in range(pts1.shape[0]):
        line_cp.append([i, i + pts1.shape[0]])
    points = np.concatenate([pts1, pts2], axis=0)
    colors = [color for i in range(len(line_cp))]  # lines are shown in green
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(line_cp),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def draw_correspondences(lane_lm, lane_det, A, Agt):
    if Agt is None:
        Agt = A
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

    if len(tp) > 0:
        tp = np.concatenate(tp, axis=0)
        colors = [[0, 1, 0] for i in range(len(tp_lines))]  # lines are shown in green
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(tp),
            lines=o3d.utility.Vector2iVector(tp_lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
    if len(fp) > 0:
        fp = np.concatenate(fp, axis=0)
        colors = [[1, 0, 0] for i in range(len(fp_lines))]  # lines are shown in red
        line_set2 = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(fp),
            lines=o3d.utility.Vector2iVector(fp_lines),
        )
        line_set2.colors = o3d.utility.Vector3dVector(colors)
    if len(tp) > 0 and len(fp) > 0:
        return [line_set, line_set2]
    elif len(tp) > 0:
        return [line_set]
    elif len(fp) > 0:
        return [line_set2]
    else:
        return []

def point_to_line(pt1, line_pt1, line_pt2):
    # compute line equation
    if line_pt1[0] == line_pt2[0]:
        k = 1e10
    else:
        k = (line_pt1[1] - line_pt2[1]) / (line_pt1[0] - line_pt2[0])
    b = line_pt1[1] - k * line_pt1[0]
    tmp = k * pt1[0] - pt1[1] + b
    dist = abs(tmp) / np.sqrt(k**2 + 1)
    left = tmp > 0
    return left, dist

def left_or_right(lane_a, lane_b):
    # lane_a: N x 3, lane_b: M x 3
    anchor_a = lane_a[lane_a.shape[0]//2, :2]
    point_on_b1 = lane_b[0, :2]
    point_on_b2 = lane_b[lane_b.shape[0]-1, :2]
    left, dist = point_to_line(anchor_a, point_on_b1, point_on_b2)
    return left, dist

def row_col_max(A):
    A_row = np.ones_like(A) * False
    for i in range(A.shape[0]):
        max_val = np.max(A[i])
        for j in range(A.shape[1]):
            if A[i][j] == max_val and max_val != 0:
                A_row[i][j] = True
    A_col = np.ones_like(A) * False
    for i in range(A.shape[1]):
        max_val = np.max(A[:, i])
        for j in range(A.shape[0]):
            if A[j][i] == max_val and max_val != 0:
                A_col[j][i] = True
    A_row_col = A_row + A_col
    return A_row_col

class KnnASSOC:
    def __init__(self):
        
        self.yaw_std = cfg.lane_asso.yaw_std
        self.xyz_std = cfg.lane_mapping.ctrl_noise[0]
        self.trans_std = cfg.lane_asso.trans_std
        self.dim = cfg.preprocess.dim
        self.knn_type = cfg.knn.knn_type
        self.min_match_ratio = cfg.knn.min_match_ratio
        self.use_consistency = cfg.knn.use_consistency
        
        self.num_lm = 0
        self.lm_xyz = []
        self.lm_kdtrees = []
        self.lm_categories = []
        
        self.num_det = 0
        self.det_xyz = []
        self.det_feature = []
        self.det_knn_thd = []
        self.det_categories = []
    
    def set_landmark(self, lanes_lm):
        self.lm_kdtrees = []
        self.lm_categories = []
        self.lm_xyz = []
        for i in range(len(lanes_lm)):
            xyz = lanes_lm[i].points[:, :self.dim]
            self.lm_xyz.append(xyz)
            xyz = xyz if self.knn_type == 'xyz' else self.construct_lmr(xyz)
            if lanes_lm[i].kdtree is None:
                self.lm_kdtrees.append(KDTree(xyz))
            else:
                self.lm_kdtrees.append(lanes_lm[i].kdtree)
            self.lm_categories.append(lanes_lm[i].category)
        self.num_lm = len(lanes_lm)
            
    def set_deteciton(self, lanes_det, pose_ab):
        self.det_xyz = []
        self.det_feature = []
        self.det_knn_thd = []
        self.det_categories = []
        for i in range(len(lanes_det)):
            xyz = lanes_det[i].points[:, :self.dim]
            distance_thd = get_dist_thd(xyz, self.yaw_std, self.trans_std, self.xyz_std, self.dim)
            xyz = self.transform_xyz(xyz, pose_ab)
            xyz = xyz if self.knn_type == 'xyz' else self.construct_lmr(xyz)
            self.det_xyz.append(xyz[:, :self.dim])
            self.det_feature.append(xyz)
            self.det_knn_thd.append(distance_thd)
            self.det_categories.append(lanes_det[i].category)
        self.num_det = len(lanes_det)

    def transform_xyz(self, xyz, pose_ab):
        if self.dim == 3:
            xyz = np.dot(pose_ab[:3, :3], xyz.T).T + pose_ab[:3, 3]
        else:
            xyz = np.dot(pose_ab[:2, :2], xyz.T).T + pose_ab[:2, 3]
        return xyz
    
    def association(self):

        if self.num_lm == 0 or self.num_det == 0:
            return [], {}

        assoc_scores = np.zeros((self.num_lm, self.num_det))
        assoc_dist = np.zeros((self.num_lm, self.num_det))
        assoc_dist_thd = np.zeros((self.num_lm, self.num_det))
        for i in range(self.num_det):
            xyz = self.det_feature[i]
            distance_thd = self.det_knn_thd[i]
            dist_candidates = []
            for j in range(self.num_lm):
                if self.lm_categories[j] != self.det_categories[i]:
                    continue
                dist, idx = self.lm_kdtrees[j].query(xyz, k=1)
                if self.knn_type == 'lmr':
                    dist = np.linalg.norm(xyz[:, :self.dim] - self.lm_kdtrees[j].data[idx, :self.dim], axis=1)
                dist_candidates.append(dist.reshape(-1, 1))
                dist_match = dist[dist < distance_thd]
                if len(dist_match) == 0:
                    continue
                score = np.mean(dist_match) * np.sqrt(len(dist) / len(dist_match))
                ideal_dist = np.mean(distance_thd) * np.sqrt(1 / self.min_match_ratio)
                assoc_dist[j, i] = score
                assoc_dist_thd[j, i] = ideal_dist
                assoc_scores[j, i] = 1 / score if score < ideal_dist else 0

        if self.use_consistency and self.num_lm > 2 and self.num_det > 2:
            C = self.construct_consistency(assoc_scores)
        else:
            C = np.ones((self.num_lm, self.num_det))
        assoc_scores_w = assoc_scores * C
        A = affinity2assoc(assoc_scores_w, method='km')

        stats = {
            'assoc_scores': assoc_scores,
            'assoc_dist': assoc_dist,
            'assoc_scores_w': assoc_scores_w,
            'assoc_dist_thd': assoc_dist_thd,
            'A': A,
            'C': C,
        }

        return A, stats

    def construct_consistency(self, affinity):
        A_consistency = np.zeros((self.num_lm, self.num_det))
        A = []
        for i in range(self.num_lm):
            for j in range(self.num_det):
                if affinity[i, j] > 0:
                    A.append([i, j])
        w = 1
        for aij in A:
            i, j = aij
            for bkl in A:
                k, l = bkl
                if i == k or j == l:
                    A_consistency[i, j] += w
                else:
                    a, dist_valid_a = left_or_right(self.lm_xyz[i][-self.det_xyz[j].shape[0]:, :self.dim],
                                                    self.lm_xyz[k][-self.det_xyz[l].shape[0]:, :self.dim])
                    b, dist_valid_b = left_or_right(self.det_xyz[j], self.det_xyz[l])
                    if a == b:
                        A_consistency[i, j] += w / (1 + np.abs(dist_valid_a - dist_valid_b))
        if A_consistency.max() > 0:
            # A_consistency = row_col_max(A_consistency)
            A_consistency = pow(A_consistency / A_consistency.max(), 2)

        else:
            A_consistency = np.ones((self.num_lm, self.num_det))

        return A_consistency
    def k2ij(self, k, n):
        k += 1

        l = n * (n-1) / 2 - k
        o = np.floor( (np.sqrt(1 + 8*l) - 1) / 2. )
        p = l - o * (o + 1) / 2
        i = n - (o + 1)
        j = n - p

        return int(i-1), int(j-1)

    def construct_lmr(self, xyz, weight = 5.0):
        # for every lane, cost 1.5ms
        num = xyz.shape[0]
        delta_angle = np.zeros((num, 1))
        for i in range(num):
            if i == 0 or i == num - 1:
                delta_angle[i] = 0
                continue
            v1 = xyz[i, :self.dim] - xyz[i-1, :self.dim]
            v2 = xyz[i+1, :self.dim] - xyz[i, :self.dim]
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            delta_angle[i] = np.arccos(np.clip(cos_theta, -1, 1))
        xyz_lmr = np.concatenate([xyz[:, :self.dim], delta_angle * weight], axis=1)
        return xyz_lmr

# class LaneInvariant(clipperpy.invariants.PairwiseInvariant):
#     def __init__(self,
#                  dim = 3):
#         clipperpy.invariants.PairwiseInvariant.__init__(self)
#
#         self.dist_std = cfg.clipper.dist_std
#         self.dim = cfg.preprocess.dim
#         self.ϵ = cfg.clipper.noise_bound
#         self.weighted = cfg.clipper.weighted
#
#     def median_dist(self, ai, aj):
#         lane_ai = ai['xyz']
#         lane_aj = aj['xyz']
#         num_ai = lane_ai.shape[0]
#         num_aj = lane_aj.shape[0]
#         # dist = np.linalg.norm(lane_ai[num_ai//2, :self.dim] - lane_aj[num_aj//2, :self.dim])
#         dist = np.linalg.norm(np.mean(lane_ai, axis=0) - np.mean(lane_aj, axis=0))
#         return dist
#
#     def __call__(self, ai, aj, bi, bj):
#         if ai['category'] != bi['category'] or aj['category'] != bj['category']:
#             return 0.0
#         l1 = self.median_dist(ai, aj)
#         l2 = self.median_dist(bi, bj)
#         c = np.abs(l1 - l2)
#         if self.weighted:
#             prob = np.exp(-0.5*c*c/(self.dist_std**2))
#         else:
#             prob = 1
#         return prob if c < self.ϵ else 0.0

# class ClipperAssoc:
#     def __init__(self):
#         self.invariant = LaneInvariant()
#         self.params = clipperpy.Params()
#         self.clipper = clipperpy.CLIPPER(self.invariant, self.params)
#         self.dim = cfg.preprocess.dim
#         self.lands_lm = []
#         self.lanes_det = []
#
#     def association(self):
#         M, C, A = self.score_pairwise_consistency(self.invariant, self.lands_lm, self.lanes_det)
#         self.clipper.set_matrix_data(M, C, A)
#         self.clipper.solve()
#         Ain = self.clipper.get_selected_associations()
#         return Ain
#
#     def score_pairwise_consistency(self, invariant, D1, D2, A = None):
#         if A is None:
#             A = clipperpy.utils.create_all_to_all(len(D1), len(D2))
#
#         m = A.shape[0]
#
#         M = np.eye(m)
#         C = np.ones((m,m))
#
#         for k in range(int(m*(m-1)/2)):
#             i, j = self.k2ij(k, m) # or clipperpy.utils.k2ij
#
#             if A[i,0] == A[j,0] or A[i,1] == A[j,1]:
#                 C[i,j] = C[j,i] = 0
#                 continue
#             d1i = D1[A[i,0]]
#             d1j = D1[A[j,0]]
#
#             d2i = D2[A[i,1]]
#             d2j = D2[A[j,1]]
#
#             scr = invariant(d1i,d1j,d2i,d2j)
#
#             if scr > 0:
#                 M[i,j] = M[j,i] = scr
#             else:
#                 C[i,j] = C[j,i] = 0
#
#         return M, C, A
#
#     def k2ij(self, k, n):
#         k += 1
#
#         l = n * (n-1) / 2 - k
#         o = np.floor( (np.sqrt(1 + 8*l) - 1) / 2. )
#         p = l - o * (o + 1) / 2
#         i = n - (o + 1)
#         j = n - p
#
#         return int(i-1), int(j-1)
#
#     def set_landmark(self, lanes_lm):
#         self.lands_lm = lanes_lm
#         for i in range(len(self.lands_lm)):
#             # self.lands_lm[i]['kdtree'] = KDTree(self.lands_lm[i]['xyz'][:, :self.dim])
#             self.lands_lm[i]['xyz'] = self.lands_lm[i]['xyz'][:, :self.dim]
#
#     def set_deteciton(self, lanes_det, pose_ab):
#         self.lanes_det = lanes_det
#         for i in range(len(self.lanes_det)):
#             xyz = self.lanes_det[i]['xyz'][:, :self.dim]
#             xyz = self.transform_xyz(xyz, pose_ab)
#             self.lanes_det[i]['xyz'] = xyz
#             # self.lanes_det[i]['kdtree'] = KDTree(self.lanes_det[i]['xyz'][:, :self.dim])
#
#     def transform_xyz(self, xyz, pose_ab):
#         if self.dim == 3:
#             xyz = np.dot(pose_ab[:3, :3], xyz.T).T + pose_ab[:3, 3]
#         else:
#             xyz = np.dot(pose_ab[:2, :2], xyz.T).T + pose_ab[:2, 2]
#         return xyz

class ShellAssoc:
    def __init__(self):
        
        self.yaw_std = cfg.lane_asso.yaw_std
        self.xyz_std = cfg.lane_mapping.ctrl_noise[0]
        self.trans_std = cfg.lane_asso.trans_std
        self.shell_R = cfg.shell.radius
        self.min_match_ratio = cfg.shell.min_match_ratio
        self.max_range = 100
        self.dim = 2
        self.max_num = math.ceil(self.max_range / self.shell_R)

        self.ref_categeories = []
        self.ref_shell = []
        self.ref_cemtroid = []

    def set_ref(self, lanes_ref):
        self.ref_shell = []
        for i in range(len(lanes_ref)):
            xyz = lanes_ref[i]['xyz']
            self.ref_categeories.append(lanes_ref[i]['category'])

            shells = [[] for _ in range(self.max_num * 4)]
            centroid_xy = np.mean(xyz, axis=0)[:2]
            xy = xyz[:, :2] - centroid_xy
            for j in range(len(xyz)):
                shell_id = self.get_shell_id(xy[j])
                if shell_id < 0:
                    continue
                shells[shell_id].append(xyz[j])
            self.ref_shell.append(shells)
            self.ref_cemtroid.append(centroid_xy)

    def get_shell_id(self, xy):
        x, y = xy
        quadrant = 0 # 0: bottom left, 1: bottom right, 2: top left, 3: top right
        quadrant = quadrant + 1 if x > 0 else quadrant
        quadrant = quadrant + 2 if y > 0 else quadrant
        range_xy = np.linalg.norm(xy)
        if range_xy > self.max_range:
            return -1
        else:
            return quadrant * self.max_num + int(range_xy / self.shell_R)

    def get_assoc(self, lane_comp, pose_ab):
        num_ref = len(self.ref_shell)
        num_comp = len(lane_comp)
        if num_ref == 0 or num_comp == 0:
            return []
        assoc_scores = np.ones((num_ref, num_comp)) * np.inf
        for comp_id in range(num_comp):
            xyz = lane_comp[comp_id]['xyz']
            xyz = np.dot(pose_ab[:3, :3], xyz.T).T + pose_ab[:3, 3]
            distance_thd = get_dist_thd(xyz, self.yaw_std, self.trans_std, self.xyz_std, self.dim)
            for ref_id in range(num_ref):
                if self.ref_categeories[ref_id] != lane_comp[comp_id]['category']:
                    continue

                dis = 0;n_ab = 0;n_b = 0
                centroid_xy = self.ref_cemtroid[ref_id]
                xy = xyz[:, :2] - centroid_xy
                for j in range(len(xyz)):
                    shell_id = self.get_shell_id(xy[j])
                    if shell_id < 0:
                        n_b += 1
                        continue
                    ref_shell = np.array(self.ref_shell[ref_id][shell_id])
                    if len(ref_shell) == 0:
                        n_b += 1
                        continue
                    min_dist = np.min(np.linalg.norm(ref_shell - xyz[j], axis=1))
                    dis += min_dist
                    n_ab += 1
                if n_ab == 0:
                    assoc_scores[ref_id, comp_id] = np.inf
                else:
                    assoc_scores[ref_id, comp_id] = dis / n_ab * math.sqrt((n_ab + n_b) / n_ab)
                    ideal_dist = np.mean(distance_thd) / self.min_match_ratio
                    if assoc_scores[ref_id, comp_id] > ideal_dist:
                        assoc_scores[ref_id, comp_id] = np.inf

        A = []
        for i in range(num_ref):
            j = np.argmin(assoc_scores[i, :])
            if assoc_scores[i, j] < np.inf:
                A.append([i, j])

        return A

    def min_dist_in_shell(self, xyz, ref_shell):
        xyz = xyz[:, :2]
        ref_shell = ref_shell[:, :2]
        dist = np.linalg.norm(xyz[:, None] - ref_shell[None], axis=2)
        return np.min(dist, axis=1)


class DASAC:
    def __init__(self,
                 max_iter_num=100,
                 end_thd=2.0):

        self.yaw_std = cfg.lane_asso.yaw_std
        self.xyz_std = cfg.lane_mapping.ctrl_noise[0]
        self.trans_std = cfg.lane_asso.trans_std
        self.max_iter_num = max_iter_num
        self.end_thd = end_thd
        self.dim = cfg.preprocess.dim

        self.lm_lmrs = None
        self.lm_kdtree = None
        self.det_lmrs = None


    def construct_lmr(self, xyz):
        num = xyz.shape[0]
        delta_angle = np.zeros(num, 1)
        for i in range(num):
            if i == 0 or i == num - 1:
                delta_angle[i] = 0
                continue
            v1 = xyz[i, :self.dim] - xyz[i-1, :self.dim]
            v2 = xyz[i+1, :self.dim] - xyz[i, :self.dim]
            delta_angle[i] = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        xyz_lmr = np.concatenate([xyz[:, :self.dim], delta_angle], axis=1)
        return xyz_lmr

    def set_landmark(self, lanes_lm):
        num = len(lanes_lm)
        lanes_lmrs = []
        for i in range(num):
            xyz = lanes_lm[i]['xyz']
            xyz_lmr = self.construct_lmr(xyz)
            category = np.ones(xyz.shape[0]) * lanes_lm[i]['category']
            lanes_lmrs.append(np.concatenate([xyz_lmr, category[:, None]], axis=1))
        self.lm_lmrs = np.concatenate(lanes_lmrs, axis=0)
        self.lm_kdtree = KDTree(self.lm_lmrs[:, :self.dim])

    def set_dection(self, lanes_det, pose_ab):
        num = len(lanes_det)
        lanes_lmrs = []
        for comp_id in range(num):
            xyz = lanes_det[comp_id]['xyz']
            xyz = np.dot(pose_ab[:3, :3], xyz.T).T + pose_ab[:3, 3]
            xyz_lmr = self.construct_lmr(xyz)
            category = np.ones(xyz.shape[0]) * lanes_det[comp_id]['category']
            lanes_lmrs.append(np.concatenate([xyz_lmr, category[:, None]], axis=1))
        self.det_lmrs = np.concatenate(lanes_lmrs, axis=0)