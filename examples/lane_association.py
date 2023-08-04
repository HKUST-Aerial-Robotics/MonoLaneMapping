#!/usr/bin/env python
# lane assocaitation benchmark
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk

import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(ROOT_DIR)
from misc.utils import Logger
import numpy as np
from datetime import datetime
from tqdm import tqdm
from lane_slam.eval_assoc import EvalOneSegment
from misc.config import define_args
from misc.config import cfg, cfg_from_yaml_file, log_config_to_file
from multiprocessing import Pool
import glob

def run_segments(bag_paths):
    eval_stats_list = []
    for rosbag_file in tqdm(bag_paths, leave=False, dynamic_ncols=True):
        evaluator = EvalOneSegment(rosbag_file)
        eval_stats = evaluator.eval_assoc()
        if eval_stats.shape[0] == 0:
            continue
        eval_stats_list.append(eval_stats)
    return eval_stats_list

def run_all_segments(bag_paths, multi_process=False, num_workers=20):
    if multi_process:
        pool = Pool(num_workers)
        results = []
        for i in range(num_workers):
            start_idx = i * len(bag_paths) // num_workers
            end_idx = (i + 1) * len(bag_paths) // num_workers
            if i == num_workers - 1:
                end_idx = len(bag_paths)
            results.append(pool.apply_async(run_segments, (bag_paths[start_idx:end_idx], )))
        pool.close()
        pool.join()
        result = []
        for res in results:
            result.extend(res.get())
    else:
        result = run_segments(bag_paths)
    return result

def main(output_file):
    openlane_dir = cfg.dataset.dataset_dir
    sys.stdout = Logger(output_file)
    log_config_to_file(cfg)

    rosbag_dir = os.path.join(openlane_dir, 'lane3d_1000', 'rosbag')
    rosbag_flies = glob.glob(os.path.join(rosbag_dir, '*.bag'))
    eval_stats_list = run_all_segments(rosbag_flies, multi_process=True, num_workers=20)
    eval_stats_list = np.concatenate(eval_stats_list, axis=0)
    result_str = 'f1 score: {:.4f}, precision: {:.4f}, recall: {:.4f}, time: {:.4f} ms'. \
        format(np.mean(eval_stats_list[:, 0]),
               np.mean(eval_stats_list[:, 1]),
               np.mean(eval_stats_list[:, 2]),
               np.mean(eval_stats_list[:, 3]*1000))
    print(result_str)
    sys.stdout.close()

# def clipper_bm():
#     cfg.lane_asso.method = 'clipper'
#     for min_match_ratio in [0.5]:
#         cfg.clipper.min_match_ratio = min_match_ratio
#         for weighted in [False]:
#             cfg.clipper.weighted = weighted
#             for noise_bound in [5]:
#                 cfg.clipper.noise_bound = noise_bound
#                 output_dir = cfg.output_dir
#                 output_file = os.path.join(output_dir, 'eval-clipper-m{}-w{}-n{}.txt'.
#                                            format(min_match_ratio, weighted, noise_bound))
#                 main(output_file)

def knn_bm():
    cfg.lane_asso.method = 'knn'
    for knn_type in ['xyz', 'lmr']:
        cfg.knn.knn_type = knn_type
        for use_consistency in [True, False]:
            cfg.knn.use_consistency = use_consistency
            for min_match_ratio in [0.5]:
                cfg.knn.min_match_ratio = min_match_ratio
                output_dir = cfg.output_dir
                output_file = os.path.join(output_dir, 'eval-knn-{}-c{}-m{}.txt'.
                                           format(knn_type, use_consistency, min_match_ratio))
                main(output_file)

def shell_bm():
    cfg.lane_asso.method = 'shell'
    for min_match_ratio in [0.5]:
        cfg.shell.min_match_ratio = min_match_ratio
        for radius in [1.5]:
            cfg.shell.radius = radius
            output_dir = cfg.output_dir
            output_file = os.path.join(output_dir, 'eval-shell-m{}-r{}.txt'.
                                       format(min_match_ratio, radius))
            main(output_file)

def benchmark():

    shell_bm()
    # clipper_bm()
    knn_bm()


if __name__ == '__main__':

    args = define_args()
    cfg_from_yaml_file(os.path.join(ROOT_DIR, args.cfg_file), cfg)
    np.random.seed(666)
    cfg_name = args.cfg_file.split('/')[-1].split('.')[0]
    cfg.output_dir = os.path.join(cfg.ROOT_DIR, "outputs", cfg_name)
    if args.bm:
        benchmark()
    else:
        output_dir = cfg.output_dir
        output_file = os.path.join(output_dir, 'eval-%s.txt' % datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        main(output_file)