#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk
import numpy as np
from tqdm import tqdm
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(ROOT_DIR)
from misc.utils import Logger
from datetime import datetime
import glob
from lane_slam.system.lane_mapping import LaneMapping
from misc.utils import mkdir_if_missing
from misc.config import define_args
from misc.config import cfg, cfg_from_yaml_file
from multiprocessing import Pool

def run_segments(bag_paths):
    result = {}
    for seq, bag_path in enumerate(tqdm(bag_paths, leave=False, dynamic_ncols=True)):
        lane_mapper = LaneMapping(bag_path)
        stats = lane_mapper.process()
        result[lane_mapper.segment] = stats
    return result

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
        result = {}
        for res in results:
            result.update(res.get())
    else:
        result = run_segments(bag_paths)
    return result

def main(bag_paths):

    result = run_all_segments(bag_paths, multi_process=not cfg.visualization)

    np.save(os.path.join(cfg.output_dir, 'stats.npy'), result)
    all_stats = {}
    for segment, stats in result.items():
        for interval, value in stats.items():
            if interval not in all_stats and type(value) == dict:
                all_stats[interval] = {}
            if type(value) == dict and len(value['error_rot']) > 0:
                all_stats[interval]['error_rot'] = all_stats[interval].get('error_rot', []) + value['error_rot']
                all_stats[interval]['error_rot_raw'] = all_stats[interval].get('error_rot_raw', []) + value['error_rot_raw']
                all_stats[interval]['error_trans'] = all_stats[interval].get('error_trans', []) + value['error_trans']
                all_stats[interval]['error_trans_raw'] = all_stats[interval].get('error_trans_raw', []) + value['error_trans_raw']
            if interval=='map_size':
                all_stats['map_size'] = all_stats.get('map_size', []) + [value]
    print("Map size: ", np.mean(all_stats['map_size']))
    odo_eval_path = os.path.join(cfg.output_dir, 'eval_results', 'odo_eval.txt')
    f = open(odo_eval_path, 'w')
    print("Odometry evaluation:")
    print("Yaw std: {:.3f}".format(cfg.pose_update.odom_noise[2]),
          'trans std: {:.3f}'.format(cfg.pose_update.odom_noise[3]))
    f.write("Odometry evaluation:\n" +
            "Yaw std: {:.3f}".format(cfg.pose_update.odom_noise[2]) +
            "trans std: {:.3f}".format(cfg.pose_update.odom_noise[3]) + '\n')
    for interval, value in all_stats.items():
        if type(value) != dict:
            continue
        output_str = 'interval: ' + str(interval) + \
        ", error rot: {:.3f}/{:.3f}".format(np.mean(value['error_rot']), np.mean(value['error_rot_raw'])) + \
        ', error trans xyz: {:.3f}/{:.3f}'.format(np.mean(value['error_trans']), np.mean(value['error_trans_raw']))
        # print(output_str)
        f.write(output_str + '\n')
    f.close()

    sys.stdout.close()

if __name__ == '__main__':

    args = define_args()
    np.random.seed(666)
    cfg_from_yaml_file(os.path.join(ROOT_DIR, args.cfg_file), cfg)

    print("Load config file: ", args.cfg_file)
    cfg.eval_pose = args.eval_pose
    cfg.visualization = True if args.debug_mode else False
    cfg_name = args.cfg_file.split('/')[-1].split('.')[0]
    cfg.output_dir = os.path.join(cfg.ROOT_DIR, "outputs", cfg_name)
    output_file = os.path.join(cfg.output_dir, 'logs', 'mapping-%s.txt' % datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    mkdir_if_missing(os.path.join(cfg.output_dir, "logs"))
    mkdir_if_missing(os.path.join(cfg.output_dir, 'visualization'))
    mkdir_if_missing(os.path.join(cfg.output_dir, 'results'))
    mkdir_if_missing(os.path.join(cfg.output_dir, 'results_det'))
    mkdir_if_missing(os.path.join(cfg.output_dir, 'eval_results'))
    sys.stdout = Logger(output_file)

    bag_files = sorted(glob.glob(os.path.join(cfg.dataset.dataset_dir, 'lane3d_1000/rosbag/*.bag')))
    main(bag_files)