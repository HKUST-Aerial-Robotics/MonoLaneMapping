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

def main(bag_file):

    result = {}
    # initialize lane mapper
    lane_mapper = LaneMapping(bag_file, save_result=False)
    # process the bag file
    stats = lane_mapper.process()
    result[lane_mapper.segment] = stats

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

    sys.stdout.close()

if __name__ == '__main__':

    args = define_args()
    np.random.seed(666)
    cfg_from_yaml_file(os.path.join(ROOT_DIR, args.cfg_file), cfg)

    print("Load config file: ", args.cfg_file)
    cfg.visualization = True
    cfg.eval_pose = args.eval_pose
    cfg_name = args.cfg_file.split('/')[-1].split('.')[0]
    cfg.output_dir = os.path.join(cfg.ROOT_DIR, "outputs", cfg_name)
    output_file = os.path.join(cfg.output_dir, 'logs', 'mapping-%s.txt' % datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    mkdir_if_missing(os.path.join(cfg.output_dir, "logs"))
    mkdir_if_missing(os.path.join(cfg.output_dir, 'visualization'))
    mkdir_if_missing(os.path.join(cfg.output_dir, 'results'))
    mkdir_if_missing(os.path.join(cfg.output_dir, 'results_det'))
    mkdir_if_missing(os.path.join(cfg.output_dir, 'eval_results'))
    sys.stdout = Logger(output_file)

    bag_file = os.path.join(ROOT_DIR, 'examples/data/segment-14486517341017504003_3406_349_3426_349_with_camera_labels.bag')
    main(bag_file)