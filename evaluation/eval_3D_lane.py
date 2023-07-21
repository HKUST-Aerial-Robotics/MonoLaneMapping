# ==============================================================================
# Binaries and/or source for the following packages or projects are presented under one or more of the following open
# source licenses:
# eval_3D_lane.py       The OpenLane Dataset Authors        Apache License, Version 2.0
#
# Contact simachonghao@pjlab.org.cn if you have any issue
# 
# See:
# https://github.com/yuliangguo/Pytorch_Generalized_3D_Lane_Detection/blob/master/tools/eval_3D_lane.py
#
# Copyright (c) 2022 The OpenLane Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Description: This code is to evaluate 3D lane detection. The optimal matching between ground-truth set and predicted 
    set of lanes are sought via solving a min cost flow.

Evaluation metrics includes:
    F-scores
    x error close (0 - 40 m)
    x error far (0 - 100 m)
    z error close (0 - 40 m)
    z error far (0 - 100 m)
"""
import os
from misc.utils import mkdir_if_missing
from misc.config import cfg
from glob import glob
from evaluation.lane_evaluator import LaneEval
# from eval.lane_evaluator_openlane import LaneEval

def eval_lane3d(dataset_dir, pred_dir, test_list, output_dir, verbose=False):

    # Initialize evaluator
    evaluator = LaneEval(dataset_dir, pred_dir)

    # Evaluation
    eval_stats = evaluator.bench_one_submit(pred_dir, dataset_dir, test_list, prob_th=0.5)

    # Print evaluation results
    output_file = os.path.join(output_dir, 'eval_result.txt')
    output_str = "===> Evaluation: " + \
                    "laneline F-measure {:.3}, ".format(eval_stats[0]) + \
                    "laneline Recall  {:.3}, ".format(eval_stats[1]) + \
                    "laneline Precision  {:.3}, ".format(eval_stats[2]) + \
                    "laneline Category Accuracy  {:.3}, ".format(eval_stats[3]) + \
                    "laneline xyz error  {:.3} m, ".format(eval_stats[4])
    with open(output_file, 'w') as f:
        f.write(output_str)
        f.close()
    if verbose:
        print(output_str)
    return eval_stats[0]

def eval_openlane(result_3d_dir, exp_name):
    print('Start to evaluate {} on test set'.format(exp_name))
    cases = ['up_down_case','curve_case','extreme_weather_case','intersection_case','merge_split_case','night_case']
    for case in cases:
        print('Start to evaluate {} on case: {}'.format(exp_name, case))
        # generate test list txt for both 2d and 3d
        eval_case_dir = os.path.abspath(os.path.join(cfg.output_dir, 'eval_results', '{}'.format(case)))
        mkdir_if_missing(eval_case_dir)
        test_list = os.path.join(eval_case_dir, 'test_list.txt')

        case_dir = os.path.join(cfg.dataset.dataset_dir, 'lane3d_1000/test', case)
        with open(test_list, 'w') as f:
            for seg_dir in glob(os.path.join(case_dir, '*')):
                for json_file in glob(os.path.join(seg_dir, '*.json')):
                    # only retain the last two parts of the path
                    json_file = os.path.join(os.path.basename(seg_dir), os.path.basename(json_file))
                    jpg_file = json_file.replace('.json', '.jpg')
                    f.write(jpg_file + '\n')
            f.close()

        # 3d evaluation
        annotation_dir = os.path.join(cfg.dataset.dataset_dir, 'lane3d_1000/test', case) + '/'
        output_dir = os.path.join(eval_case_dir, 'eval_3d') + '/'
        eval_3d_cmd(annotation_dir, result_3d_dir, test_list, output_dir, exp_name)

    print('Start to evaluate {} on validation set'.format(exp_name))
    annotation_dir = os.path.join(cfg.dataset.dataset_dir, 'lane3d_1000/validation') + '/'
    mkdir_if_missing(os.path.abspath(os.path.join(cfg.output_dir, 'eval_results/validation')))
    test_list = os.path.abspath(os.path.join(cfg.output_dir, 'eval_results/validation', 'test_list.txt'))
    with open(test_list, 'w') as f:
        for seg_dir in glob(os.path.join(annotation_dir, '*')):
            for json_file in glob(os.path.join(seg_dir, '*.json')):
                # only retain the last two parts of the path
                json_file = os.path.join(os.path.basename(seg_dir), os.path.basename(json_file))
                jpg_file = json_file.replace('.json', '.jpg')
                f.write(jpg_file + '\n')
        f.close()
    output_dir = os.path.abspath(os.path.join(cfg.output_dir, 'eval_results/validation', 'eval_3d')) + '/'
    eval_3d_cmd(annotation_dir, result_3d_dir, test_list, output_dir, exp_name)

def eval_3d_cmd(annotation_dir, result_dir, test_list, output_dir_default, exp_name):
    if 'results_det' in result_dir:
        output_dir = output_dir_default.replace('eval_3d', 'eval_3d_PersFormer')
    else:
        output_dir = output_dir_default
    output_dir = output_dir.replace('eval_3d', 'eval_3d-{}-{}'.format(cfg.evaluation.overlap_thd, cfg.evaluation.dist_thd))
    mkdir_if_missing(output_dir)
    result_dir = os.path.abspath(result_dir) + '/'
    eval_lane3d(annotation_dir, result_dir, test_list, output_dir, verbose=True)

def eval_epoch():
    result_3d_dir = [
        os.path.join(cfg.output_dir, 'results'),
        os.path.join(cfg.output_dir, 'results_det'),
    ]
    print('')
    eval_openlane(result_3d_dir[0], 'lane_mapping')
    print('')
    eval_openlane(result_3d_dir[1], 'PersFormer')