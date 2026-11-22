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
from pathlib import Path
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
    test_dir = Path(cfg.dataset.dataset_dir) / 'lane3d_1000/test'
    validation_dir = Path(cfg.dataset.annotation_dir)
    cases = {
        'up_down_case': '1000_updown.txt',
        'curve_case': '1000_curve.txt',
        'extreme_weather_case': '1000_extreme_weather.txt',
        'intersection_case': '1000_intersection.txt',
        'merge_split_case': '1000_merge_split_case.txt',
        'night_case': '1000_night.txt',
    }
    for case, manifest in cases.items():
        print('Start to evaluate {} on case: {}'.format(exp_name, case))
        annotation_dir = test_dir / case
        frames = _annotation_image_paths(annotation_dir)
        if not frames:
            split_file = test_dir / manifest
            frames = _manifest_image_paths(split_file)
            annotation_dir = validation_dir
        _eval_split(annotation_dir, frames, case, result_3d_dir, exp_name)

    print('Start to evaluate {} on validation set'.format(exp_name))
    _eval_split(validation_dir, _annotation_image_paths(validation_dir),
                'validation', result_3d_dir, exp_name)


def _manifest_image_paths(split_file):
    frames = {}
    for line in split_file.read_text().splitlines():
        if not line.strip():
            continue
        segment, image = line.strip().split('/')[-2:]
        # One published curve entry omits "_with" from the Waymo segment name.
        if segment.endswith('_camera_labels') and not segment.endswith('_with_camera_labels'):
            segment = segment.removesuffix('_camera_labels') + '_with_camera_labels'
        frames[f'{segment}/{image}'] = None
    return list(frames)


def _annotation_image_paths(annotation_dir):
    return [str(path.relative_to(annotation_dir).with_suffix('.jpg'))
            for path in sorted(annotation_dir.glob('*/*.json'))]


def _eval_split(annotation_dir, frames, split, result_dir, exp_name):
    if not frames:
        raise ValueError(f'No frames for {split} in {annotation_dir}')
    eval_dir = Path(cfg.output_dir) / 'eval_results' / split
    eval_dir.mkdir(parents=True, exist_ok=True)
    test_list = eval_dir / 'test_list.txt'
    test_list.write_text('\n'.join(frames) + '\n')
    eval_3d_cmd(str(annotation_dir) + '/', result_dir, str(test_list),
                str(eval_dir / 'eval_3d'), exp_name)


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