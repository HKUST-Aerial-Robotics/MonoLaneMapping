import numpy as np
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(ROOT_DIR)
from misc.config import define_args
from misc.config import cfg, cfg_from_yaml_file
from evaluation.eval_3D_lane import eval_epoch
from tqdm import tqdm
import glob
from lane_slam.system.lane_mapping import LaneMapping
from misc.pcd_utils import make_open3d_point_cloud
from misc.plot_utils import pointcloud_to_spheres, visualize_points_list
from lane_slam.lane_utils import points_downsample
from misc.curve.catmull_rom import CatmullRomSplineList

def dataset_stats():
    cfg.eval_pose = False
    bag_paths = sorted(glob.glob(os.path.join(cfg.dataset.dataset_dir, 'lane3d_1000/rosbag/*.bag')))
    result = []
    for seq, bag_path in enumerate(tqdm(bag_paths, leave=False)):
        lane_mapper = LaneMapping(bag_path)
        stats = lane_mapper.dataset_inspect()
        result.append(stats)
        if seq == 10:
            break
    keys = result[0].keys()
    for key in keys:
        avg = 0
        for seq, stats in enumerate(result):
            avg += stats[key]
        avg /= len(result)
        print(key, ": ", avg)
        print()
    return result

def draw_maps():
    visualization_dir = os.path.join(cfg.output_dir, 'visualization')
    segments = os.listdir(visualization_dir)

    for segment in segments:
        map_file = os.path.join(visualization_dir, segment, 'map.npy')
        map_items = np.load(map_file, allow_pickle=True).item()['lanes_in_map']
        visualize_map(map_items, segment)

def visualize_map(map_items, segment):

    vis_pcds = []
    vis_points = []
    print(segment)
    for lane_id, lane_dict in map_items.items():
        lane_pts = lane_dict['xyz_raw']
        ctrl_pts = lane_dict['ctrl_pts']
        lane_pts = points_downsample(lane_pts, 0.3)
        pcd = make_open3d_point_cloud(lane_pts, color=[0.4, 0.4, 0.4])
        ctrl_pcds = pointcloud_to_spheres(ctrl_pts, sphere_size=0.4)
        curve = CatmullRomSplineList(np.array(ctrl_pts))
        fitted_pts = curve.get_points(30)
        vis_points.append(fitted_pts)

        lane_text = [pcd, ctrl_pcds]

        # text_pos = np.mean(ctrl_pts, axis=0) + np.array([0.5, -0.5, 2])
        # text_pcd = text_3d(str(lane_id), text_pos, font_size=100, degree=-90.0)
        # lane_text.append(text_pcd)

        # visualize_points_list([fitted_pts], extra_pcd=lane_text, title='lane {}'.format(lane_id))
        vis_pcds.extend(lane_text)

    visualize_points_list(vis_points, extra_pcd=vis_pcds, title=segment)

def analyse_stats():
    file_path = os.path.join(cfg.output_dir, 'stats.npy')
    stats = np.load(file_path, allow_pickle=True).item()
    graph_time = 0
    isam_time = 0
    for seg, result in stats.items():
        graph_time = graph_time + result['graph']
        isam_time = isam_time + result['isam']
    print(graph_time/202)
    print(isam_time/202)

if __name__ == '__main__':

    args = define_args()
    cfg_from_yaml_file(os.path.join(ROOT_DIR, args.cfg_file), cfg)
    cfg_name = args.cfg_file.split('/')[-1].split('.')[0]
    cfg.output_dir = os.path.join(cfg.ROOT_DIR, "outputs", cfg_name)
    np.random.seed(666)
    # draw_maps()
    eval_epoch()
    # dataset_stats()
    # analyse_stats()