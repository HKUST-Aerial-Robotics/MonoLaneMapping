import numpy as np
from scipy.spatial.transform import Rotation as R
# /home/qzj/code/catkin_iros23/devel/lib/python3/dist-packages
import os
import sys
sys.path.append("/home/qzj/code/catkin_iros23/devel/lib/python3/dist-packages")
from openlane_bag.msg import LaneList, Lane, LanePoint
from geometry_msgs.msg import PoseStamped

def posemsg_to_np(pose_msg):
    pose = np.eye(4)
    pose[:3, 3] = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z])
    pose[:3, :3] = R.from_quat([pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w]).as_matrix()
    return pose

def lanemsg_to_list(lane_list_msg:LaneList):
    lane_list = []
    for lane_id in range(lane_list_msg.num_lanes):
        lane = lane_list_msg.lane_list[lane_id]
        lane_dict = {'xyz': [], 'category': lane.category, 'visibility': [], 'track_id': lane.track_id, 'attribute': lane.attribute}
        for lane_point_id in range(lane.num_points):
            lane_point = lane.lane[lane_point_id]
            lane_dict['xyz'].append([lane_point.x, lane_point.y, lane_point.z])
            lane_dict['visibility'].append(lane_point.visibility)
        lane_dict['xyz'] = np.asarray(lane_dict['xyz'])
        lane_dict['visibility'] = np.asarray(lane_dict['visibility'])
        lane_list.append(lane_dict)
    return lane_list