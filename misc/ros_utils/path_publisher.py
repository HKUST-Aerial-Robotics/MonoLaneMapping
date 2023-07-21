#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import numpy as np
from scipy.spatial.transform import Rotation

class PathPublisher:

    def __init__(self, topic):
        self.path_pub = rospy.Publisher(topic, Path, queue_size=1000)
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'map'

    def publish_path_stamped(self, pose, timestamp):
        # np 4x4
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = 'map'
        pose_msg.header.stamp = timestamp
        q = Rotation.from_matrix(pose[:3, :3]).as_quat()
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]
        pose_msg.pose.position.x = pose[0, 3]
        pose_msg.pose.position.y = pose[1, 3]
        pose_msg.pose.position.z = pose[2, 3]

        self.path_msg.header.stamp = timestamp
        self.path_msg.poses.append(pose_msg)
        self.path_pub.publish(self.path_msg)