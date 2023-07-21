#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk

import rospy
import rosbag
import os
from nav_msgs.msg import Odometry
import std_msgs
from cv_bridge import CvBridge

from misc.utils import mkdirs
from misc.lie_utils import so3_to_quat
from misc.ros_utils.cloud_publisher import create_cloud_xyzcvt32

class BagManager:
    def __init__(self, bag_path, enable = True):
        self.bag_path = bag_path
        dir_path = os.path.dirname(self.bag_path)
        mkdirs(dir_path)
        self.bag = rosbag.Bag(self.bag_path, 'w')
        self.enable = enable
        self.bridge = CvBridge()

    def read_bag(self, bag_path, topic_name):
        for topic, msg, t in self.bag.read_messages(topics=[topic_name]):
            if topic == topic_name:
                yield msg

    def write_odom(self, rotation, translation, topic_name, timestamp_micros):
        # pose: 4x4
        # timestamp: us
        if not self.enable:
            return
        odom = Odometry()
        odom.header.stamp = rospy.Time.from_seconds(timestamp_micros / 1e6)
        odom.header.frame_id = 'map'
        odom.pose.pose.position.x = translation[0]
        odom.pose.pose.position.y = translation[1]
        odom.pose.pose.position.z = translation[2]
        rotation = so3_to_quat(rotation)
        odom.pose.pose.orientation.x = rotation[0]
        odom.pose.pose.orientation.y = rotation[1]
        odom.pose.pose.orientation.z = rotation[2]
        odom.pose.pose.orientation.w = rotation[3]
        self.bag.write(topic_name, odom, odom.header.stamp)

    def write_pcd(self, points, topic_name, timestamp_micros):
        # points: Nx6 numpy.darray or list
        if not self.enable:
            return
        stamp = rospy.Time.from_seconds(timestamp_micros / 1e6)
        header = std_msgs.msg.Header()
        header.stamp = stamp
        header.frame_id = 'camera_body'
        cloud = create_cloud_xyzcvt32(header, points)
        self.bag.write(topic_name, cloud, stamp)

    def close(self):
        self.bag.close()