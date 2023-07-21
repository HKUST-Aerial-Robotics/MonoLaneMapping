#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk

import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation
import tf

class TFPublisher:

    def __init__(self, frame_name = 'camera_link'):
        self.br = tf.TransformBroadcaster()
        self.frame_name = frame_name

    def publish_tf(self, pose, timestamp):
        # np 4x4
        q = Rotation.from_matrix(pose[:3, :3]).as_quat()
        self.br.sendTransform((pose[0, 3], pose[1, 3], pose[2, 3]), q, timestamp, self.frame_name, 'map')