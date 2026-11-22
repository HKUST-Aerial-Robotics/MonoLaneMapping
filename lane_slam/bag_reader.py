"""Read the published OpenLane recordings."""

from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation

TOPICS = {
    '/gt_pose_wc': 'gt_pose',
    '/lanes_gt': 'lanes_gt',
    '/lanes_predict': 'lanes_predict',
}


def _pose_matrix(message):
    position = message.pose.position
    orientation = message.pose.orientation
    pose = np.eye(4)
    pose[:3, 3] = [position.x, position.y, position.z]
    pose[:3, :3] = Rotation.from_quat(
        [orientation.x, orientation.y, orientation.z, orientation.w]
    ).as_matrix()
    return pose


def _lane_list(message):
    lanes = []
    for lane_index in range(message.num_lanes):
        lane = message.lane_list[lane_index]
        points = [lane.lane[index] for index in range(lane.num_points)]
        lanes.append({
            'xyz': np.asarray([[point.x, point.y, point.z] for point in points]).reshape(-1, 3),
            'category': lane.category,
            'visibility': np.asarray([point.visibility for point in points]),
            'track_id': lane.track_id,
            'attribute': lane.attribute,
        })
    return lanes


def read_openlane_frames(bag_file):
    """Yield complete pose/GT/detection triples matched by sensor header stamp."""
    pending = {}
    with AnyReader([Path(bag_file)]) as reader:
        connections = [connection for connection in reader.connections if connection.topic in TOPICS]
        missing_topics = TOPICS.keys() - {connection.topic for connection in connections}
        if missing_topics:
            raise ValueError(f'{bag_file}: missing topics {", ".join(sorted(missing_topics))}')

        for connection, _, payload in reader.messages(connections=connections):
            message = reader.deserialize(payload, connection.msgtype)
            stamp = message.header.stamp
            key = (stamp.sec, stamp.nanosec)
            frame = pending.setdefault(key, {})
            frame[TOPICS[connection.topic]] = message
            if len(frame) == len(TOPICS):
                yield {
                    # Use the timestamp conversion expected by the published annotation names.
                    'timestamp': float(stamp.sec) + float(stamp.nanosec) / 1e9,
                    'gt_pose': _pose_matrix(frame['gt_pose']),
                    'lanes_gt': _lane_list(frame['lanes_gt']),
                    'lanes_predict': _lane_list(frame['lanes_predict']),
                }
                del pending[key]
