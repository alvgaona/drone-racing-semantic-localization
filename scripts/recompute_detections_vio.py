#!/usr/bin/env python3
"""Recompute gate detections using VIO poses instead of mocap.

Reads the recorded bag with VIO odometry and ground truth gate positions,
computes body-relative detections from the VIO pose to the gate positions
in earth frame, and writes a new bag with the recomputed detections.

Usage:
    pixi run python3 scripts/recompute_detections_vio.py \
        --input-bag rosbag2_2026_03_25-18_59_49 \
        --output-bag rosbag2_vio_detections \
        --detection-range 15.0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from as2_msgs.msg import PoseStampedWithID, PoseStampedWithIDArray


GATES_EARTH = {
    "gate_1": np.array([4.0, 1.3, 1.13]),
    "gate_2": np.array([4.0, -1.34, 1.16]),
    "gate_3": np.array([-4.0, -1.29, 1.16]),
    "gate_4": np.array([-3.97, 1.28, 1.17]),
}

EARTH_TO_MAP_T = np.array([-0.133262, -1.518612, -0.155813])
EARTH_TO_MAP_RPY = [0.008312, -0.001632, -3.125066]
R_E2M = Rotation.from_euler("xyz", EARTH_TO_MAP_RPY).as_matrix()


def vio_to_earth(pos, quat_xyzw):
    R_odom = Rotation.from_quat(quat_xyzw).as_matrix()
    earth_pos = R_E2M @ pos + EARTH_TO_MAP_T
    earth_rot = R_E2M @ R_odom
    return earth_pos, earth_rot


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-bag", type=Path, required=True)
    parser.add_argument("--output-bag", type=Path, required=True)
    parser.add_argument("--detection-range", type=float, default=15.0)
    parser.add_argument("--storage", choices=["mcap", "sqlite3"], default="mcap")
    parser.add_argument("--input-storage", choices=["mcap", "sqlite3"], default="mcap")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_bag.exists():
        print(f"ERROR: Output {args.output_bag} already exists")
        sys.exit(1)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.input_bag), storage_id=args.input_storage),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )

    all_topics = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in all_topics}

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(args.output_bag), storage_id=args.storage),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )

    for t in all_topics:
        if t.name == "/detections/gates":
            continue
        topic_meta = rosbag2_py.TopicMetadata(
            name=t.name, type=t.type, serialization_format="cdr"
        )
        writer.create_topic(topic_meta)

    det_topic = rosbag2_py.TopicMetadata(
        name="/detections/gates",
        type="as2_msgs/msg/PoseStampedWithIDArray",
        serialization_format="cdr",
    )
    writer.create_topic(det_topic)

    last_vio_pos = None
    last_vio_quat = None
    last_vio_ts = None
    last_vio_stamp = None
    det_count = 0
    msg_count = 0

    del reader
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.input_bag), storage_id=args.input_storage),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )

    while reader.has_next():
        topic, data, ts = reader.read_next()

        if topic == "/ov_msckf/odomimu":
            msg = deserialize_message(data, Odometry)
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            last_vio_pos = np.array([p.x, p.y, p.z])
            last_vio_quat = np.array([q.x, q.y, q.z, q.w])
            last_vio_ts = ts
            last_vio_stamp = msg.header.stamp
            writer.write(topic, data, ts)
            msg_count += 1

        elif topic == "/detections/gates":
            if last_vio_pos is None:
                continue

            earth_pos, earth_rot = vio_to_earth(last_vio_pos, last_vio_quat)

            det_msg = PoseStampedWithIDArray()
            header = Header()
            header.stamp = last_vio_stamp
            header.frame_id = "base_link"

            for gate_id, gate_earth in GATES_EARTH.items():
                rel_pos = earth_rot.T @ (gate_earth - earth_pos)
                dist = np.linalg.norm(rel_pos)

                if dist > args.detection_range:
                    continue

                rel_rot = earth_rot.T
                q = Rotation.from_matrix(rel_rot).as_quat()

                det = PoseStampedWithID()
                det.id = gate_id
                det.pose = PoseStamped()
                det.pose.header = header
                det.pose.pose = Pose(
                    position=Point(x=float(rel_pos[0]), y=float(rel_pos[1]), z=float(rel_pos[2])),
                    orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]),
                )
                det_msg.poses.append(det)

            if len(det_msg.poses) > 0:
                writer.write("/detections/gates", serialize_message(det_msg), ts)
                det_count += 1

            msg_count += 1

        else:
            writer.write(topic, data, ts)
            msg_count += 1

    del writer
    print(f"Wrote {msg_count} messages, {det_count} detection frames")
    print(f"Output: {args.output_bag}")


if __name__ == "__main__":
    main()
