#!/usr/bin/env python3
"""Patch CameraInfo in an existing bag with calibrated intrinsics.

Reads all messages from the input bag, replaces /camera/camera_info
messages with calibrated equidistant fisheye parameters, writes to
a new bag (or in-place with --inplace).

Usage:
    pixi run python3 scripts/fix_camera_info.py \
        --bag dataset/piloted/flight-01p-ellipse/rosbag2_vio_detections
"""
import argparse
import shutil
import tempfile
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import CameraInfo

CALIBRATED = {
    "fx": 291.520, "fy": 390.011, "cx": 316.447, "cy": 240.442,
    "distortion_model": "equidistant",
    "d": [0.0478, -0.0282, 0.0376, -0.0184],
}


def make_camera_info(header, width=640, height=480):
    fx, fy = CALIBRATED["fx"], CALIBRATED["fy"]
    cx, cy = CALIBRATED["cx"], CALIBRATED["cy"]

    msg = CameraInfo()
    msg.header = header
    msg.width = width
    msg.height = height
    msg.distortion_model = CALIBRATED["distortion_model"]
    msg.d = CALIBRATED["d"]
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output bag (default: overwrite input)")
    parser.add_argument("--topic", default="/camera/camera_info")
    parser.add_argument("--storage", default="mcap")
    args = parser.parse_args()

    inplace = args.output is None
    output_path = Path(tempfile.mkdtemp()) / "patched" if inplace else args.output

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id=args.storage),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"))

    topics = reader.get_all_topics_and_types()
    topic_types = {t.name: t.type for t in topics}

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(output_path), storage_id=args.storage),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"))

    for t in topics:
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=t.name, type=t.type,
            serialization_format=t.serialization_format))

    patched = 0
    total = 0
    while reader.has_next():
        topic, data, ts = reader.read_next()
        total += 1
        if topic == args.topic:
            orig = deserialize_message(data, CameraInfo)
            fixed = make_camera_info(orig.header, orig.width, orig.height)
            data = serialize_message(fixed)
            patched += 1
        writer.write(topic, data, ts)

    del writer
    del reader

    if inplace:
        shutil.rmtree(str(args.bag))
        shutil.move(str(output_path), str(args.bag))

    print(f"Patched {patched}/{total} messages in {args.bag}")


if __name__ == "__main__":
    main()
