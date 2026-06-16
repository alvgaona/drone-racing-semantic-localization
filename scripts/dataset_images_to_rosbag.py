#!/usr/bin/env python3
"""Convert TII dataset camera images to a standalone ROS2 bag.

This creates a lightweight bag with only /camera/image_raw and /camera/camera_info,
separate from the main dataset bag so images don't need to be regenerated.

Usage:
    pixi run python3 scripts/dataset_images_to_rosbag.py \
        --dataset-dir dataset/piloted/flight-01p-ellipse

    pixi run python3 scripts/dataset_images_to_rosbag.py \
        --dataset-dir dataset/piloted/flight-01p-ellipse \
        --camera-skip 3 --compress
"""
import argparse
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import yaml

import rosbag2_py
from rclpy.serialization import serialize_message

from builtin_interfaces.msg import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header


CALIBRATED_INTRINSICS = {
    "fx": 291.520, "fy": 390.011, "cx": 316.447, "cy": 240.442,
    "distortion_model": "equidistant",
    "d": [0.0478, -0.0282, 0.0376, -0.0184],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output bag directory (default: <dataset-dir>/rosbag_images)")
    parser.add_argument("--compress", action="store_true", help="Enable zstd compression")
    parser.add_argument("--image-encoding", choices=["bgr8", "mono8"], default="bgr8")
    parser.add_argument("--camera-skip", type=int, default=0,
                        help="Skip N frames between each written frame (e.g., 3 → ~30Hz from 120Hz)")
    return parser.parse_args()


def ts_us_to_ros(timestamp_us: int) -> Time:
    t = Time()
    t.sec = int(timestamp_us // 1_000_000)
    t.nanosec = int((timestamp_us % 1_000_000) * 1000)
    return t


def ts_us_to_ns(timestamp_us: int) -> int:
    return int(timestamp_us) * 1000


def make_header(timestamp_us: int, frame_id: str) -> Header:
    h = Header()
    h.stamp = ts_us_to_ros(timestamp_us)
    h.frame_id = frame_id
    return h


def make_camera_info(width: int = 640, height: int = 480) -> CameraInfo:
    fx = CALIBRATED_INTRINSICS["fx"]
    fy = CALIBRATED_INTRINSICS["fy"]
    cx = CALIBRATED_INTRINSICS["cx"]
    cy = CALIBRATED_INTRINSICS["cy"]

    msg = CameraInfo()
    msg.width = width
    msg.height = height
    msg.distortion_model = CALIBRATED_INTRINSICS["distortion_model"]
    msg.d = CALIBRATED_INTRINSICS["d"]
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    flight_id = dataset_dir.name

    camera_csv = dataset_dir / "csv_raw" / f"camera_{flight_id}.csv"
    image_dir = dataset_dir / f"camera_{flight_id}"
    metadata_yaml = dataset_dir / f"metadata_{flight_id}.yaml"
    output_dir = args.output_dir or dataset_dir / "rosbag_images"

    if output_dir.exists():
        print(f"ERROR: Output {output_dir} already exists. Remove it first.")
        return

    with open(metadata_yaml) as f:
        metadata = yaml.safe_load(f)

    width = metadata.get("camera", {}).get("image_width", 640)
    height = metadata.get("camera", {}).get("image_height", 480)

    writer = rosbag2_py.SequentialWriter()
    storage_config_uri = ""
    if args.compress:
        config_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        config_file.write("output:\n  compression: zstd\n  chunk_size: 4194304\n")
        config_file.close()
        storage_config_uri = config_file.name

    writer.open(
        rosbag2_py.StorageOptions(uri=str(output_dir), storage_id="mcap",
                                  storage_config_uri=storage_config_uri),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                    output_serialization_format="cdr"),
    )

    for topic_name, topic_type in [
        ("/camera/image_raw", "sensor_msgs/msg/Image"),
        ("/camera/camera_info", "sensor_msgs/msg/CameraInfo"),
    ]:
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=topic_name, type=topic_type, serialization_format="cdr"))

    camera_info = make_camera_info(width, height)
    df = pd.read_csv(camera_csv)
    encoding = args.image_encoding
    count = 0

    for idx, row in df.iterrows():
        if args.camera_skip > 0 and idx % (args.camera_skip + 1) != 0:
            continue

        ts = int(row["timestamp"])
        img_path = image_dir / row["filename"]
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        if encoding == "mono8":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        header = make_header(ts, "camera")

        img_msg = Image()
        img_msg.header = header
        img_msg.height = img.shape[0]
        img_msg.width = img.shape[1]
        img_msg.encoding = encoding
        img_msg.is_bigendian = False
        img_msg.step = img.shape[1] if encoding == "mono8" else img.shape[1] * 3
        img_msg.data = img.tobytes()

        camera_info.header = header
        ts_ns = ts_us_to_ns(ts)
        writer.write("/camera/image_raw", serialize_message(img_msg), ts_ns)
        writer.write("/camera/camera_info", serialize_message(camera_info), ts_ns)
        count += 1

        if count % 1000 == 0:
            print(f"  {count}/{len(df)} images...")

    del writer
    print(f"Wrote {count} images to {output_dir}")


if __name__ == "__main__":
    main()
