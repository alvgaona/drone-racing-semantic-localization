#!/usr/bin/env python3
"""Convert SLAM CSV logs to TUM format for evo evaluation.

Generates three TUM files: ground_truth.tum, vio_raw.tum, slam_corrected.tum

Usage:
    pixi run python3 scripts/csv_to_tum.py
    pixi run python3 scripts/csv_to_tum.py --log-dir results/run-01p-dual-2.0m-0.5m
    pixi run python3 scripts/csv_to_tum.py --e2m-xyz -0.536 -1.718 -0.108 --e2m-rpy 0.020 -0.049 3.133

Then use evo:
    evo_ape tum ground_truth.tum slam_corrected.tum -a
    evo_ape tum ground_truth.tum vio_raw.tum -a
    evo_traj tum ground_truth.tum vio_raw.tum slam_corrected.tum --plot
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: same as log-dir)")
    parser.add_argument("--e2m-xyz", type=float, nargs=3, default=None,
                        help="earth_to_map translation for raw VIO transform")
    parser.add_argument("--e2m-rpy", type=float, nargs=3, default=None,
                        help="earth_to_map rotation for raw VIO transform")
    return parser.parse_args()


def write_tum(path, timestamps, positions, quaternions):
    with open(path, "w") as f:
        for i in range(len(timestamps)):
            t = timestamps[i]
            p = positions[i]
            q = quaternions[i]
            f.write(f"{t:.9f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
    print(f"Wrote {len(timestamps)} poses to {path}")


def main():
    args = parse_args()
    output_dir = args.output_dir or args.log_dir

    gt = pd.read_csv(args.log_dir / "slam_ground_truth.csv")
    odom = pd.read_csv(args.log_dir / "slam_odom.csv")
    odom = odom.apply(pd.to_numeric, errors="coerce").dropna()

    gt_ts = gt["sec"] + gt["nsec"] * 1e-9
    odom_ts = odom["sec"] + odom["nsec"] * 1e-9

    write_tum(
        output_dir / "ground_truth.tum",
        gt_ts.values,
        gt[["x", "y", "z"]].values,
        gt[["qx", "qy", "qz", "qw"]].values,
    )

    write_tum(
        output_dir / "slam_corrected.tum",
        odom_ts.values,
        odom[["cor_x", "cor_y", "cor_z"]].values,
        odom[["cor_qx", "cor_qy", "cor_qz", "cor_qw"]].values,
    )

    if args.e2m_xyz is not None and args.e2m_rpy is not None:
        R_e2m = Rotation.from_euler("xyz", args.e2m_rpy).as_matrix()
        t_e2m = np.array(args.e2m_xyz)

        raw_pos = odom[["raw_x", "raw_y", "raw_z"]].values
        raw_quat = odom[["raw_qx", "raw_qy", "raw_qz", "raw_qw"]].values

        earth_pos = (R_e2m.T @ (raw_pos - t_e2m).T).T
        earth_quat = np.array([
            (Rotation.from_matrix(R_e2m.T) * Rotation.from_quat(q)).as_quat()
            for q in raw_quat
        ])

        write_tum(
            output_dir / "vio_raw.tum",
            odom_ts.values,
            earth_pos,
            earth_quat,
        )
    else:
        print("Skipping vio_raw.tum (no --e2m-xyz/--e2m-rpy provided)")


if __name__ == "__main__":
    main()
