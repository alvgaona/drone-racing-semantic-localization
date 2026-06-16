#!/usr/bin/env python3
"""Compute RPE (Relative Pose Error) over fixed distance segments.

Reads slam_odom.csv and slam_ground_truth.csv, computes RPE over configurable
distance windows (default: 1m, 5m, 10m).

Usage:
    pixi run python3 scripts/compute_rpe.py --e2m-xyz 0.408 -0.311 -0.271 --e2m-rpy -0.000 -0.033 -1.527
    pixi run python3 scripts/compute_rpe.py --e2m-xyz 0.408 -0.311 -0.271 --e2m-rpy -0.000 -0.033 -1.527 \
        --deltas 1 2 5 10
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
    parser.add_argument("--e2m-xyz", type=float, nargs=3, required=True)
    parser.add_argument("--e2m-rpy", type=float, nargs=3, required=True)
    parser.add_argument("--deltas", type=float, nargs="+", default=[1.0, 5.0, 10.0],
                        help="Distance segments in meters (default: 1 5 10)")
    return parser.parse_args()


def compute_rpe(positions, gt_positions, delta_m):
    rpe = []
    cumulative_dist = np.zeros(len(gt_positions))
    for i in range(1, len(gt_positions)):
        cumulative_dist[i] = cumulative_dist[i-1] + np.linalg.norm(gt_positions[i] - gt_positions[i-1])

    for i in range(len(positions)):
        target_dist = cumulative_dist[i] + delta_m
        j = np.searchsorted(cumulative_dist, target_dist)
        if j >= len(positions):
            break

        gt_rel = gt_positions[j] - gt_positions[i]
        est_rel = positions[j] - positions[i]
        rpe.append(np.linalg.norm(est_rel - gt_rel))

    return np.array(rpe)


def main():
    args = parse_args()

    odom = pd.read_csv(args.log_dir / "slam_odom.csv").apply(pd.to_numeric, errors="coerce").dropna()
    gt = pd.read_csv(args.log_dir / "slam_ground_truth.csv")

    R_e2m = Rotation.from_euler("xyz", args.e2m_rpy).as_matrix()
    t_e2m = np.array(args.e2m_xyz)

    gt_ts = (gt["sec"] + gt["nsec"] * 1e-9).values
    odom_ts = (odom["sec"] + odom["nsec"] * 1e-9).values
    gt_xyz = gt[["x", "y", "z"]].values

    cor_xyz = odom[["cor_x", "cor_y", "cor_z"]].values
    raw_pos = odom[["raw_x", "raw_y", "raw_z"]].values
    raw_xyz = (R_e2m.T @ (raw_pos - t_e2m).T).T

    gt_idx = np.array([np.argmin(np.abs(gt_ts - t)) for t in odom_ts])
    matched_gt = gt_xyz[gt_idx]

    total_dist = np.sum(np.linalg.norm(np.diff(matched_gt, axis=0), axis=1))
    ate_raw = np.sqrt(np.mean(np.sum((raw_xyz - matched_gt)**2, axis=1)))
    ate_cor = np.sqrt(np.mean(np.sum((cor_xyz - matched_gt)**2, axis=1)))

    print(f"Samples: {len(odom)}, Trajectory: {total_dist:.0f}m")
    print(f"ATE:  VIO={ate_raw:.3f}m  Corrected={ate_cor:.3f}m  "
          f"({(1-ate_cor/ate_raw)*100:+.1f}%)")
    print(f"ATE/m: VIO={ate_raw/total_dist*100:.3f}%  Corrected={ate_cor/total_dist*100:.3f}%")
    print()

    print(f"{'Delta':<8} {'VIO RMSE':<12} {'Cor RMSE':<12} {'Improv':<10} {'VIO %/m':<10} {'Cor %/m':<10} {'n'}")
    print("-" * 70)

    for delta in args.deltas:
        rpe_raw = compute_rpe(raw_xyz, matched_gt, delta)
        rpe_cor = compute_rpe(cor_xyz, matched_gt, delta)

        if len(rpe_raw) == 0:
            print(f"{delta:.0f}m       — (not enough data)")
            continue

        rmse_raw = np.sqrt(np.mean(rpe_raw**2))
        rmse_cor = np.sqrt(np.mean(rpe_cor**2))
        pct = (1 - rmse_cor / rmse_raw) * 100

        print(f"{delta:.0f}m       {rmse_raw:.4f}m      {rmse_cor:.4f}m      {pct:+.1f}%     "
              f"{rmse_raw/delta*100:.2f}%      {rmse_cor/delta*100:.2f}%      {len(rpe_raw)}")


if __name__ == "__main__":
    main()
