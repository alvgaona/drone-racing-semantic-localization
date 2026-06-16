#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def compute_earth_to_map(fixed):
    earth_pts = fixed[["earth_x", "earth_y", "earth_z"]].values
    map_pts = fixed[["map_x", "map_y", "map_z"]].values
    earth_centroid = earth_pts.mean(axis=0)
    map_centroid = map_pts.mean(axis=0)
    H = (earth_pts - earth_centroid).T @ (map_pts - map_centroid)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = map_centroid - R @ earth_centroid
    return R, t


def transform_points(pts_xyz, R, t):
    return (R @ pts_xyz.T).T + t


def main():
    parser = argparse.ArgumentParser(description="Plot SemanticSlam CSV logs")
    parser.add_argument("--log-dir", type=Path, default=Path("."),
                        help="Directory containing slam_*.csv files")
    parser.add_argument("--show", action="store_true", help="Show interactive plot")
    parser.add_argument("--output", type=str, default="slam_debug_plots.png",
                        help="Output image filename")
    args = parser.parse_args()

    odom_path = args.log_dir / "slam_odom.csv"
    kf_path = args.log_dir / "slam_keyframes.csv"
    fixed_path = args.log_dir / "slam_fixed_objects.csv"
    gt_path = args.log_dir / "slam_ground_truth.csv"

    fixed = pd.read_csv(fixed_path) if fixed_path.exists() else None
    R_e2m, t_e2m = None, None
    if fixed is not None and len(fixed) >= 3:
        R_e2m, t_e2m = compute_earth_to_map(fixed)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    gt_map = None
    if gt_path.exists() and R_e2m is not None:
        gt = pd.read_csv(gt_path)
        gt_earth = gt[["x", "y", "z"]].values
        gt_map_pts = transform_points(gt_earth, R_e2m, t_e2m)
        gt_map = pd.DataFrame({"x": gt_map_pts[:, 0], "y": gt_map_pts[:, 1], "z": gt_map_pts[:, 2]})

    odom = pd.read_csv(odom_path) if odom_path.exists() else None
    kf = pd.read_csv(kf_path) if kf_path.exists() else None

    ax = axes[0, 0]
    if gt_path.exists():
        gt_raw = pd.read_csv(gt_path)
        ax.plot(gt_raw["x"], gt_raw["y"], "k-", alpha=0.4, linewidth=1.5, label="Ground Truth")
    if odom is not None:
        ax.plot(odom["cor_x"], odom["cor_y"], "b-", alpha=0.8, linewidth=1.0, label="Corrected")
    if fixed is not None:
        ax.scatter(fixed["map_x"], fixed["map_y"], c="green", s=100, marker="*",
                   zorder=5, label="Fixed gates (map)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Ground Truth vs Corrected (XY)")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if kf is not None:
        kf_idx = np.arange(len(kf))

        ax = axes[0, 1]
        ax.plot(kf_idx, kf["tf_x"], label="tf_x")
        ax.plot(kf_idx, kf["tf_y"], label="tf_y")
        ax.plot(kf_idx, kf["tf_z"], label="tf_z")
        ax.set_xlabel("Keyframe index")
        ax.set_ylabel("Translation (m)")
        ax.set_title("Map -> Odom Transform")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.semilogy(kf_idx, kf["chi2_before"].clip(lower=1e-6), "b-", alpha=0.5, label="chi2 before")
        ax.semilogy(kf_idx, kf["chi2_after"].clip(lower=1e-6), "r-", alpha=0.8, label="chi2 after")
        ax2 = ax.twinx()
        ax2.plot(kf_idx, kf["iterations"], "g--", alpha=0.5, label="iterations")
        ax2.set_ylabel("Iterations", color="green")
        ax.set_xlabel("Keyframe index")
        ax.set_ylabel("Chi2")
        ax.set_title("Graph Optimization Convergence")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if gt_path.exists():
        gt_raw = pd.read_csv(gt_path)
        ax.plot(gt_raw["x"], gt_raw["y"], "k-", alpha=0.4, linewidth=1.5, label="Ground Truth")
    if odom is not None:
        ax.plot(odom["cor_x"], odom["cor_y"], "b-", alpha=0.8, linewidth=1.0, label="Corrected")
    if fixed is not None:
        ax.scatter(fixed["earth_x"], fixed["earth_y"], c="green", s=100, marker="*",
                   zorder=5, label="Fixed gates")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Corrected vs Ground Truth")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = args.log_dir / args.output
    plt.savefig(str(out_path), dpi=150)
    print(f"Saved to {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
