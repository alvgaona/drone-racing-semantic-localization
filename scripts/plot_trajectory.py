#!/usr/bin/env python3
"""Plot trajectory comparison: VIO vs SLAM-corrected vs Ground Truth.

Generates a publication-quality figure with XY, 3D, X, Y, Z, Roll, Pitch, Yaw subplots.

Usage:
    pixi run python3 scripts/plot_trajectory.py \
        --e2m-xyz 0.408 -0.311 -0.271 --e2m-rpy -0.000 -0.033 -1.527

    pixi run python3 scripts/plot_trajectory.py \
        --log-dir results/run-09p-dual-2.0m-0.5m \
        --e2m-xyz 0.408 -0.311 -0.271 --e2m-rpy -0.000 -0.033 -1.527 \
        --output results/traj_09p.pdf
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", type=Path, default=Path("."))
    parser.add_argument("--e2m-xyz", type=float, nargs=3, required=True)
    parser.add_argument("--e2m-rpy", type=float, nargs=3, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"),
                        help="Directory to save individual plot images")
    parser.add_argument("--prefix", type=str, default="traj",
                        help="Filename prefix for plots")
    parser.add_argument("--title", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    odom = pd.read_csv(args.log_dir / "slam_odom.csv").apply(pd.to_numeric, errors="coerce").dropna()
    gt = pd.read_csv(args.log_dir / "slam_ground_truth.csv")

    R_e2m = Rotation.from_euler("xyz", args.e2m_rpy).as_matrix()
    t_e2m = np.array(args.e2m_xyz)

    gt_ts = (gt["sec"] + gt["nsec"] * 1e-9).values
    odom_ts = (odom["sec"] + odom["nsec"] * 1e-9).values
    t0 = odom_ts[0]
    odom_t = odom_ts - t0
    gt_t = gt_ts - t0

    gt_xyz = gt[["x", "y", "z"]].values
    gt_quat = gt[["qx", "qy", "qz", "qw"]].values
    gt_rpy = np.degrees(Rotation.from_quat(gt_quat).as_euler("xyz"))

    cor_xyz = odom[["cor_x", "cor_y", "cor_z"]].values
    cor_quat = odom[["cor_qx", "cor_qy", "cor_qz", "cor_qw"]].values
    cor_rpy = np.degrees(Rotation.from_quat(cor_quat).as_euler("xyz"))

    raw_pos = odom[["raw_x", "raw_y", "raw_z"]].values
    raw_quat = odom[["raw_qx", "raw_qy", "raw_qz", "raw_qw"]].values
    raw_xyz = (R_e2m.T @ (raw_pos - t_e2m).T).T
    raw_rpy = np.degrees(np.array([
        (Rotation.from_matrix(R_e2m.T) * Rotation.from_quat(q)).as_euler("xyz")
        for q in raw_quat
    ]))

    c_gt = "#000000"
    c_vio = "#e67e22"
    c_cor = "#2980b9"
    ls_gt = "--"
    ls_vio = "-"
    ls_cor = "-"
    lw_gt = 1.2
    lw = 0.8
    alpha_vio = 0.7

    gate_corners = [
        np.array([[4.010, 2.077, 1.922], [4.014, 0.539, 1.931],
                  [3.976, 0.555, 0.405], [4.039, 2.060, 0.397]]),
        np.array([[3.891, -0.534, 1.927], [3.989, -2.053, 1.944],
                  [4.000, -2.034, 0.408], [3.983, -0.524, 0.397]]),
        np.array([[- 3.900, -0.552, 1.936], [-3.942, -2.093, 1.931],
                  [-3.973, -2.074, 0.398], [-3.957, -0.569, 0.404]]),
        np.array([[-3.912, 0.547, 1.904], [-4.006, 2.089, 1.910],
                  [-4.044, 2.059, 0.382], [-3.961, 0.553, 0.379]]),
    ]
    gate_labels = ["G1", "G2", "G3", "G4"]
    c_gate = "#7f8c8d"

    def draw_gates_2d(ax):
        for corners, lbl in zip(gate_corners, gate_labels):
            top = corners[:2]
            ax.plot([top[0, 0], top[1, 0]], [top[0, 1], top[1, 1]],
                    color=c_gate, lw=2.5, solid_capstyle="round")
            center = corners.mean(axis=0)
            ax.annotate(lbl, (center[0], center[1]), textcoords="offset points",
                        xytext=(6, 6), fontsize=7, color=c_gate, fontweight="bold")

    def draw_gates_3d(ax):
        for corners, lbl in zip(gate_corners, gate_labels):
            closed = np.vstack([corners, corners[0]])
            ax.plot(closed[:, 0], closed[:, 1], closed[:, 2],
                    color=c_gate, lw=2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    title = args.title or ""

    def save_fig(fig, name):
        path = args.output_dir / f"{args.prefix}_{name}.jpg"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  {path}")

    # XY trajectory
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], color=c_gt, lw=lw_gt, ls=ls_gt, label="Ground Truth")
    ax.plot(raw_xyz[:, 0], raw_xyz[:, 1], color=c_vio, lw=lw, alpha=alpha_vio, label="OpenVINS")
    ax.plot(cor_xyz[:, 0], cor_xyz[:, 1], color=c_cor, lw=lw, label="Ours (Localization)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    draw_gates_2d(ax)
    ax.set_title(f"XY Trajectory — {title}" if title else "XY Trajectory")
    ax.set_aspect("equal")
    ax.legend()
    save_fig(fig, "xy")

    # 3D trajectory
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], color=c_gt, lw=lw_gt, ls=ls_gt, label="Ground Truth")
    ax.plot(raw_xyz[:, 0], raw_xyz[:, 1], raw_xyz[:, 2],
            color=c_vio, lw=lw, alpha=alpha_vio, label="OpenVINS")
    ax.plot(cor_xyz[:, 0], cor_xyz[:, 1], cor_xyz[:, 2],
            color=c_cor, lw=lw, label="Ours (Localization)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    draw_gates_3d(ax)
    ax.set_title(f"3D Trajectory — {title}" if title else "3D Trajectory")
    ax.legend(fontsize=7)
    save_fig(fig, "3d")

    # 3D trajectory — zenithal (top-down) view
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], color=c_gt, lw=lw_gt, ls=ls_gt, label="Ground Truth")
    ax.plot(raw_xyz[:, 0], raw_xyz[:, 1], raw_xyz[:, 2],
            color=c_vio, lw=lw, alpha=alpha_vio, label="OpenVINS")
    ax.plot(cor_xyz[:, 0], cor_xyz[:, 1], cor_xyz[:, 2],
            color=c_cor, lw=lw, label="Ours (Localization)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    draw_gates_3d(ax)
    ax.set_title(f"3D Trajectory (top) — {title}" if title else "3D Trajectory (top)")
    ax.view_init(elev=90, azim=-90)
    ax.legend(fontsize=7)
    save_fig(fig, "3d_top")

    # X, Y, Z vs time
    for i, (label, name) in enumerate(zip(["X (m)", "Y (m)", "Z (m)"], ["x", "y", "z"])):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(gt_t, gt_xyz[:, i], color=c_gt, lw=lw_gt, ls=ls_gt, label="GT")
        ax.plot(odom_t, raw_xyz[:, i], color=c_vio, lw=lw, alpha=alpha_vio, label="OpenVINS")
        ax.plot(odom_t, cor_xyz[:, i], color=c_cor, lw=lw, label="SLAM")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(label)
        ax.set_title(f"{name.upper()} vs Time — {title}" if title else f"{name.upper()} vs Time")
        ax.legend()
        save_fig(fig, name)

    # Roll, Pitch, Yaw vs time
    for i, (label, name) in enumerate(zip(["Roll (deg)", "Pitch (deg)", "Yaw (deg)"],
                                          ["roll", "pitch", "yaw"])):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(gt_t, gt_rpy[:, i], color=c_gt, lw=lw_gt, ls=ls_gt, label="GT")
        ax.plot(odom_t, raw_rpy[:, i], color=c_vio, lw=lw, alpha=alpha_vio, label="OpenVINS")
        ax.plot(odom_t, cor_rpy[:, i], color=c_cor, lw=lw, label="SLAM")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(label)
        ax.set_title(f"{name.capitalize()} vs Time — {title}" if title else f"{name.capitalize()} vs Time")
        ax.legend()
        save_fig(fig, name)

    print(f"Saved 8 plots to {args.output_dir}/")


if __name__ == "__main__":
    main()
