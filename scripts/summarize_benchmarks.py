#!/usr/bin/env python3
"""Summarize benchmark results: mean ± std per configuration and track type.

Groups results by (mode, main_thresh, temp_thresh, det_cov, orient_cov) and track
(ellipse/lemniscate), then computes mean ± std across flights for each metric.

Usage:
    pixi run python3 scripts/summarize_benchmarks.py
    pixi run python3 scripts/summarize_benchmarks.py --csv results/benchmark_results.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path,
                        default=Path("results/benchmark_results.csv"))
    parser.add_argument("--output", type=Path, default=None,
                        help="Save summary to CSV (default: print only)")
    parser.add_argument("--detailed", action="store_true",
                        help="Show ATE XY and Z breakdowns")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)

    df = df[df["mode"] != "VIO only"].copy()
    df["temp_thresh"] = df["temp_thresh"].fillna("")

    config_cols = ["mode", "main_thresh", "temp_thresh", "det_cov", "orient_cov"]
    metric_cols = ["ate_3d", "pct_3d", "ate_xy", "pct_xy", "ate_z", "pct_z",
                   "ate_rot", "pct_rot", "nodes", "edges", "opt_p95"]

    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    results = []

    for track in ["ellipse", "lemniscate"]:
        track_df = df[df["track"] == track]
        if track_df.empty:
            continue

        grouped = track_df.groupby(config_cols)

        for config, group in grouped:
            if len(group) < 2:
                continue

            row = {"track": track, "n_flights": len(group)}
            for i, col_name in enumerate(config_cols):
                row[col_name] = config[i]

            for metric in metric_cols:
                vals = group[metric].dropna()
                if len(vals) > 0:
                    row[f"{metric}_mean"] = vals.mean()
                    row[f"{metric}_std"] = vals.std()

            results.append(row)

    if not results:
        print("No configurations with >= 2 flights found")
        return

    summary = pd.DataFrame(results)

    vio_df = pd.read_csv(args.csv)
    vio_df = vio_df[vio_df["mode"] == "VIO only"]

    for track in ["ellipse", "lemniscate"]:
        track_summary = summary[summary["track"] == track]
        if track_summary.empty:
            continue

        print(f"\n{'='*80}")
        print(f"  {track.upper()} — Mean ± Std across flights")
        print(f"{'='*80}\n")

        vio_track = vio_df[vio_df["track"] == track]
        if not vio_track.empty:
            n = len(vio_track)
            print(f"VIO baseline (n={n} flights):")
            print(f"  ATE TRANS:  {vio_track['ate_3d'].mean():.3f} ± {vio_track['ate_3d'].std():.3f} m")
            if args.detailed:
                print(f"  ATE XY:  {vio_track['ate_xy'].mean():.3f} ± {vio_track['ate_xy'].std():.3f} m")
                print(f"  ATE Z:   {vio_track['ate_z'].mean():.3f} ± {vio_track['ate_z'].std():.3f} m")
            print(f"  ATE ROT: {vio_track['ate_rot'].mean():.2f} ± {vio_track['ate_rot'].std():.2f} deg")
            print()

        for _, row in track_summary.iterrows():
            config_str = f"{row['mode']} main={row['main_thresh']}"
            if row["temp_thresh"]:
                config_str += f" temp={row['temp_thresh']}"

            print(f"Config: {config_str}  (n={int(row['n_flights'])} flights)")
            print(f"  ATE TRANS:  {row['ate_3d_mean']:.3f} ± {row['ate_3d_std']:.3f} m"
                  f"  ({row['pct_3d_mean']:+.1f} ± {row['pct_3d_std']:.1f}%)")
            if args.detailed:
                print(f"  ATE XY:  {row['ate_xy_mean']:.3f} ± {row['ate_xy_std']:.3f} m"
                      f"  ({row['pct_xy_mean']:+.1f} ± {row['pct_xy_std']:.1f}%)")
                print(f"  ATE Z:   {row['ate_z_mean']:.3f} ± {row['ate_z_std']:.3f} m"
                      f"  ({row['pct_z_mean']:+.1f} ± {row['pct_z_std']:.1f}%)")
            print(f"  ATE ROT: {row['ate_rot_mean']:.2f} ± {row['ate_rot_std']:.2f} deg"
                  f"  ({row['pct_rot_mean']:+.1f} ± {row['pct_rot_std']:.1f}%)")
            print(f"  Nodes:   {row['nodes_mean']:.0f} ± {row['nodes_std']:.0f}"
                  f"  Edges: {row['edges_mean']:.0f} ± {row['edges_std']:.0f}"
                  f"  Opt P95: {row['opt_p95_mean']:.1f} ± {row['opt_p95_std']:.1f} ms")
            print()

    if args.output:
        summary.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
