#!/usr/bin/env python3
"""Analyze keyframe CSV for graph stats and optimization latency.

Usage:
    pixi run python3 scripts/analyze_keyframes.py
    pixi run python3 scripts/analyze_keyframes.py --log-dir results/run-01p-dual-2.0m-0.5m
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    df = pd.read_csv(args.log_dir / "slam_keyframes.csv")
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    if len(df) == 0:
        print("No valid keyframes found")
        return

    last = df.iloc[-1]
    opt = df["opt_time_ms"]
    nodes = int(last["main_nodes"])
    edges = int(last["main_edges"])
    gate_edges = edges - nodes

    print(f"Keyframes:  {len(df)}")
    print(f"Nodes:      {nodes}")
    print(f"Edges:      {edges}")
    print(f"Gate edges: {gate_edges} (approx)")
    print()
    print("=== Optimization latency (ms) ===")
    print(f"Mean:  {opt.mean():.1f}")
    print(f"P50:   {opt.quantile(0.50):.1f}")
    print(f"P90:   {opt.quantile(0.90):.1f}")
    print(f"P95:   {opt.quantile(0.95):.1f}")
    print(f"P99:   {opt.quantile(0.99):.1f}")
    print(f"Max:   {opt.max():.1f}")
    print()
    print("=== Chi2 ===")
    chi2 = df["chi2_after"]
    print(f"First non-zero: {chi2[chi2 > 0].iloc[0]:.2f}" if (chi2 > 0).any() else "No non-zero chi2")
    print(f"Last:           {last['chi2_after']:.2f}")


if __name__ == "__main__":
    main()
