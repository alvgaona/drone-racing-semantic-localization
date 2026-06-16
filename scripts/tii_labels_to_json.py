#!/usr/bin/env python3
"""Convert TII per-frame .txt labels to labels.json for GateNet training.

TII format (per line): class cx cy w h c1x c1y vis1 c2x c2y vis2 c3x c3y vis3 c4x c4y vis4
JSON format: {"filename.jpg": [[c1x, c1y, c2x, c2y, c3x, c3y, c4x, c4y], ...], ...}

Usage:
    pixi run -e train python3 scripts/tii_labels_to_json.py \
        --labels-dir dataset/piloted/flight-01p-ellipse/label_flight-01p-ellipse \
        --output dataset/piloted/flight-01p-ellipse/labels.json
"""
import argparse
import json
from pathlib import Path


def parse_label_line(line):
    parts = line.strip().split()
    if len(parts) < 17:
        return None

    corners = []
    idx = 5
    all_visible = True
    for _ in range(4):
        cx = float(parts[idx])
        cy = float(parts[idx + 1])
        vis = int(parts[idx + 2])
        if vis == 0:
            all_visible = False
        corners.extend([cx, cy])
        idx += 3

    if not all_visible:
        return None

    return corners


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-ext", default=".jpg")
    args = parser.parse_args()

    labels_dict = {}
    total_gates = 0
    skipped = 0

    for txt_file in sorted(args.labels_dir.glob("*.txt")):
        image_name = txt_file.stem + args.image_ext
        gates = []

        with open(txt_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                corners = parse_label_line(line)
                if corners is not None:
                    gates.append(corners)
                    total_gates += 1
                else:
                    skipped += 1

        if gates:
            labels_dict[image_name] = gates

    with open(args.output, "w") as f:
        json.dump(labels_dict, f)

    print(f"Images: {len(labels_dict)}")
    print(f"Gates: {total_gates} (skipped {skipped} with occluded corners)")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
