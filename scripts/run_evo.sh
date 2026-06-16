#!/usr/bin/env bash
# Run evo evaluation on a recorded SLAM bag.
#
# Usage:
#   ./scripts/run_evo.sh slam_eval_03p/
#   ./scripts/run_evo.sh slam_eval_03p/ --save_results results/

set -euo pipefail

BAG="${1:?Usage: $0 <bag_path> [extra_evo_args...]}"
shift
EXTRA_ARGS="${@}"

REF="/ground_truth/pose"
VIO="/ov_msckf/odomimu"
SLAM="/drone/slam/corrected_localization"

echo "=== Trajectory plot ==="
pixi run evo_traj bag2 "$BAG" "$VIO" "$SLAM" --ref "$REF" -a --plot $EXTRA_ARGS

echo ""
echo "=== APE: VIO vs GT ==="
pixi run evo_ape bag2 "$BAG" "$REF" "$VIO" -a $EXTRA_ARGS

echo ""
echo "=== APE: Corrected vs GT ==="
pixi run evo_ape bag2 "$BAG" "$REF" "$SLAM" -a $EXTRA_ARGS
