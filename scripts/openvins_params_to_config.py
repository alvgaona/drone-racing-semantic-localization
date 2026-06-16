#!/usr/bin/env python3
"""Convert OpenVINS online calibration output to kalibr_imucam_chain.yaml format.

Usage:
    pixi run python3 scripts/openvins_params_to_config.py \
        --quat "-0.364,0.366,-0.611,0.600" \
        --p_IinC "0.032,-0.005,-0.072" \
        --intrinsics "290.715,389.767,317.249,242.889" \
        --distortion "0.052,-0.034,0.042,-0.020" \
        --time-offset -0.01366

Paste values directly from the OpenVINS output lines:
    cam0 extrinsics = <quat> | <p_IinC>
    cam0 intrinsics = <intrinsics> | <distortion>
    camera-imu timeoffset = <value>
"""
import argparse
import json
import math

import numpy as np


def jpl_quat_to_rot(qx, qy, qz, qw):
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy + qz * qw), 2 * (qx * qz - qy * qw)],
        [2 * (qx * qy - qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz + qx * qw)],
        [2 * (qx * qz + qy * qw), 2 * (qy * qz - qx * qw), 1 - 2 * (qx**2 + qy**2)],
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quat", required=True,
                        help="JPL quaternion (x,y,z,w) from cam0 extrinsics")
    parser.add_argument("--p_IinC", required=True,
                        help="Translation from cam0 extrinsics")
    parser.add_argument("--intrinsics", required=True,
                        help="fx,fy,cx,cy from cam0 intrinsics")
    parser.add_argument("--distortion", required=True,
                        help="k1,k2,k3,k4 from cam0 intrinsics")
    parser.add_argument("--time-offset", type=float, default=0.0,
                        help="camera-imu timeoffset value")
    parser.add_argument("--save-json", type=str, default=None,
                        help="Save calibration to JSON file")
    args = parser.parse_args()

    q = np.array([float(x) for x in args.quat.split(",")])
    q = q / np.linalg.norm(q)
    qx, qy, qz, qw = q

    p_IinC = np.array([float(x) for x in args.p_IinC.split(",")])
    intrinsics = [float(x) for x in args.intrinsics.split(",")]
    distortion = [float(x) for x in args.distortion.split(",")]

    R_ItoC = jpl_quat_to_rot(qx, qy, qz, qw)
    R_CtoI = R_ItoC.T
    p_CinI = -R_CtoI @ p_IinC

    cam_z_in_imu = R_CtoI @ np.array([0, 0, 1])
    tilt = math.degrees(math.atan2(-cam_z_in_imu[2], cam_z_in_imu[0]))

    print("=" * 60)
    print("Camera tilt: {:.1f} deg below horizontal".format(tilt))
    print("=" * 60)
    print()
    print("kalibr_imucam_chain.yaml:")
    print()
    print("  T_imu_cam:")
    for i in range(3):
        print("    - [{:.10f}, {:.10f}, {:.10f}, {:.10f}]".format(
            R_CtoI[i, 0], R_CtoI[i, 1], R_CtoI[i, 2], p_CinI[i]))
    print("    - [0.0, 0.0, 0.0, 1.0]")
    print("  distortion_coeffs: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*distortion))
    print("  distortion_model: equidistant")
    print("  intrinsics: [{:.3f}, {:.3f}, {:.3f}, {:.3f}]".format(*intrinsics))
    print()
    print("kalibr_imu_chain.yaml:")
    print()
    print("  time_offset: {:.5f}".format(args.time_offset))

    if args.save_json:
        data = {
            "cam0_extrinsics_quat_jpl_xyzw": [float(qx), float(qy), float(qz), float(qw)],
            "cam0_extrinsics_p_IinC": p_IinC.tolist(),
            "cam0_intrinsics": intrinsics,
            "cam0_distortion_equidistant": distortion,
            "camera_imu_timeoffset": args.time_offset,
            "T_imu_cam_R_CtoI": R_CtoI.tolist(),
            "T_imu_cam_p_CinI": p_CinI.tolist(),
        }
        with open(args.save_json, "w") as f:
            json.dump(data, f, indent=2)
        print("\nSaved to {}".format(args.save_json))


if __name__ == "__main__":
    main()
