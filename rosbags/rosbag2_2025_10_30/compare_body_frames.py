#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare mocap odometry twist vs PX4 body rates to determine body-frame sign/mapping.

It:
- Reads /mocap/itrl_rov_1/odom  (nav_msgs/Odometry)
- Reads PX4 gyro from:
    1) /itrl_rov_1/fmu/out/sensor_combined (px4_msgs/SensorCombined).gyro_rad[3], OR
    2) /itrl_rov_1/fmu/out/vehicle_odometry (px4_msgs/VehicleOdometry).angular_velocity[3]
- Time-aligns, then tests:
    - Whether mocap twist appears already in body frame or in parent/world (and needs rotation to body)
    - Which axis/sign flip best matches PX4 body NED (X fwd, Y right, Z down)

Outputs per-axis correlations and the recommended mapping.

Usage:
    python compare_body_frames.py /path/to/rosbag2_YYYY_MM_DD-...   # bag directory OR .db3
"""

from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd

from rosbags.highlevel import AnyReader
from rosbags.typesys import get_types_from_idl, get_types_from_msg

# ----------- USER: adjust only if your topic names differ -----------
MOCAP_ODOM = "/mocap/itrl_rov_1/odom"                   # nav_msgs/msg/Odometry
PX4_SC     = "/itrl_rov_1/fmu/out/sensor_combined"      # px4_msgs/msg/SensorCombined (gyro_rad)
PX4_VODOM  = "/itrl_rov_1/fmu/out/vehicle_odometry"     # px4_msgs/msg/VehicleOdometry (angular_velocity)

# Merge tolerance (seconds) for aligning streams
ALIGN_TOL = 0.02  # 20 ms is fine at ~100 Hz

# Where .msg/.idl live, if needed for px4_msgs types
def _custom_type_dirs(script_path: Path):
    # try alongside script in a 'types' folder
    return [script_path.parent / "types"]

# ----------------- math helpers -----------------
def quat_to_R_n2b(x, y, z, w):
    n = math.sqrt(x*x + y*y + z*z + w*w) or 1.0
    x, y, z, w = x/n, y/n, z/n, w/n
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)],
    ], dtype=float)

# ----------------- type registration -----------------
def register_custom_types(typestore, base_dirs):
    found = []
    for base in base_dirs:
        if not base.exists():
            continue
        for ext in (".msg", ".idl"):
            for path in base.rglob(f"*{ext}"):
                if path.parent.name.lower() != "msg" or not path.is_file():
                    continue
                pkg = path.parent.parent.name  # e.g. px4_msgs
                typename = f"{pkg}/msg/{path.stem}"
                try:
                    text = path.read_text(encoding="utf-8")
                    defs = get_types_from_msg(text, typename) if ext == ".msg" else get_types_from_idl(text, typename)
                    typestore.register(defs)
                    found.append(typename)
                except Exception as e:
                    print(f"[warn] Could not register {typename} from {path}: {e}")
    if found:
        print(f"[i] Registered custom types: {', '.join(sorted(set(found)))}")

# ----------------- main extraction -----------------
def read_streams(bag_path: Path):
    with AnyReader([bag_path]) as reader:
        register_custom_types(reader.typestore, _custom_type_dirs(Path(__file__)))
        conns = {c.topic: c for c in reader.connections}
        if MOCAP_ODOM not in conns:
            raise RuntimeError(f"Missing {MOCAP_ODOM}. Available: {sorted(conns.keys())}")

        # Time base
        t0_ns = None

        # --- mocap odom ---
        mo_rows = []  # t, q (x,y,z,w), ang (x,y,z), child_frame_id
        for c, ts, raw in reader.messages(connections=[conns[MOCAP_ODOM]]):
            if t0_ns is None:
                t0_ns = ts
            t = (ts - t0_ns) * 1e-9
            m = reader.deserialize(raw, c.msgtype)
            qx,qy,qz,qw = float(m.pose.pose.orientation.x), float(m.pose.pose.orientation.y), float(m.pose.pose.orientation.z), float(m.pose.pose.orientation.w)
            wx,wy,wz = float(m.twist.twist.angular.x), float(m.twist.twist.angular.y), float(m.twist.twist.angular.z)
            child = getattr(m, "child_frame_id", "")
            mo_rows.append((t, qx,qy,qz,qw, wx,wy,wz, child))

        df_mo = pd.DataFrame(mo_rows, columns=["t","qx","qy","qz","qw","wx","wy","wz","child"]).sort_values("t")
        # Decide if mocap twist is in body already (per REP-105, Odometry.twist is in child_frame)
        # We'll still test both "as_is" and "rotated parent->body".
        # Rotated-body-from-parent:
        W_parent = df_mo[["wx","wy","wz"]].to_numpy()
        Q = df_mo[["qx","qy","qz","qw"]].to_numpy()
        W_body_from_rot = []
        for (wx,wy,wz),(qx,qy,qz,qw) in zip(W_parent, Q):
            Rn2b = quat_to_R_n2b(qx,qy,qz,qw)
            # If twist was expressed in parent/world, body twist = R * w_parent
            W_body_from_rot.append((Rn2b @ np.array([wx,wy,wz])).tolist())
        W_body_from_rot = np.asarray(W_body_from_rot)

        # --- PX4 gyro: prefer SensorCombined, else VehicleOdometry ---
        px_rows = []
        if PX4_SC in conns:
            for c, ts, raw in reader.messages(connections=[conns[PX4_SC]]):
                if t0_ns is None: t0_ns = ts
                t = (ts - t0_ns) * 1e-9
                m = reader.deserialize(raw, c.msgtype)
                g = getattr(m, "gyro_rad", None)
                if g is None:
                    continue
                gx,gy,gz = float(g[0]), float(g[1]), float(g[2])
                px_rows.append((t, gx,gy,gz))
        elif PX4_VODOM in conns:
            for c, ts, raw in reader.messages(connections=[conns[PX4_VODOM]]):
                if t0_ns is None: t0_ns = ts
                t = (ts - t0_ns) * 1e-9
                m = reader.deserialize(raw, c.msgtype)
                av = getattr(m, "angular_velocity", None)
                if av is None:
                    continue
                gx,gy,gz = float(av[0]), float(av[1]), float(av[2])
                px_rows.append((t, gx,gy,gz))
        else:
            raise RuntimeError(f"Neither {PX4_SC} nor {PX4_VODOM} present. Available: {sorted(conns.keys())}")

        df_px = pd.DataFrame(px_rows, columns=["t","gx","gy","gz"]).sort_values("t")

    return df_mo, W_body_from_rot, df_px

# ----------------- scoring & decision -----------------
def correlate(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 1: a = a.reshape(-1,1)
    if b.ndim == 1: b = b.reshape(-1,1)
    a = a - np.nanmean(a, axis=0)
    b = b - np.nanmean(b, axis=0)
    den = np.sqrt(np.nanmean(a*a, axis=0) * np.nanmean(b*b, axis=0))
    den[den==0] = np.nan
    c = np.nanmean((a*b)/den, axis=0)
    return float(np.nanmean(c))

def main(bag_path: Path):
    print(f"[i] Opening: {bag_path}")
    df_mo, W_body_from_rot, df_px = read_streams(bag_path)

    # Two mocap candidates:
    #  A) Assume mocap twist is already body-frame:    W_mocap_body = [wx,wy,wz]
    #  B) Assume mocap twist is parent/world -> rotate to body:  W_body_from_rot
    W_mocap_as_is = df_mo[["wx","wy","wz"]].to_numpy()
    t_mo = df_mo["t"].to_numpy()
    t_px = df_px["t"].to_numpy()
    G_px = df_px[["gx","gy","gz"]].to_numpy()

    # Merge-asof to align times
    dfa = pd.DataFrame({"t": t_mo, "mx": W_mocap_as_is[:,0], "my": W_mocap_as_is[:,1], "mz": W_mocap_as_is[:,2]})
    dfr = pd.DataFrame({"t": t_mo, "rx": W_body_from_rot[:,0], "ry": W_body_from_rot[:,1], "rz": W_body_from_rot[:,2]})
    dfp = pd.DataFrame({"t": t_px, "gx": G_px[:,0], "gy": G_px[:,1], "gz": G_px[:,2]})

    for label, dmc in [("as_is", dfa), ("rotated", dfr)]:
        merged = pd.merge_asof(
            dmc.sort_values("t"),
            dfp.sort_values("t"),
            on="t", direction="nearest", tolerance=ALIGN_TOL
        ).dropna()

        if merged.empty:
            print(f"[warn] No overlap for mode={label}.")
            continue

        M = merged[["mx","my","mz"]].to_numpy() if label=="as_is" else merged[["rx","ry","rz"]].to_numpy()
        G = merged[["gx","gy","gz"]].to_numpy()

        # Candidate flips between ENU-like vs NED-like body:
        candidates = {
            "identity"      : np.diag([1, 1, 1]),
            "flip_yz"       : np.diag([1,-1,-1]),  # common ENU<->NED body
            "flip_xz"       : np.diag([-1,1,-1]),
            "flip_xy"       : np.diag([-1,-1,1]),
        }

        best = None
        for name,S in candidates.items():
            Mc = (M @ S.T)  # apply sign/axis flip in body
            # Per-axis corrs
            cx = correlate(Mc[:,0], G[:,0])
            cy = correlate(Mc[:,1], G[:,1])
            cz = correlate(Mc[:,2], G[:,2])
            score = abs(cx)+abs(cy)+abs(cz)
            if (best is None) or (score > best["score"]):
                best = {"mode":label, "flip":name, "S":S, "cx":cx, "cy":cy, "cz":cz, "score":score, "n":len(merged)}

        if best:
            print(f"\n[i] Candidate result: mocap_mode={best['mode']}  flip={best['flip']}  (N={best['n']})")
            print(f"    corr(p): {best['cx']:+.3f}   corr(q): {best['cy']:+.3f}   corr(r): {best['cz']:+.3f}")
            # Heuristic verdict
            if best["score"] > 2.0:  # ~ average |corr| > 0.66 across axes
                print("    → Strong match.")
            elif best["score"] > 1.2:
                print("    → Moderate match (some axis noisier).")
            else:
                print("    → Weak match; sensors may be desynced or filtered differently.")

    print("\nLegend:")
    print("  mocap_mode=as_is     : using Odometry.twist.angular directly")
    print("  mocap_mode=rotated   : assuming Odometry.twist was parent/world; rotated to body")
    print("  flip=flip_yz (diag[1,-1,-1]) is the common ENU↔NED body difference")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        # Default to last path structure
        here = Path(__file__).parent
        bag = here / "rosbag2_2025_10_30-16_31_20"
    else:
        bag = Path(sys.argv[1])
    if not bag.exists():
        raise FileNotFoundError(f"Bag path not found: {bag}")
    main(bag)