#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BlueROV2 rosbag2 -> CSV for model training (wrench version)

Outputs columns:
t,x,y,z,phi,theta,psi,u,v,w,p,q,r,Fx,Fy,Fz,Mx,My,Mz

Order of sources:
1) mocap odometry; else mocap pose+velocity; else PX4 vehicle_odometry (NED->ENU).
Velocities saved in BODY frame. Twist frame auto-inferred if ambiguous.
Inputs are BODY-FRAME WRENCH (Fx,Fy,Fz,Mx,My,Mz) computed from ActuatorMotors
using the PX4 60002 BlueROV2 Heavy control allocation geometry.
All streams merged and resampled to RESAMPLE_HZ, then saved.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import pandas as pd

from rosbags.highlevel import AnyReader
from rosbags.typesys import get_types_from_idl, get_types_from_msg

# ======================= USER SETTINGS =======================
BAG_PATH = Path(__file__).parent / "rosbag2_2025_11_06-manual"

# Primary mocap topics
MOCAP_ODOM = "/mocap/itrl_rov_1/odom"          # nav_msgs/msg/Odometry
MOCAP_POSE = "/mocap/itrl_rov_1/pose"          # geometry_msgs/msg/PoseStamped
MOCAP_VEL  = "/mocap/itrl_rov_1/velocity"      # geometry_msgs/msg/TwistStamped

# PX4 fallbacks and actuators
PX4_VODOM  = "/itrl_rov_1/fmu/out/vehicle_odometry"  # px4_msgs/msg/VehicleOdometry
PX4_MOTORS = "/itrl_rov_1/fmu/out/actuator_motors"   # px4_msgs/msg/ActuatorMotors

# Optional TF topics (unused in export)
TF_TOPIC = "/tf"
TF_STATIC_TOPIC = "/tf_static"

# Where custom .msg/.idl live (expected: types/px4_msgs/msg/*.idl or *.msg)
CUSTOM_TYPE_DIRS = [Path(__file__).parent / "types"]

# Output and resampling
RESAMPLE_HZ = 50
OUT_BASENAME = "koopman_wrench_dataset_50Hz"
WRITE_PARQUET = False

# Video
MAKE_VIDEO = True
VIDEO_PATH = "media/bag_topdown__manual_wrench.mp4"
VIDEO_SPEED = 6.0
VIDEO_MAX_FRAMES = 4000
VIDEO_TAIL_SECS = 12.0
VIDEO_DPI = 120
# ============================================================

# ================= WRENCH / ALLOCATION GEOMETRY ==============
# Geometry from PX4 airframe 60002_uuv_bluerov2_heavy
# Positions [m] in body frame (x forward, y right, z down) and
# unit thrust directions for each thruster (T1..T8).
#
# Rotor layout (matching CA_ROTOR*):
# 0: bow starboard horizontal
# 1: bow port    horizontal
# 2: stern starboard horizontal
# 3: stern port    horizontal
# 4: bow starboard vertical
# 5: bow port    vertical
# 6: stern starboard vertical
# 7: stern port    vertical

THRUSTER_POS = np.array([
    [ 0.14,  0.10, 0.06],   # T1
    [ 0.14, -0.10, 0.06],   # T2
    [-0.14,  0.10, 0.06],   # T3
    [-0.14, -0.10, 0.06],   # T4
    [ 0.12,  0.22, 0.00],   # T5
    [ 0.12, -0.22, 0.00],   # T6
    [-0.12,  0.22, 0.00],   # T7
    [-0.12, -0.22, 0.00],   # T8
], dtype=float)

THRUSTER_DIR = np.array([
    [ 1.0, -1.0,  0.0],   # T1: AX=1, AY=-1, AZ=0
    [ 1.0,  1.0,  0.0],   # T2
    [ 1.0,  1.0,  0.0],   # T3
    [ 1.0, -1.0,  0.0],   # T4
    [ 0.0,  0.0, -1.0],   # T5
    [ 0.0,  0.0,  1.0],   # T6
    [ 0.0,  0.0,  1.0],   # T7
    [ 0.0,  0.0, -1.0],   # T8
], dtype=float)

# Normalize horizontal thruster directions to unit length
for i in range(4):
    n = np.linalg.norm(THRUSTER_DIR[i])
    if n > 0:
        THRUSTER_DIR[i] /= n

def build_allocation_matrix() -> np.ndarray:
    """
    Build 6x8 allocation matrix T such that:
        tau = T @ F_vec
    where F_vec[i] is scalar thrust of thruster i (in arbitrary units),
    tau = [Fx,Fy,Fz,Mx,My,Mz] in body frame.

    We assume F_i is already signed: F_i > 0 thrusts along THRUSTER_DIR[i].
    """
    T = np.zeros((6, 8), dtype=float)
    for i in range(8):
        d = THRUSTER_DIR[i]
        r = THRUSTER_POS[i]
        # force contribution
        T[0:3, i] = d
        # moment contribution (r x F*d) = F * (r x d)
        T[3:6, i] = np.cross(r, d)
    return T

ALLOC_T = build_allocation_matrix()

def motors_to_signed_commands(raw: np.ndarray) -> np.ndarray:
    """
    Infer whether motors are in [0,1] or [-1,1] and return signed cmds in [-1,1].
    Uses simple heuristics on the array stats.
    """
    amin = float(np.nanmin(raw))
    amax = float(np.nanmax(raw))
    # Already looks like [-1,1]
    if amin < -0.01 and amax > 0.01:
        return np.clip(raw, -1.0, 1.0)
    # Looks like [0,1] (typical PX4 actuator topic)
    if amin >= -0.01 and amax <= 1.05:
        signed = 2.0 * raw - 1.0
        return np.clip(signed, -1.0, 1.0)
    # Fallback: just clip, assume approximate [-1,1]
    return np.clip(raw, -1.0, 1.0)

def motors_to_wrench(u8: np.ndarray) -> np.ndarray:
    """
    Convert 8 motor commands (dimensionless, raw) to 6D body wrench.

    Steps:
    - Infer signed motor commands in [-1,1].
    - Treat these as scalar thrusts F_i (units arbitrary but consistent).
    - Apply tau = T @ F.
    """
    u8 = np.asarray(u8, dtype=float).reshape(8)
    u_signed = motors_to_signed_commands(u8)
    # Here we simply take F_i = u_signed; if you later know the thrust curve,
    # replace this with F_i = k_t * sign(u) * |u|^alpha or similar.
    F_vec = u_signed
    tau = ALLOC_T @ F_vec
    return tau  # [Fx,Fy,Fz,Mx,My,Mz]
# ============================================================


# ===================== CUSTOM TYPE LOADER ====================
def register_custom_types(typestore):
    found = []
    for base in CUSTOM_TYPE_DIRS:
        if not base.exists():
            continue
        for ext in (".msg", ".idl"):
            for path in base.rglob(f"*{ext}"):
                if path.parent.name.lower() != "msg" or not path.is_file():
                    continue
                pkg = path.parent.parent.name
                typename = f"{pkg}/msg/{path.stem}"
                try:
                    text = path.read_text(encoding="utf-8")
                    defs = get_types_from_msg(text, typename) if ext == ".msg" else get_types_from_idl(text, typename)
                    typestore.register(defs)
                    found.append(typename)
                except Exception as e:
                    print(f"[warn] Could not register {typename} from {path}: {e}")
    if found:
        uniq = sorted(set(found))
        print(f"[i] Registered custom types: {', '.join(uniq)}")
    else:
        print("[i] No custom types found under CUSTOM_TYPE_DIRS.")
# ============================================================


# ======================== MATH HELPERS =======================
def quat_to_R_n2b(x, y, z, w):
    n = math.sqrt(x*x + y*y + z*z + w*w) or 1.0
    x, y, z, w = x/n, y/n, z/n, w/n
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=float)

def quat_to_euler_xyz(x, y, z, w):
    sinr_cosp = 2*(w*x + y*z);  cosr_cosp = 1 - 2*(x*x + y*y)
    phi = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2*(w*y - z*x)
    theta = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny_cosp = 2*(w*z + x*y);  cosy_cosp = 1 - 2*(y*y + z*z)
    psi = math.atan2(siny_cosp, cosy_cosp)
    return phi, theta, psi

def ned_to_enu_vec(v):
    x, y, z = v
    return np.array([y, x, -z], dtype=float)

def ned_quat_to_enu(qx, qy, qz, qw):
    phi, theta, psi = quat_to_euler_xyz(qx, qy, qz, qw)  # NED
    phi_e, theta_e, psi_e = theta, phi, -psi             # ENU
    cx, sx = math.cos(phi_e/2), math.sin(phi_e/2)
    cy, sy = math.cos(theta_e/2), math.sin(theta_e/2)
    cz, sz = math.cos(psi_e/2), math.sin(psi_e/2)
    qw2 = cx*cy*cz + sx*sy*sz
    qx2 = sx*cy*cz - cx*sy*sz
    qy2 = cx*sy*cz + sx*cy*sz
    qz2 = cx*cy*sz - sx*sy*cz
    return qx2, qy2, qz2, qw2
# ============================================================


# ==================== FRAME INFERENCE ========================
def infer_twist_frame(times, pos_world, lin_twist_msgs, quats):
    if len(times) < 5:
        return "parent"
    t = np.asarray(times)
    p = np.asarray(pos_world)
    v_fd = np.gradient(p, t, axis=0)
    v_msg_world = np.asarray(lin_twist_msgs)
    v_body_to_world = []
    for (qx,qy,qz,qw), vb in zip(quats, v_msg_world):
        Rn2b = quat_to_R_n2b(qx,qy,qz,qw)
        Rb2n = Rn2b.T
        v_body_to_world.append(Rb2n @ vb)
    v_body_to_world = np.asarray(v_body_to_world)
    def rmse(a,b):
        return float(np.sqrt(np.mean((a-b)**2)))
    e_parent = rmse(v_fd, v_msg_world)
    e_body   = rmse(v_fd, v_body_to_world)
    which = "parent" if e_parent <= e_body else "body"
    print(f"[i] Twist frame inference: {which} (RMSE world={e_parent:.4f}, body->world={e_body:.4f})")
    return which
# ============================================================


# ====================== DATA CONTAINERS ======================
@dataclass
class OdomRow:
    t: float
    x: float; y: float; z: float
    phi: float; theta: float; psi: float
    u: float; v: float; w: float
    p: float; q: float; r: float
# ============================================================


# ======================= BAG READING =========================
def read_bag():
    bag_path = Path(BAG_PATH)
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag not found: {bag_path}")

    odom_rows: list[OdomRow] = []
    wrench_rows: list[dict] = []

    with AnyReader([bag_path]) as reader:
        register_custom_types(reader.typestore)
        dbg_px4 = [k for k in reader.typestore.fielddefs.keys() if k.startswith("px4_msgs/msg/")]
        if dbg_px4:
            print(f"[i] px4 types available: {', '.join(sorted(dbg_px4)[:10])}")

        conns = {c.topic: c for c in reader.connections}
        t0_ns = None

        # ---------- Prefer mocap/odom ----------
        if MOCAP_ODOM in conns:
            conn = conns[MOCAP_ODOM]
            times, posW, linTw, quats, child_frames = [], [], [], [], []
            raw = []
            ok, bad = 0, 0
            for c, ts, data in reader.messages(connections=[conn]):
                if t0_ns is None: t0_ns = ts
                t = (ts - t0_ns) * 1e-9
                try:
                    m = reader.deserialize(data, c.msgtype)
                except Exception:
                    bad += 1
                    continue
                px,py,pz = float(m.pose.pose.position.x), float(m.pose.pose.position.y), float(m.pose.pose.position.z)
                qx,qy,qz,qw = float(m.pose.pose.orientation.x), float(m.pose.pose.orientation.y), float(m.pose.pose.orientation.z), float(m.pose.pose.orientation.w)
                vlx,vly,vlz = float(m.twist.twist.linear.x), float(m.twist.twist.linear.y), float(m.twist.twist.linear.z)
                vax,vay,vaz = float(m.twist.twist.angular.x), float(m.twist.twist.angular.y), float(m.twist.twist.angular.z)
                cf = getattr(m, "child_frame_id", "")
                times.append(t); posW.append([px,py,pz]); quats.append([qx,qy,qz,qw]); linTw.append([vlx,vly,vlz]); child_frames.append(cf)
                raw.append((t, px,py,pz, qx,qy,qz,qw, vlx,vly,vlz, vax,vay,vaz, cf))
                ok += 1
            if bad: print(f"[i] mocap/odom: skipped {bad} decode errors; kept {ok}")

            mode = None
            if child_frames and any(cf for cf in child_frames):
                cf0 = child_frames[len(child_frames)//2]
                if isinstance(cf0, str) and ("base_link" in cf0 or "body" in cf0 or "base" in cf0):
                    mode = "body"
            if mode is None:
                mode = infer_twist_frame(times, posW, linTw, quats)

            for t, px,py,pz, qx,qy,qz,qw, vlx,vly,vlz, vax,vay,vaz, _ in raw:
                phi, theta, psi = quat_to_euler_xyz(qx,qy,qz,qw)
                if mode == "parent":
                    R = quat_to_R_n2b(qx,qy,qz,qw)
                    u,v,w = (R @ np.array([vlx,vly,vlz])).tolist()
                    p,q,r = (R @ np.array([vax,vay,vaz])).tolist()
                else:
                    u,v,w = vlx,vly,vlz
                    p,q,r = vax,vay,vaz
                odom_rows.append(OdomRow(t, px,py,pz, phi,theta,psi, u,v,w, p,q,r))

        # ---------- Fallback: mocap pose + mocap vel ----------
        elif (MOCAP_POSE in conns) and (MOCAP_VEL in conns):
            pose_rows, vel_rows = [], []
            for c, ts, data in reader.messages(connections=[conns[MOCAP_POSE]]):
                if t0_ns is None: t0_ns = ts
                t = (ts - t0_ns) * 1e-9
                m = reader.deserialize(data, c.msgtype)
                pos = m.pose.position; ori = m.pose.orientation
                pose_rows.append({"t": t, "px": float(pos.x), "py": float(pos.y), "pz": float(pos.z),
                                  "qx": float(ori.x), "qy": float(ori.y), "qz": float(ori.z), "qw": float(ori.w)})
            for c, ts, data in reader.messages(connections=[conns[MOCAP_VEL]]):
                if t0_ns is None: t0_ns = ts
                t = (ts - t0_ns) * 1e-9
                m = reader.deserialize(data, c.msgtype)
                lin, ang = m.twist.linear, m.twist.angular
                vel_rows.append({"t": t, "vlx": float(lin.x), "vly": float(lin.y), "vlz": float(lin.z),
                                 "vax": float(ang.x), "vay": float(ang.y), "vaz": float(ang.z)})

            if pose_rows and vel_rows:
                dfp = pd.DataFrame(pose_rows).sort_values("t")
                dfv = pd.DataFrame(vel_rows).sort_values("t")
                tol = 0.5 / float(RESAMPLE_HZ)
                dfm = pd.merge_asof(dfp, dfv, on="t", direction="nearest", tolerance=tol).dropna()

                mode = infer_twist_frame(
                    dfm["t"].to_numpy(),
                    dfm[["px","py","pz"]].to_numpy(),
                    dfm[["vlx","vly","vlz"]].to_numpy(),
                    dfm[["qx","qy","qz","qw"]].to_numpy(),
                )

                for _, r in dfm.iterrows():
                    phi, theta, psi = quat_to_euler_xyz(r.qx, r.qy, r.qz, r.qw)
                    if mode == "parent":
                        R = quat_to_R_n2b(r.qx,r.qy,r.qz,r.qw)
                        u,v,w = (R @ np.array([r.vlx, r.vly, r.vlz])).tolist()
                        p,q,rr = (R @ np.array([r.vax, r.vay, r.vaz])).tolist()
                    else:
                        u,v,w = float(r.vlx), float(r.vly), float(r.vlz)
                        p,q,rr = float(r.vax), float(r.vay), float(r.vaz)
                    odom_rows.append(OdomRow(float(r.t), float(r.px), float(r.py), float(r.pz),
                                             phi, theta, psi, u,v,w, p,q,rr))

        # ---------- Fallback: PX4 VehicleOdometry (NED -> ENU) ----------
        elif PX4_VODOM in conns:
            conn = conns[PX4_VODOM]
            for c, ts, data in reader.messages(connections=[conn]):
                if t0_ns is None: t0_ns = ts
                t = (ts - t0_ns) * 1e-9
                m = reader.deserialize(data, c.msgtype)
                px_ned = [float(m.position[0]), float(m.position[1]), float(m.position[2])]
                qx,qy,qz,qw = float(m.q[0]), float(m.q[1]), float(m.q[2]), float(m.q[3])
                vlin_ned = [float(m.velocity[0]), float(m.velocity[1]), float(m.velocity[2])]
                vang_ned = [float(m.angular_velocity[0]), float(m.angular_velocity[1]), float(m.angular_velocity[2])]

                px,py,pz = ned_to_enu_vec(px_ned)
                qx_e, qy_e, qz_e, qw_e = ned_quat_to_enu(qx,qy,qz,qw)
                phi, theta, psi = quat_to_euler_xyz(qx_e, qy_e, qz_e, qw_e)

                vlin_enu = ned_to_enu_vec(vlin_ned)
                vang_enu = ned_to_enu_vec(vang_ned)

                R = quat_to_R_n2b(qx_e,qy_e,qz_e,qw_e)
                u,v,w = (R @ vlin_enu).tolist()
                p,q,r = (R @ vang_enu).tolist()

                odom_rows.append(OdomRow(t, px,py,pz, phi,theta,psi, u,v,w, p,q,r))

        else:
            avail = sorted({c.topic for c in reader.connections})
            raise RuntimeError("No pose/odometry stream found. "
                               f"Looked for {MOCAP_ODOM}, {MOCAP_POSE}+{MOCAP_VEL}, {PX4_VODOM}. "
                               f"Available: {avail}")

        # ---------- Wrench from PX4 ActuatorMotors ----------
        if PX4_MOTORS in conns:
            conn = conns[PX4_MOTORS]
            raw_vals = []
            for c, ts, data in reader.messages(connections=[conn]):
                t = (ts - (t0_ns if t0_ns is not None else ts)) * 1e-9
                m = reader.deserialize(data, c.msgtype)
                arr = list(m.control) if hasattr(m, "control") else []
                if not arr:
                    continue
                raw_vals.append((t, np.asarray(arr[:8], dtype=float)))

            if raw_vals:
                mat = np.vstack([a for _, a in raw_vals])
                amin, amax = float(np.nanmin(mat)), float(np.nanmax(mat))
                print(f"[i] Actuator raw range: [{amin:.3f}, {amax:.3f}]")

                # For stats, interpret signed commands
                signed_mat = motors_to_signed_commands(mat)
                print("[i] Signed motor command stats (inferred [-1,1]):")
                print(pd.DataFrame(
                    signed_mat,
                    columns=[f"m{i+1}" for i in range(8)]
                ).agg(['min','median','max']).to_string())

                for t, a in raw_vals:
                    tau = motors_to_wrench(a)  # [Fx,Fy,Fz,Mx,My,Mz]
                    wrench_rows.append({
                        "t": t,
                        "Fx": float(tau[0]),
                        "Fy": float(tau[1]),
                        "Fz": float(tau[2]),
                        "Mx": float(tau[3]),
                        "My": float(tau[4]),
                        "Mz": float(tau[5]),
                    })
            else:
                print("[i] No actuator samples decoded.")
        else:
            print("[i] Actuator topic not present.")

    df_odom = pd.DataFrame([r.__dict__ for r in odom_rows]).sort_values("t").reset_index(drop=True)
    df_wrench  = pd.DataFrame(wrench_rows).sort_values("t").reset_index(drop=True) if wrench_rows else None
    return df_odom, df_wrench
# ============================================================


# ================== RESAMPLE, MERGE, SAVE ====================
def resample_and_join(df_odom: pd.DataFrame, df_wrench: pd.DataFrame | None) -> pd.DataFrame:
    if df_odom.empty:
        raise RuntimeError("No odometry rows.")

    # Unwrap Euler before resample
    for ang in ("phi", "theta", "psi"):
        df_odom[ang] = np.unwrap(df_odom[ang].to_numpy())

    if RESAMPLE_HZ and RESAMPLE_HZ > 0:
        idx = pd.to_timedelta(df_odom["t"], unit="s")
        idx.name = "time"
        step = pd.Timedelta(seconds=1.0 / float(RESAMPLE_HZ))
        df_odom = (
            df_odom.set_index(idx)
                   .drop(columns=["t"])
                   .resample(step).mean()
                   .interpolate(method="time")
                   .reset_index(names="time")
        )
        df_odom["t"] = df_odom["time"].dt.total_seconds()
        df_odom = df_odom.drop(columns=["time"])

    wrench_cols = ["Fx","Fy","Fz","Mx","My","Mz"]

    # Merge wrench by nearest time
    if df_wrench is not None and not df_wrench.empty:
        tol = 0.5 / float(RESAMPLE_HZ if RESAMPLE_HZ else 50.0)
        merged = pd.merge_asof(
            df_odom[["t"]].sort_values("t"),
            df_wrench.sort_values("t"),
            on="t", direction="nearest", tolerance=tol
        )
        df = pd.concat([df_odom, merged.drop(columns=["t"])], axis=1)
    else:
        df = df_odom.copy()
        for c in wrench_cols:
            df[c] = 0.0

    # Fill gaps (if any)
    df[wrench_cols] = (
        df[wrench_cols]
        .apply(pd.to_numeric, errors="coerce")
        .ffill().bfill()
        .fillna(0.0)
    )

    state_cols = ["x","y","z","phi","theta","psi","u","v","w","p","q","r"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=state_cols).reset_index(drop=True)
    cols = ["t"] + state_cols + wrench_cols
    return df[cols]

def save_outputs(df: pd.DataFrame, out_base: Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_base.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"[ok] Saved: {csv_path}")
    if WRITE_PARQUET:
        try:
            pq_path = out_base.with_suffix(".parquet")
            df.to_parquet(pq_path, index=False)
            print(f"[ok] Saved: {pq_path}")
        except Exception as e:
            print(f"[warn] Parquet not written: {e}")
# ============================================================


# ===================== QUICKLOOK VIDEO =======================
def make_topdown_video(df: pd.DataFrame, path: str):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.patches import FancyArrowPatch
    except Exception as e:
        print(f"[warn] Matplotlib not available ({e}); skipping video.")
        return

    N = len(df)
    if N < 2:
        print("[warn] Not enough samples for video.")
        return

    stride = max(1, int(math.ceil(N / max(1, VIDEO_MAX_FRAMES))))
    dfv = df.iloc[::stride].reset_index(drop=True).copy()
    T = len(dfv)
    dt = float(np.median(np.diff(dfv["t"]))) if T > 1 else 0.05

    xs = dfv["x"].to_numpy(); ys = dfv["y"].to_numpy(); psis = dfv["psi"].to_numpy()
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    pad_x = 0.10 * max(1e-6, x_max - x_min); pad_y = 0.10 * max(1e-6, y_max - y_min)
    x_lim = (x_min - pad_x, x_max + pad_x); y_lim = (y_min - pad_y, y_max + pad_y)

    tail = max(1, int(VIDEO_TAIL_SECS / max(dt, 1e-9)))

    fig, ax = plt.subplots(figsize=(6,6), dpi=VIDEO_DPI)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*x_lim); ax.set_ylim(*y_lim)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Top-down trajectory")

    path_line, = ax.plot([], [], lw=2)
    dot, = ax.plot([], [], "o", ms=6)
    head = FancyArrowPatch((0,0), (0,0), arrowstyle='-|>', mutation_scale=14, lw=2, zorder=5)
    ax.add_patch(head)
    tt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    head_len = 0.08 * max(x_max - x_min, y_max - y_min) if (x_max > x_min or y_max > y_min) else 1.0

    def init():
        path_line.set_data([], [])
        dot.set_data([], [])
        head.set_positions((0,0), (0,0))
        tt.set_text("")
        return (path_line, dot, head, tt)

    def update(i):
        s = max(0, i - tail)
        path_line.set_data(xs[s:i+1], ys[s:i+1])
        dot.set_data([xs[i]], [ys[i]])
        x0, y0, psi = xs[i], ys[i], psis[i]
        x1, y1 = x0 + head_len*math.cos(psi), y0 + head_len*math.sin(psi)
        head.set_positions((x0, y0), (x1, y1))
        tt.set_text(f"t = {dfv['t'].iloc[i]:.2f} s\nz = {dfv['z'].iloc[i]:.2f} m")
        return (path_line, dot, head, tt)

    interval_ms = int(max(1, 1000.0 * dt / max(VIDEO_SPEED, 1e-6)))
    ani = FuncAnimation(fig, update, frames=T, init_func=init, blit=True, interval=interval_ms)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        from matplotlib.animation import FFMpegWriter, PillowWriter
        if path.lower().endswith(".gif"):
            fps = int(round(1.0 / max(dt, 1e-6) * VIDEO_SPEED))
            ani.save(path, writer=PillowWriter(fps=fps), dpi=VIDEO_DPI)
        else:
            fps = int(round(1.0 / max(dt, 1e-6) * VIDEO_SPEED))
            ani.save(path, writer=FFMpegWriter(fps=fps), dpi=VIDEO_DPI)
        print(f"[ok] Video saved -> {path}  (frames: {T}, stride: {stride}x)")
    except Exception as e:
        print(f"[warn] Could not save video ({e}). Try .gif or install ffmpeg.")
    finally:
        plt.close(fig)
# ============================================================


# ============================ MAIN ===========================
def main():
    print(f"[i] Opening bag at: {BAG_PATH}")
    df_odom, df_wrench = read_bag()
    if len(df_odom) < 2:
        raise RuntimeError("Not enough odometry samples.")

    dt_med = np.median(np.diff(df_odom['t'].to_numpy()))
    hz = 1.0 / max(dt_med, 1e-6)
    print(f"[i] Odom samples: {len(df_odom)} | median dt ≈ {dt_med:.4f}s (~{hz:.1f} Hz)")

    out_base = (Path(BAG_PATH) if Path(BAG_PATH).is_dir() else Path(BAG_PATH).parent) / OUT_BASENAME
    df_out = resample_and_join(df_odom, df_wrench)
    save_outputs(df_out, out_base)

    if MAKE_VIDEO:
        media_dir = (Path(BAG_PATH) / ".." / ".." / "media").resolve()
        make_topdown_video(df_out, str(media_dir / Path(VIDEO_PATH).name))

if __name__ == "__main__":
    main()