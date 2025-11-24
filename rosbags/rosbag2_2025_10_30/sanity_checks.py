#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- locate CSV robustly ----------
HERE = Path(__file__).parent
CSV = HERE / "rosbag2_2025_10_30-16_31_20" / "koopman_dataset_50Hz.csv"

if not CSV.exists():
    # Helpful diagnostics
    print("[err] CSV not found:", CSV)
    print("[i] Run this script from anywhere; it resolves paths relative to itself.")
    print("[i] Did you save the CSV to:", CSV.parent.as_posix(), "?")
    # List nearby files to debug
    maybe = list((HERE / "rosbag2_2025_10_30-16_31_20").glob("*.csv"))
    if maybe:
        print("[i] Found CSVs in folder:", [m.name for m in maybe])
    raise SystemExit(2)

print("[i] Reading:", CSV.as_posix())
df = pd.read_csv(CSV)

# ---------- helpers ----------
def euler_to_R_n2b(phi, theta, psi):
    c, s = np.cos, np.sin
    # NED frame uses right-handed x→forward, y→right, z→down
    Rz = np.array([[ c(psi),  s(psi), 0],
                   [-s(psi),  c(psi), 0],
                   [ 0,        0,      1]])
    Ry = np.array([[ c(theta), 0, -s(theta)],
                   [ 0,        1,  0],
                   [ s(theta), 0,  c(theta)]])
    Rx = np.array([[1, 0, 0],
                   [0, c(phi),  s(phi)],
                   [0,-s(phi),  c(phi)]])
    return Rx @ Ry @ Rz  # n→b for NED

# ---------- Check 1: kinematic consistency (ż vs rotated w) ----------
t = df["t"].to_numpy()
z = df["z"].to_numpy()
zdot_fd = np.gradient(z, t)

uvw = df[["u","v","w"]].to_numpy()
eul = df[["phi","theta","psi"]].to_numpy()
z_world_from_body = np.empty(len(df))
for i, ((u,v,w), (phi,th,psi)) in enumerate(zip(uvw, eul)):
    Rn2b = euler_to_R_n2b(phi, th, psi)
    Rb2n = Rn2b.T
    z_world_from_body[i] = (Rb2n @ np.array([u,v,w]))[2]

rmse = float(np.sqrt(np.mean((zdot_fd - z_world_from_body)**2)))
corr = float(np.corrcoef(zdot_fd, z_world_from_body)[0,1])
print(f"[ok] ż FD vs rotate(w): RMSE={rmse:.4f} m/s, corr={corr:.3f}")

# ---------- Check 2: actuator sign sanity ----------
U = df[[f"u{i}" for i in range(1,9)]].to_numpy()
corrs = [float(np.corrcoef(U[:, i], zdot_fd)[0,1]) for i in range(8)]
print("[ok] corr(u_i, ż):", [f"{c:.3f}" for c in corrs])

# ---------- Check 3: passive buoyancy drift ----------
near_zero = (df[[f"u{i}" for i in range(1,9)]].abs().max(axis=1) < 0.05)
if near_zero.any():
    dz = float(z[near_zero][-1] - z[near_zero][0])
    print(f"[ok] Δz during u≈0 segment: {dz:.3f} m")
else:
    print("[i] No long u≈0 segment found; skip buoyancy drift check.")