#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# ---------- locate CSV robustly ----------
HERE = Path(__file__).parent
CSV = HERE / "rosbag2_2025_11_06-manual" / "koopman_dataset_50Hz.csv"
OUT = CSV.with_name(CSV.stem + "_with_wrench.csv")

# ---------- check input exists -----------
if not CSV.exists():
    print("[err] CSV not found:", CSV)
    print("[i] Run this script from anywhere; it resolves paths relative to itself.")
    print("[i] Did you save the CSV to:", CSV.parent.as_posix(), "?")
    maybe = list(CSV.parent.glob("*.csv"))
    if maybe:
        print("[i] Found CSVs in folder:", [m.name for m in maybe])
    raise SystemExit(2)

print("[i] Reading:", CSV.as_posix())
df = pd.read_csv(CSV)

# ---------- rotor model ----------
@dataclass(frozen=True)
class Rotor:
    axis: np.ndarray
    pos: np.ndarray
    km: float = 0.0


def bluerov2_heavy_non_sim_rotors() -> List[Rotor]:
    return [
        Rotor(axis=np.array([ 1.0, -1.0,  0.0]), pos=np.array([ 0.14,  0.10, 0.06])),
        Rotor(axis=np.array([ 1.0,  1.0,  0.0]), pos=np.array([ 0.14, -0.10, 0.06])),
        Rotor(axis=np.array([ 1.0,  1.0,  0.0]), pos=np.array([-0.14,  0.10, 0.06])),
        Rotor(axis=np.array([ 1.0, -1.0,  0.0]), pos=np.array([-0.14, -0.10, 0.06])),
        Rotor(axis=np.array([ 0.0,  0.0, -1.0]), pos=np.array([ 0.12,  0.22, 0.00])),
        Rotor(axis=np.array([ 0.0,  0.0,  1.0]), pos=np.array([ 0.12, -0.22, 0.00])),
        Rotor(axis=np.array([ 0.0,  0.0,  1.0]), pos=np.array([-0.12,  0.22, 0.00])),
        Rotor(axis=np.array([ 0.0,  0.0, -1.0]), pos=np.array([-0.12, -0.22, 0.00])),
    ]


def effectiveness_matrix(rotors: List[Rotor], normalize_axes: bool = True) -> np.ndarray:
    E = np.zeros((6, len(rotors)), dtype=float)
    for i, rtr in enumerate(rotors):
        a = rtr.axis.astype(float)
        if normalize_axes:
            na = np.linalg.norm(a)
            if na > 0:
                a /= na
        r = rtr.pos.astype(float)

        E[0:3, i] = a
        E[3:6, i] = np.cross(r, a)
    return E


# ---------- actuator columns (auto) ----------
u_cols = [f"u{i}" for i in range(1, 9)]
if all(c in df.columns for c in u_cols):
    act_cols = u_cols
else:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 8:
        raise SystemExit("[err] Could not find 8 actuator columns.")
    act_cols = numeric_cols[-8:]

print("[i] Using actuator columns:", act_cols)

U = df[act_cols].to_numpy(dtype=float)
if np.nanmax(np.abs(U)) > 1.05:
    print("[warn] actuator values exceed ~[-1,1]. Proceeding anyway.")

# ---------- compute normalized wrench ----------
rotors = bluerov2_heavy_non_sim_rotors()
E = effectiveness_matrix(rotors, normalize_axes=True)
W = (E @ U.T).T  # (N,6)

# ---------- MODIFY DATAFRAME CONTENT ----------
df_out = df.drop(columns=act_cols)

df_out.insert(len(df_out.columns), "Fx_sp", W[:, 0])
df_out.insert(len(df_out.columns), "Fy_sp", W[:, 1])
df_out.insert(len(df_out.columns), "Fz_sp", W[:, 2])
df_out.insert(len(df_out.columns), "Tx_sp", W[:, 3])
df_out.insert(len(df_out.columns), "Ty_sp", W[:, 4])
df_out.insert(len(df_out.columns), "Tz_sp", W[:, 5])

df_out.to_csv(OUT, index=False)
print("[ok] Wrote:", OUT.as_posix())