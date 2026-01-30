#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- locate XLSX robustly ----------
HERE = Path(__file__).parent
XLSX = HERE / "T200-Public-Performance-Data-10-20V-September-2019.xlsx"

# ---------- plugin settings (from gz SDF) ----------
MIN_PWM = 1100.0
MAX_PWM = 1900.0
MID_PWM = 0.5 * (MIN_PWM + MAX_PWM)  # 1500
HALF_RANGE = 0.5 * (MAX_PWM - MIN_PWM)  # 400

# Poly length in plugin: [a0, a1, a2, a3, a4, a5]
POLY_DEG = 5


@dataclass(frozen=True)
class FitResult:
    # coefficients in ascending power: a0 + a1*u + ... + a5*u^5 (plugin format)
    pos_coeff: np.ndarray
    neg_coeff: np.ndarray


def _find_sheet_name(xls: pd.ExcelFile, voltage_v: int) -> str:
    """
    Prefer a sheet that looks like '16V' / '16 V' / '16v' (and similarly for other voltages).
    Fall back to any sheet containing the voltage number and 'V'.
    """
    names = xls.sheet_names
    v = str(voltage_v)

    # exact-ish matches first
    for key in (f"{v}V", f"{v} V", f"{v}v", f"{v} v"):
        for n in names:
            if n.strip().lower() == key.strip().lower():
                return n

    # fuzzy
    for n in names:
        s = n.strip().lower().replace(" ", "")
        if v in s and "v" in s:
            return n

    raise ValueError(f"Could not find a {v}V sheet. Available sheets: {names}")


def _guess_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Return (pwm_col, thrust_col) guessing from common Blue Robotics naming.
    We want thrust in kgf and PWM in microseconds.
    """
    cols = list(df.columns)

    # Normalize: lowercase, strip, collapse spaces, remove parentheses, replace µ with u
    def norm(s: str) -> str:
        s = s.strip().lower().replace("µ", "u")
        s = s.replace("(", " ").replace(")", " ")
        s = " ".join(s.split())          # collapse whitespace
        s = s.replace(" ", "")           # remove all spaces for robust matching
        return s

    ncols = [norm(c) for c in cols]

    def find_any(pred) -> Optional[str]:
        for c, nc in zip(cols, ncols):
            if pred(nc):
                return c
        return None

    # PWM candidates (e.g. "pwm(us)")
    pwm_col = find_any(lambda s: "pwm" in s and ("us" in s or "u s" in s or "µs" in s)) or find_any(lambda s: "pwm" in s)
    if pwm_col is None:
        raise ValueError(f"Could not identify PWM column. Columns: {cols}")

    # Thrust/force in kgf candidates:
    # excel file has "Force (Kg f)" -> normalized becomes "forcekgf"
    thrust_col = (
        find_any(lambda s: ("force" in s or "thrust" in s) and "kgf" in s)
        or find_any(lambda s: "kgf" in s)
        or find_any(lambda s: "thrust" in s)
        or find_any(lambda s: "force" in s)
    )
    if thrust_col is None:
        raise ValueError(f"Could not identify thrust(kgf) column. Columns: {cols}")

    return pwm_col, thrust_col


def pwm_to_u(pwm: np.ndarray) -> np.ndarray:
    """
    Plugin-like bidirectional normalization based on min/max duty cycle:
        u = (pwm - 1500) / 400  in [-1, 1]
    """
    return (pwm - MID_PWM) / HALF_RANGE


def eval_poly_asc(coeff_asc: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Evaluate a0 + a1*u + ... in a numerically-stable way."""
    y = np.zeros_like(u, dtype=float)
    for a in coeff_asc[::-1]:
        y = y * u + a
    return y


def fit_forward_reverse(pwm: np.ndarray, thrust_kgf: np.ndarray) -> FitResult:
    """
    Fit two 5th order polynomials in terms of u_mag in [0,1]:
      - forward:  u_mag =  u, thrust_mag = +thrust
      - reverse:  u_mag = -u, thrust_mag = -thrust
    Return coefficients in ascending power (plugin format).
    """
    u = pwm_to_u(pwm)

    # mask and magnitudes
    pos_m = u >= 0
    neg_m = u <= 0

    u_pos = u[pos_m]
    t_pos = thrust_kgf[pos_m]

    u_neg_mag = -u[neg_m]
    t_neg_mag = -thrust_kgf[neg_m]  # make positive magnitude

    # Basic sanity: remove NaNs and weird entries
    def clean(x, y):
        ok = np.isfinite(x) & np.isfinite(y)
        return x[ok], y[ok]

    u_pos, t_pos = clean(u_pos, t_pos)
    u_neg_mag, t_neg_mag = clean(u_neg_mag, t_neg_mag)

    # Fit in Newtons (plugin force), but we keep kgf too for plotting.
    g0 = 9.80665
    y_pos_N = t_pos * g0
    y_neg_N = t_neg_mag * g0

    # Fit highest-degree-first then convert to ascending for plugin
    pos_desc = np.polyfit(u_pos, y_pos_N, deg=POLY_DEG)
    neg_desc = np.polyfit(u_neg_mag, y_neg_N, deg=POLY_DEG)

    pos_asc = pos_desc[::-1]
    neg_asc = neg_desc[::-1]
    return FitResult(pos_coeff=pos_asc, neg_coeff=neg_asc)


def main() -> None:
    # ---------- check input exists -----------
    if not XLSX.exists():
        print("[err] XLSX not found:", XLSX)
        print("[i] Put the file next to this script.")
        raise SystemExit(2)

    print("[i] Reading:", XLSX.as_posix())
    xls = pd.ExcelFile(XLSX)

    for V in (14, 16, 18, 20):
        sheet = _find_sheet_name(xls, V)
        out_png = HERE / f"T200_{V}V_thrust_polynomial_fit.png"

        print("\n[i] Using sheet:", sheet)

        df = pd.read_excel(XLSX, sheet_name=sheet)

        # Drop fully-empty rows
        df = df.dropna(how="all").copy()
        print("[i] Sheet rows:", len(df), "cols:", len(df.columns))

        pwm_col, thrust_col = _guess_columns(df)
        print("[i] PWM column:", pwm_col)
        print("[i] Thrust column:", thrust_col)

        pwm = pd.to_numeric(df[pwm_col], errors="coerce").to_numpy(dtype=float)
        thrust_kgf = pd.to_numeric(df[thrust_col], errors="coerce").to_numpy(dtype=float)

        # Keep only PWM within your plugin range, so the fit matches what you will command
        m = np.isfinite(pwm) & np.isfinite(thrust_kgf) & (pwm >= MIN_PWM) & (pwm <= MAX_PWM)
        pwm = pwm[m]
        thrust_kgf = thrust_kgf[m]

        # Sort by PWM for nicer plots
        order = np.argsort(pwm)
        pwm = pwm[order]
        thrust_kgf = thrust_kgf[order]

        # ---------- fit ----------
        fit = fit_forward_reverse(pwm, thrust_kgf)

        # Print coefficients in the exact SDF list format:
        #   positiveThrustPolynomial -> forward magnitude, Newtons
        #   negativeThrustPolynomial -> reverse magnitude, Newtons
        print(f"\n[ok] Fitted coefficients @ {V}V (plugin format: [a0, a1, a2, a3, a4, a5])")
        print("positiveThrustPolynomial =", np.array2string(fit.pos_coeff, separator=", "))
        print("negativeThrustPolynomial =", np.array2string(fit.neg_coeff, separator=", "))

        # ---------- plot on PWM axis in kgf ----------
        # Create dense PWM axis for lines
        pwm_grid = np.linspace(MIN_PWM, MAX_PWM, 801)
        u_grid = pwm_to_u(pwm_grid)

        # Evaluate piecewise (magnitudes, then apply sign to get kgf)
        g0 = 9.80665

        y_fit_N = np.zeros_like(u_grid, dtype=float)
        pos = u_grid >= 0
        neg = u_grid < 0

        # forward: + poly_pos(u)
        y_fit_N[pos] = eval_poly_asc(fit.pos_coeff, u_grid[pos])

        # reverse: - poly_neg(|u|)
        y_fit_N[neg] = -eval_poly_asc(fit.neg_coeff, -u_grid[neg])

        y_fit_kgf = y_fit_N / g0

        # Measured forward/reverse split
        is_fwd = pwm >= MID_PWM
        is_rev = pwm < MID_PWM

        # Colors
        c_meas_fwd = "#0b3d91"   # dark blue
        c_meas_rev = "#7fb3ff"   # light blue
        c_fit_fwd  = "#1b7f1b"   # green (darker)
        c_fit_rev  = "#6fdc6f"   # green (lighter)

        plt.figure()
        plt.scatter(pwm[is_fwd], thrust_kgf[is_fwd], s=18, color=c_meas_fwd, label="Measured (forward)")
        plt.scatter(pwm[is_rev], thrust_kgf[is_rev], s=18, color=c_meas_rev, label="Measured (reverse)")

        # Plot fitted lines split for clearer legend/coloring
        plt.plot(pwm_grid[pos], y_fit_kgf[pos], color=c_fit_fwd, linewidth=2.0, label="Fitted (forward)")
        plt.plot(pwm_grid[neg], y_fit_kgf[neg], color=c_fit_rev, linewidth=2.0, label="Fitted (reverse)")

        plt.xlim(MIN_PWM, MAX_PWM)
        plt.xlabel("PWM (µs)")
        plt.ylabel("Thrust (kgf)")
        plt.title(f"T200 Thrust Fit @ {V}V (fit domain: PWM {int(MIN_PWM)}–{int(MAX_PWM)})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        print("[ok] Saved plot:", out_png.as_posix())
        plt.show()


if __name__ == "__main__":
    main()