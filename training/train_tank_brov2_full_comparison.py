#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train & compare 4 models for BlueROV2 Heavy dynamics on a recorded dataset:

  1) Koopman EDMDc (12x8)
  2) Physics-based Fossen model (BlueROV2 6-DOF)
  3) Learned Double Integrator (DI)
  4) PINc (physics-informed residual DNN on reduced state)

Outputs:
  - Endpoint RMSE for H in {1, 10, 100}
  - Training / metric / rollout timings for all models
  - 2x3 top-view animation (TRUE + 4 models)
  - 2D static figure (TRUE + 4 models, x-y, with depths printed) for LaTeX
"""

import os
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from Koopman.koopmanEDMDc import KoopmanEDMDc
from fossen.BlueROV2 import BlueROV2
from fossen.bluerov_torch import bluerov_compute

# ------------------------------------------------------------------
#  Config
# ------------------------------------------------------------------
DATASET_NAME = "koopman_dataset_50Hz.csv"
TRAIN_SPLIT = 0.80
N_RBFS = 500
GAMMA = 3.0
RIDGE = 1e-1
OPEN_LOOP_STEPS = 500         # steps for animation / static plot
PLOT_FIG_SECONDS = 10.0       # ~10s of open-loop rollout for 2D figure

# PINc config
PINc_HIDDEN = [64, 64, 64, 64]
PINc_EPOCHS = 1000
PINc_BATCH = 256
PINc_LR = 3e-3
PINc_ROLLOUT_STEPS = 10
PINc_USE_PHYSICS = True
PINc_USE_ROLLOUT = True
PINc_CKPT = Path("models") / "pinc_best.pt"


# ------------------------------------------------------------------
#  Helpers: IO, dataset, metrics
# ------------------------------------------------------------------
def find_project_root(start: Path) -> Path:
    """
    Walk up from 'start' until we find a folder that contains 'rosbags'.
    Fallback to 'start' if not found.
    """
    p = start.resolve()
    for q in [p, *p.parents]:
        if (q / "rosbags").exists():
            return q
    return p


def find_latest_csv(root: Path, name: str) -> Path:
    cands = list(root.rglob(name))
    if not cands:
        raise FileNotFoundError(f"Could not find any '{name}' under: {root}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def load_dataset(csv_path: Path):
    print(f"[i] Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    STATE_COLS = ["x", "y", "z",
                  "phi", "theta", "psi",
                  "u", "v", "w",
                  "p", "q", "r"]
    INPUT_COLS = [f"u{i}" for i in range(1, 9)]

    for c in STATE_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing state column: {c}")
    for c in INPUT_COLS:
        if c not in df.columns:
            df[c] = 0.0

    if "t" not in df.columns:
        raise ValueError("CSV must contain a 't' time column.")
    df = df.sort_values("t").drop_duplicates(subset="t")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=STATE_COLS)

    X = df[STATE_COLS].to_numpy(float)
    U = df[INPUT_COLS].to_numpy(float)
    t = df["t"].to_numpy(float)
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.05

    print(f"[i] Samples: {len(df)} | median dt ≈ {dt:.5f}s (~{1.0 / max(dt, 1e-9):.2f} Hz)")
    return X, U, dt


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ------------------------------------------------------------------
#  Animation (top view, TRUE + 4 models)
# ------------------------------------------------------------------
def animate_xy_five(
    true_traj: np.ndarray,
    koop_traj: np.ndarray,
    fossen_traj: np.ndarray,
    di_traj: np.ndarray,
    pinc_traj: np.ndarray,
    dt: float,
    save_path: str | None = None,
    title: str = "Recorded CSV: True vs. 4 Models (top view)",
    tail_secs: float = 10.0,
    speed: float = 1.0,
    dpi: int = 130,
):
    """
    2x3 top-view animation:

        TRUE  |  KOOPMAN  |  FOSSEN
        DI    |  PINc     |  (unused)

    All trajectories must be shape (T, >=6) and same length.
    """
    assert true_traj.shape == koop_traj.shape == fossen_traj.shape == di_traj.shape == pinc_traj.shape
    T, n = true_traj.shape
    assert n >= 6, "Expected state with psi at index 5."

    xs = np.concatenate([
        true_traj[:, 0],
        koop_traj[:, 0],
        fossen_traj[:, 0],
        di_traj[:, 0],
        pinc_traj[:, 0],
    ])
    ys = np.concatenate([
        true_traj[:, 1],
        koop_traj[:, 1],
        fossen_traj[:, 1],
        di_traj[:, 1],
        pinc_traj[:, 1],
    ])
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    pad_x = 0.10 * max(1e-6, x_max - x_min)
    pad_y = 0.10 * max(1e-6, y_max - y_min)
    x_lim = (x_min - pad_x, x_max + pad_x)
    y_lim = (y_min - pad_y, y_max + pad_y)

    tail = max(1, int(tail_secs / max(dt, 1e-9)))

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), dpi=dpi, constrained_layout=True)
    fig.suptitle(title)

    axs_flat = axs.ravel()
    panels = [
        (axs_flat[0], "TRUE (Recorded)",      true_traj,  "C0"),
        (axs_flat[1], "KOOPMAN",              koop_traj,  "C1"),
        (axs_flat[2], "FOSSEN (BlueROV2)",    fossen_traj,"C2"),
        (axs_flat[3], "DOUBLE INTEGRATOR",    di_traj,    "C3"),
        (axs_flat[4], "PINc (ResDNN)",        pinc_traj,  "C4"),
    ]
    axs_flat[5].axis("off")

    artists = []
    for ax, name, traj, color in panels:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(name)

        path, = ax.plot([], [], lw=2, alpha=0.9, color=color)
        dot,  = ax.plot([], [], "o", ms=6, color=color)
        arrow = FancyArrowPatch((0, 0), (0, 0),
                                arrowstyle='-|>', mutation_scale=12,
                                lw=2, color=color, zorder=5)
        ax.add_patch(arrow)
        txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
        artists.append((traj, path, dot, arrow, txt))

    head_len = 0.1 * max(x_max - x_min, y_max - y_min) if (x_max > x_min or y_max > y_min) else 1.0

    def init():
        outs = []
        for traj, path, dot, arrow, txt in artists:
            path.set_data([], [])
            dot.set_data([], [])
            arrow.set_positions((0, 0), (0, 0))
            txt.set_text("")
            outs.extend([path, dot, arrow, txt])
        return tuple(outs)

    def update(i):
        outs = []
        s = max(0, i - tail)
        for traj, path, dot, arrow, txt in artists:
            x = traj[:, 0]
            y = traj[:, 1]
            z = traj[:, 2]
            psi = traj[:, 5]

            path.set_data(x[s:i+1], y[s:i+1])
            dot.set_data([x[i]], [y[i]])

            x0, y0, yaw = x[i], y[i], psi[i]
            x1 = x0 + head_len * math.cos(yaw)
            y1 = y0 + head_len * math.sin(yaw)
            arrow.set_positions((x0, y0), (x1, y1))

            txt.set_text(f"t = {i * dt:5.2f} s\nz = {z[i]:.2f} m")
            outs.extend([path, dot, arrow, txt])
        return tuple(outs)

    interval_ms = int(max(1, 1000.0 * dt / max(speed, 1e-6)))
    ani = FuncAnimation(fig, update, frames=T, init_func=init,
                        blit=True, interval=interval_ms)

    if save_path is None:
        plt.show()
    else:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        try:
            if save_path.lower().endswith(".gif"):
                from matplotlib.animation import PillowWriter
                ani.save(save_path, writer=PillowWriter(fps=int(round(1.0/dt*speed))), dpi=dpi)
            else:
                from matplotlib.animation import FFMpegWriter
                ani.save(save_path, writer=FFMpegWriter(fps=int(round(1.0/dt*speed))), dpi=dpi)
            print(f"[ok] Animation saved -> {save_path}")
        except Exception as e:
            print(f"[warn] Could not save animation ({e}). Falling back to showing it.")
            plt.show()

    plt.close(fig)
    return ani


# ------------------------------------------------------------------
#  2D static figure for LaTeX (TRUE + 4 models, x–y with depths printed)
# ------------------------------------------------------------------
def plot_2d_trajectories_with_depth(
    true_traj: np.ndarray,
    koop_traj: np.ndarray,
    fossen_traj: np.ndarray,
    di_traj: np.ndarray,
    pinc_traj: np.ndarray,
    dt: float,
    seconds: float,
    save_path: str = "media/true_vs_4models_2D.png",
):
    """
    2D x–y plot of first ~`seconds` of open-loop rollout for TRUE + 4 models.
    Depth (z) and time printed bottom-right.
    Arrows at end of each trajectory indicate direction.
    """

    Path(save_path).parent.mkdir(exist_ok=True, parents=True)

    max_steps = int(seconds / max(dt, 1e-9))
    horizon = min(
        max_steps,
        true_traj.shape[0],
        koop_traj.shape[0],
        fossen_traj.shape[0],
        di_traj.shape[0],
        pinc_traj.shape[0],
    )

    if horizon < 2:
        print("[warn] Not enough steps for 2D plot.")
        return

    # Slice
    X_true = true_traj[:horizon]
    X_k    = koop_traj[:horizon]
    X_f    = fossen_traj[:horizon]
    X_d    = di_traj[:horizon]
    X_p    = pinc_traj[:horizon]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    # Thicker lines
    line_true, = ax.plot(X_true[:, 0], X_true[:, 1], label="True",    linestyle="-",  linewidth=2.5)
    line_k,    = ax.plot(X_k[:, 0],    X_k[:, 1],    label="Koopman", linestyle="--", linewidth=2.5)
    line_f,    = ax.plot(X_f[:, 0],    X_f[:, 1],    label="Fossen",  linestyle="-.", linewidth=2.5)
    line_d,    = ax.plot(X_d[:, 0],    X_d[:, 1],    label="DI",      linestyle=":",  linewidth=2.5)
    line_p,    = ax.plot(X_p[:, 0],    X_p[:, 1],    label="PINc",    linestyle="-.", linewidth=2.5)

    # Arrow size based on trajectory spread
    xs = np.concatenate([X_true[:,0], X_k[:,0], X_f[:,0], X_d[:,0], X_p[:,0]])
    ys = np.concatenate([X_true[:,1], X_k[:,1], X_f[:,1], X_d[:,1], X_p[:,1]])
    span = max(1e-6, max(xs.max() - xs.min(), ys.max() - ys.min()))
    head_len = 0.07 * span

    colors = [
        line_true.get_color(),
        line_k.get_color(),
        line_f.get_color(),
        line_d.get_color(),
        line_p.get_color(),
    ]
    trajs = [X_true, X_k, X_f, X_d, X_p]

    # Arrows at the end
    for traj, color in zip(trajs, colors):
        x_end, y_end = traj[-1, 0], traj[-1, 1]
        psi_end = traj[-1, 5]
        x_head = x_end + head_len * math.cos(psi_end)
        y_head = y_end + head_len * math.sin(psi_end)

        ax.annotate(
            "",
            xy=(x_head, y_head),
            xytext=(x_end, y_end),
            arrowprops=dict(arrowstyle="->", lw=2.0, color=color),
        )

    # Depth + time in bottom-right corner
    T = horizon
    txt = (
        f"t ≈ {(T-1)*dt:5.2f} s\n"
        f"z_true = {X_true[-1,2]:.2f} m\n"
        f"z_K    = {X_k[-1,2]:.2f} m\n"
        f"z_F    = {X_f[-1,2]:.2f} m\n"
        f"z_DI   = {X_d[-1,2]:.2f} m\n"
        f"z_PINc = {X_p[-1,2]:.2f} m"
    )
    ax.text(
        0.98, 0.02, txt,
        transform=ax.transAxes,
        va="bottom", ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.25)
    )

    ax.set_xlabel("x [m]", fontsize=11)
    ax.set_ylabel("y [m]", fontsize=11)
    ax.set_title(f"Open-loop rollout (~{seconds:.1f}s, top view)", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    # Larger legend font
    ax.legend(loc="best", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] 2D trajectory figure saved -> {save_path}")


# ------------------------------------------------------------------
#  Fossen (BlueROV2) model
# ------------------------------------------------------------------
def simulate_physics(x0: np.ndarray, U_seq: np.ndarray, dt: float, rov: BlueROV2):
    """
    Explicit-Euler rollout of the BlueROV2 model using recorded inputs.
    Returns trajectory of shape (len(U_seq)+1, 12).
    """
    H = len(U_seq)
    traj = np.zeros((H + 1, x0.shape[0]))
    traj[0] = x0
    x = x0.copy()
    for k in range(H):
        dx = rov.dynamics(x, U_seq[k], dt)
        x = x + dt * dx
        traj[k + 1] = x
    return traj


def multistep_rmse_endpoint_physics(X_test: np.ndarray,
                                    U_test: np.ndarray,
                                    H: int,
                                    dt: float) -> float:
    T = len(X_test)
    n_states = X_test.shape[1]
    n_start = T - H
    if n_start <= 0:
        return float("nan")

    rov = BlueROV2(dt=dt)
    se_total = 0.0
    for k in range(n_start):
        x0 = X_test[k]
        U_seq = U_test[k:k+H]
        x_end = simulate_physics(x0, U_seq, dt, rov)[-1]
        err = x_end - X_test[k + H]
        se_total += float(np.dot(err, err))
    return float(np.sqrt(se_total / (n_start * n_states)))


# ------------------------------------------------------------------
#  Double Integrator model
# ------------------------------------------------------------------
def _euler_to_R_b2n(phi, theta, psi):
    """Rotation body->world using Z(psi) Y(theta) X(phi)."""
    cph, sph = math.cos(phi), math.sin(phi)
    cth, sth = math.cos(theta), math.sin(theta)
    cps, sps = math.cos(psi), math.sin(psi)
    Rz = np.array([[cps, -sps, 0.0],
                   [sps,  cps, 0.0],
                   [0.0,  0.0,  1.0]])
    Ry = np.array([[cth, 0.0, sth],
                   [0.0, 1.0, 0.0],
                   [-sth, 0.0, cth]])
    Rx = np.array([[1.0, 0.0,  0.0],
                   [0.0, cph, -sph],
                   [0.0, sph,  cph]])
    return Rz @ Ry @ Rx


def estimate_di_gains(X_train: np.ndarray, U_train: np.ndarray,
                      dt: float, ridge: float = 1e-3):
    """
    Learn linear maps:
        dv_body ≈ U @ K_lin   (K_lin: 8x3)
        dw_body ≈ U @ K_ang   (K_ang: 8x3)
    using forward differences on [u,v,w,p,q,r].
    """
    V = X_train[:, 6:9]    # body linear velocities
    W = X_train[:, 9:12]   # body angular rates
    dV = (V[1:] - V[:-1]) / max(dt, 1e-9)   # (N-1,3)
    dW = (W[1:] - W[:-1]) / max(dt, 1e-9)   # (N-1,3)
    G  = U_train[:-1]                       # (N-1,8)

    GTG = G.T @ G
    I = np.eye(GTG.shape[0])
    K_lin = np.linalg.solve(GTG + ridge * I, G.T @ dV)   # (8,3)
    K_ang = np.linalg.solve(GTG + ridge * I, G.T @ dW)   # (8,3)
    return K_lin, K_ang


def simulate_double_integrator(x0: np.ndarray,
                               U_seq: np.ndarray,
                               dt: float,
                               K_lin: np.ndarray,
                               K_ang: np.ndarray):
    """
    Discrete DI in body frame:
      v_{k+1} = v_k + dt * (U_k @ K_lin)
      w_{k+1} = w_k + dt * (U_k @ K_ang)
      p_{k+1} = p_k + dt * w_k  (small-angle approx)
      x_{k+1} = x_k + dt * (R_b2n(phi,theta,psi) @ v_k)
    """
    H = len(U_seq)
    traj = np.zeros((H + 1, x0.shape[0]))
    traj[0] = x = x0.copy()

    for k in range(H):
        pos = x[0:3]
        ang = x[3:6]
        v = x[6:9]
        w = x[9:12]

        a_body = U_seq[k] @ K_lin   # (3,)
        alpha  = U_seq[k] @ K_ang   # (3,)

        v_next = v + dt * a_body
        w_next = w + dt * alpha

        phi, theta, psi = ang
        Rb2n = _euler_to_R_b2n(phi, theta, psi)
        pos_next = pos + dt * (Rb2n @ v)

        ang_next = ang + dt * w     # small-angle approx

        x_next = np.zeros_like(x)
        x_next[0:3] = pos_next
        x_next[3:6] = ang_next
        x_next[6:9] = v_next
        x_next[9:12] = w_next

        traj[k + 1] = x = x_next

    return traj


def multistep_rmse_endpoint_di(X_test: np.ndarray,
                               U_test: np.ndarray,
                               H: int,
                               dt: float,
                               K_lin: np.ndarray,
                               K_ang: np.ndarray) -> float:
    T = len(X_test)
    n_states = X_test.shape[1]
    n_start = T - H
    if n_start <= 0:
        return float("nan")

    se_total = 0.0
    for k in range(n_start):
        x0 = X_test[k]
        U_seq = U_test[k:k+H]
        x_end = simulate_double_integrator(x0, U_seq, dt, K_lin, K_ang)[-1]
        err = x_end - X_test[k + H]
        se_total += float(np.dot(err, err))
    return float(np.sqrt(se_total / (n_start * n_states)))


# ------------------------------------------------------------------
#  PINc model: 12D <-> 9D, thruster map, network, training, rollout
# ------------------------------------------------------------------
def thrusters_to_body_wrenches(U8_row: np.ndarray,
                               dt: float,
                               old_model_obj: BlueROV2) -> np.ndarray:
    """
    Use the 6-DOF thruster map to compute body wrenches.
    Returns [X, Y, Z, Mz] from tau[0], tau[1], tau[2], tau[5]
    """
    tau6 = old_model_obj.compute_thruster_forces(U8_row, dt)  # [Fx, Fy, Fz, K, M, N]
    return np.array([tau6[0], tau6[1], tau6[2], tau6[5]], dtype=float)


def dataset12_to_9(x12: np.ndarray) -> np.ndarray:
    """
    [x,y,z,phi,theta,psi,u,v,w,p,q,r]
      -> [x,y,z,cosψ,sinψ,u,v,w,r]
    """
    cospsi, sinpsi = math.cos(x12[5]), math.sin(x12[5])
    return np.array([
        x12[0], x12[1], x12[2],
        cospsi, sinpsi,
        x12[6], x12[7], x12[8],
        x12[11],
    ], dtype=float)


def batch12_to_9(X12: np.ndarray) -> np.ndarray:
    return np.stack([dataset12_to_9(x) for x in X12], axis=0)


def state9_to_12(x9: np.ndarray) -> np.ndarray:
    """
    9D PINc state -> 12D dataset state for plotting/metrics.
    """
    x, y, z, cpsi, spsi, u, v, w, r = x9
    psi = math.atan2(spsi, cpsi)
    out = np.zeros(12, dtype=float)
    out[0:3] = [x, y, z]
    out[3:6] = [0.0, 0.0, psi]      # phi, theta ignored
    out[6:9] = [u, v, w]
    out[9:12] = [0.0, 0.0, r]       # p, q ignored
    return out


def batch9_to_12(X9: np.ndarray) -> np.ndarray:
    return np.stack([state9_to_12(x) for x in X9], axis=0)


class AdaptiveSoftplus(nn.Module):
    def __init__(self, beta_init=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x):
        return F.softplus(self.beta * x) / (self.beta + 1e-12)


class PINcNet(nn.Module):
    """
    Residual integrator:
      x_{k+1} = x_k + f_theta([x_k, u_k, dt])

    States: [x,y,z,cosψ,sinψ,u,v,w,r]  (9)
    Inputs: [X,Y,Z,Mz]                 (4)
    """
    def __init__(self, hidden_sizes=(64, 64, 64, 64)):
        super().__init__()
        self.Nx, self.Nu = 9, 4
        Nin  = self.Nx + self.Nu + 1   # +1 for dt
        Nout = self.Nx

        layers = []
        sizes = [Nin, *hidden_sizes, Nout]
        for i in range(len(sizes) - 2):
            layers += [
                nn.Linear(sizes[i], sizes[i+1]),
                AdaptiveSoftplus(),
                nn.LayerNorm(sizes[i+1]),
            ]
        layers += [nn.Linear(sizes[-2], sizes[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, z):  # z = [x9,u4,dt]
        """
        x_k = z[:, :9]
        """
        dx = self.net(z)                   # (B,9)

        # Rotate residual's x,y components body->world using current yaw
        cpsi = z[:, 3]
        spsi = z[:, 4]
        dx_bx = dx[:, 0]
        dx_by = dx[:, 1]
        dx_wx = cpsi * dx_bx - spsi * dx_by
        dx_wy = spsi * dx_bx + cpsi * dx_by

        base = z[:, :9] + dx              # provisional next state

        # Replace x,y increments with rotated ones
        x_next_xy = torch.stack([
            dx_wx + z[:, 0],
            dx_wy + z[:, 1]
        ], dim=1)

        # Re-normalize cos/sin without in-place ops
        c = base[:, 3]
        s = base[:, 4]
        norm = torch.clamp(torch.sqrt(c * c + s * s), min=1e-6)
        c_hat = c / norm
        s_hat = s / norm

        x_next = torch.cat(
            [
                x_next_xy,               # x, y
                base[:, 2:3],            # z
                c_hat.unsqueeze(1),      # cosψ
                s_hat.unsqueeze(1),      # sinψ
                base[:, 5:9],            # u, v, w, r
            ],
            dim=1,
        )
        return x_next


def make_pinc_dataset(X12: np.ndarray,
                      U8: np.ndarray,
                      dt: float,
                      old6: BlueROV2):
    """
    Build (x9_k, u4_k, dt) -> target x9_{k+1} pairs for PINc.
    """
    X9 = batch12_to_9(X12)  # (N,9)
    U4 = np.stack(
        [thrusters_to_body_wrenches(u8, dt, old6) for u8 in U8],
        axis=0
    )                        # (N,4)

    xk  = X9[:-1]
    uk  = U4[:-1]
    xk1 = X9[1:]
    dts = np.full((len(xk), 1), dt, dtype=float)

    z_in = np.hstack([xk, uk, dts])  # (N-1,14)
    y    = xk1                      # (N-1,9)
    return z_in, y, U4


@torch.no_grad()
def physics_loss(bluerov_rhs,
                 x_next_pred: torch.Tensor,
                 u4: torch.Tensor):
    """
    Physics guidance: evaluate continuous RHS at predicted next state
    x_{k+1}, penalize its norm.
    """
    t_dummy = 0.0
    rhs = bluerov_rhs(t_dummy, x_next_pred, u4)  # (B,9)
    return (rhs ** 2).mean()


def rollout_loss(model: PINcNet,
                 z_seq: torch.Tensor,
                 steps: int):
    """
    Multi-step rollout loss using model predictions.

    z_seq: (N,14) with [x9, u4, dt] per row.
    """
    N = z_seq.shape[0]
    if steps <= 0 or N < steps + 1:
        return torch.tensor(0.0, device=z_seq.device)

    z0 = z_seq[0:1, :]         # (1,14)
    x_curr = z0[:, :9]
    dt_val = z0[:, 13:14]

    loss = 0.0
    for i in range(steps):
        u_i = z_seq[i:i+1, 9:13]
        z_i = torch.cat([x_curr, u_i, dt_val], dim=1)
        x_next_pred = model(z_i)

        x_next_true = z_seq[i+1:i+2, :9]
        loss = loss + F.mse_loss(x_next_pred, x_next_true)

        x_curr = x_next_pred

    return loss / float(steps)


def train_pinc(z_train: np.ndarray,
               y_train: np.ndarray,
               u4_train: np.ndarray,
               dt: float,
               device,
               epochs: int = 100,
               batch: int = 256,
               lr: float = 3e-3,
               use_physics: bool = True,
               use_rollout: bool = True,
               rollout_steps: int = 10) -> PINcNet:
    model = PINcNet(hidden_sizes=PINc_HIDDEN).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    Z = torch.from_numpy(z_train).float().to(device)  # (N,14)
    Y = torch.from_numpy(y_train).float().to(device)  # (N,9)
    U = torch.from_numpy(u4_train[:-1]).float().to(device)  # align length

    ds = TensorDataset(Z, Y, U)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)

    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for z_b, y_b, u_b in dl:
            x_pred = model(z_b)
            loss = F.mse_loss(x_pred, y_b)

            if use_physics:
                loss = loss + 0.5 * physics_loss(bluerov_compute, x_pred, u_b)

            if use_rollout:
                K = min(rollout_steps, z_b.shape[0] - 1)
                if K > 0:
                    loss = loss + rollout_loss(model, z_b, K)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += float(loss.item())

        if (ep + 1) % 10 == 0:
            print(f"[PINc] epoch {ep+1:4d}/{epochs} | loss ~ {ep_loss / len(dl):.6f}")

    return model


def simulate_pinc(x0_12: np.ndarray,
                  U_seq_8: np.ndarray,
                  dt: float,
                  model: PINcNet,
                  old_model_for_map: BlueROV2,
                  device):
    """
    Rollout PINc model; returns trajectory in 12D for comparison.
    """
    H = len(U_seq_8)
    traj12 = np.zeros((H + 1, 12), float)
    x9 = dataset12_to_9(x0_12)
    traj12[0] = x0_12.copy()

    model.eval()
    with torch.no_grad():
        for k in range(H):
            u4 = thrusters_to_body_wrenches(U_seq_8[k], dt, old_model_for_map)
            z = np.hstack([x9, u4, [dt]])[None, :]  # (1,14)
            z_t = torch.from_numpy(z).float().to(device)
            x9_next = model(z_t).cpu().numpy()[0]
            x9 = x9_next
            traj12[k + 1] = state9_to_12(x9_next)

    return traj12


def multistep_rmse_endpoint_pinc(X_test: np.ndarray,
                                 U_test: np.ndarray,
                                 H: int,
                                 dt: float,
                                 model: PINcNet,
                                 old_model_for_map: BlueROV2,
                                 device) -> float:
    """
    Endpoint RMSE for PINc using 12D projection.
    """
    T = len(X_test)
    n_states = X_test.shape[1]
    n_start = T - H
    if n_start <= 0:
        return float("nan")

    se_total = 0.0
    for k in range(n_start):
        x0 = X_test[k]
        U_seq = U_test[k:k+H]
        x_end = simulate_pinc(x0, U_seq, dt, model, old_model_for_map, device)[-1]
        err = x_end - X_test[k + H]
        se_total += float(np.dot(err, err))
    return float(np.sqrt(se_total / (n_start * n_states)))


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Torch device: {device}")

    # 1) Load dataset
    repo_root = find_project_root(Path(__file__).parent)
    # Adjust this path as needed
    rosbags_dir = repo_root / "rosbags/rosbag2_2025_11_06/rosbag2_2025_11_06-manual"
    csv_path = find_latest_csv(rosbags_dir, DATASET_NAME)

    X, U, dt = load_dataset(csv_path)
    N = len(X)
    if N < 3:
        raise RuntimeError("Not enough samples to train/evaluate.")

    # Train/test split (here we just reuse full dataset for both)
    split = int(TRAIN_SPLIT * N)
    X_train, U_train = X[:split], U[:split]
    X_test,  U_test  = X[split:], U[split:]
    print(f"[i] Train: {len(X_train)} | Test: {len(X_test)}")

    Path("models").mkdir(parents=True, exist_ok=True)
    Path("media").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    #  2) Fit Koopman EDMDc (timed)
    # ------------------------------------------------------------------
    modelK = KoopmanEDMDc(
        state_dim=12,
        input_dim=8,
        n_rbfs=N_RBFS,
        gamma=GAMMA,
        ridge=RIDGE,
    )
    t0 = perf_counter()
    modelK.fit(X_train, U_train)
    t_fit_koop = perf_counter() - t0
    print(f"[ok] Koopman model fitted. (fit time = {t_fit_koop:.3f} s)")

    # ------------------------------------------------------------------
    #  3) Learn Double Integrator gains (timed)
    # ------------------------------------------------------------------
    t0 = perf_counter()
    K_lin, K_ang = estimate_di_gains(X_train, U_train, dt, ridge=1e-3)
    t_fit_di = perf_counter() - t0
    print(f"[ok] Double Integrator gains learned. (fit time = {t_fit_di:.3f} s)")

    # Fossen has no training step
    t_fit_phys = 0.0

    # ------------------------------------------------------------------
    #  4) Train or load PINc (timed)
    # ------------------------------------------------------------------
    rov_old = BlueROV2(dt=dt)   # used for thruster map
    if PINc_CKPT.exists():
        print(f"[i] Loading PINc checkpoint: {PINc_CKPT}")
        pinc = PINcNet(hidden_sizes=PINc_HIDDEN).to(device)
        pinc.load_state_dict(torch.load(PINc_CKPT, map_location=device))
        t_fit_pinc = 0.0
    else:
        print("[i] Training PINc (no checkpoint found).")
        z_train, y_train, u4_all = make_pinc_dataset(X_train, U_train, dt, rov_old)
        t0 = perf_counter()
        pinc = train_pinc(
            z_train, y_train, u4_all, dt, device,
            epochs=PINc_EPOCHS,
            batch=PINc_BATCH,
            lr=PINc_LR,
            use_physics=PINc_USE_PHYSICS,
            use_rollout=PINc_USE_ROLLOUT,
            rollout_steps=PINc_ROLLOUT_STEPS,
        )
        t_fit_pinc = perf_counter() - t0
        torch.save(pinc.state_dict(), PINc_CKPT)
        print(f"[ok] Saved PINc checkpoint -> {PINc_CKPT}")
    print(f"[ok] PINc ready. (fit/load time = {t_fit_pinc:.3f} s)")

    # ------------------------------------------------------------------
    #  5) Metrics: endpoint H-step RMSE + timings for all 4 models
    # ------------------------------------------------------------------
    print("\n[metrics] Endpoint RMSE (full 12D state) with identical evaluator:")

    # Koopman (use its own multistep_rmse)
    t0 = perf_counter(); rmse_1_kgen   = modelK.multistep_rmse(X_test, U_test, H=1);   t1_koop   = perf_counter() - t0
    t0 = perf_counter(); rmse_10_kgen  = modelK.multistep_rmse(X_test, U_test, H=10);  t10_koop  = perf_counter() - t0
    t0 = perf_counter(); rmse_100_kgen = modelK.multistep_rmse(X_test, U_test, H=100); t100_koop = perf_counter() - t0

    # Physics
    t0 = perf_counter(); rmse_1_phys   = multistep_rmse_endpoint_physics(X_test, U_test, H=1,   dt=dt); t1_phys   = perf_counter() - t0
    t0 = perf_counter(); rmse_10_phys  = multistep_rmse_endpoint_physics(X_test, U_test, H=10,  dt=dt); t10_phys  = perf_counter() - t0
    t0 = perf_counter(); rmse_100_phys = multistep_rmse_endpoint_physics(X_test, U_test, H=100, dt=dt); t100_phys = perf_counter() - t0

    # Double Integrator
    t0 = perf_counter(); rmse_1_di     = multistep_rmse_endpoint_di(X_test, U_test, H=1,   dt=dt, K_lin=K_lin, K_ang=K_ang);   t1_di   = perf_counter() - t0
    t0 = perf_counter(); rmse_10_di    = multistep_rmse_endpoint_di(X_test, U_test, H=10,  dt=dt, K_lin=K_lin, K_ang=K_ang);  t10_di  = perf_counter() - t0
    t0 = perf_counter(); rmse_100_di   = multistep_rmse_endpoint_di(X_test, U_test, H=100, dt=dt, K_lin=K_lin, K_ang=K_ang); t100_di = perf_counter() - t0

    # PINc
    t0 = perf_counter(); rmse_1_pinc   = multistep_rmse_endpoint_pinc(X_test, U_test, H=1,   dt=dt, model=pinc, old_model_for_map=rov_old, device=device);   t1_pinc   = perf_counter() - t0
    t0 = perf_counter(); rmse_10_pinc  = multistep_rmse_endpoint_pinc(X_test, U_test, H=10,  dt=dt, model=pinc, old_model_for_map=rov_old, device=device);  t10_pinc  = perf_counter() - t0
    t0 = perf_counter(); rmse_100_pinc = multistep_rmse_endpoint_pinc(X_test, U_test, H=100, dt=dt, model=pinc, old_model_for_map=rov_old, device=device); t100_pinc = perf_counter() - t0

    print("  Model                 | 1-step RMSE | 10-step RMSE | 100-step RMSE")
    print("  ----------------------|------------:|-------------:|--------------:")
    print(f"  Koopman               | {rmse_1_kgen:11.6f} | {rmse_10_kgen:12.6f} | {rmse_100_kgen:13.6f}")
    print(f"  Fossen (BlueROV2)     | {rmse_1_phys:11.6f} | {rmse_10_phys:12.6f} | {rmse_100_phys:13.6f}")
    print(f"  Double Integrator     | {rmse_1_di:11.6f} | {rmse_10_di:12.6f} | {rmse_100_di:13.6f}")
    print(f"  PINc (ResDNN)         | {rmse_1_pinc:11.6f} | {rmse_10_pinc:12.6f} | {rmse_100_pinc:13.6f}")

    print("\n[timings] Computation time (seconds):")
    print("  Phase \\ Model         |   Koopman |    Fossen |        DI |      PINc")
    print("  ----------------------|----------:|----------:|----------:|----------:")
    print(f"  Train/Fit             | {t_fit_koop:10.4f} | {t_fit_phys:10.4f} | {t_fit_di:10.4f} | {t_fit_pinc:10.4f}")
    print(f"  Metrics H=1           | {t1_koop:10.4f} | {t1_phys:10.4f} | {t1_di:10.4f} | {t1_pinc:10.4f}")
    print(f"  Metrics H=10          | {t10_koop:10.4f} | {t10_phys:10.4f} | {t10_di:10.4f} | {t10_pinc:10.4f}")
    print(f"  Metrics H=100         | {t100_koop:10.4f} | {t100_phys:10.4f} | {t100_di:10.4f} | {t100_pinc:10.4f}")

    # ------------------------------------------------------------------
    #  6) Open-loop demo for all 4 models + TRUE + rollout timings
    # ------------------------------------------------------------------
    horizon = min(OPEN_LOOP_STEPS, len(X_test) - 1)
    start = int(0.4 * (len(X_test) - horizon))
    x0 = X_test[start]
    U_seq = U_test[start:start+horizon]

    # Koopman
    t0 = perf_counter()
    predK = modelK.simulate(x0, U_seq)
    t_roll_koop = perf_counter() - t0

    # Fossen
    rov_phys = BlueROV2(dt=dt)
    t0 = perf_counter()
    predF = simulate_physics(x0, U_seq, dt, rov_phys)
    t_roll_phys = perf_counter() - t0

    # Double Integrator
    t0 = perf_counter()
    predD = simulate_double_integrator(x0, U_seq, dt, K_lin, K_ang)
    t_roll_di = perf_counter() - t0

    # PINc
    t0 = perf_counter()
    predP = simulate_pinc(x0, U_seq, dt, pinc, rov_old, device)
    t_roll_pinc = perf_counter() - t0

    true_traj = X_test[start:start+horizon+1]

    print("\n[timings] Rollout time for animation horizon:")
    print("  Model                 | Rollout time [s]")
    print("  ----------------------|-----------------:")
    print(f"  Koopman               | {t_roll_koop:16.6f}")
    print(f"  Fossen (BlueROV2)     | {t_roll_phys:16.6f}")
    print(f"  Double Integrator     | {t_roll_di:16.6f}")
    print(f"  PINc (ResDNN)         | {t_roll_pinc:16.6f}")

    # ------------------------------------------------------------------
    #  7) Top-view animation (TRUE + 4 models)
    # ------------------------------------------------------------------
    animate_xy_five(
        true_traj=true_traj,
        koop_traj=predK,
        fossen_traj=predF,
        di_traj=predD,
        pinc_traj=predP,
        dt=dt,
        save_path="media/csv_true_vs_4models.gif",
        title="Recorded CSV: True vs. Koopman / Fossen / DI / PINc",
        tail_secs=10.0,
        speed=1.0,
        dpi=130,
    )

    # ------------------------------------------------------------------
    #  8) 2D static figure (~10s) for LaTeX (TRUE + 4 models)
    # ------------------------------------------------------------------
    plot_2d_trajectories_with_depth(
        true_traj=true_traj,
        koop_traj=predK,
        fossen_traj=predF,
        di_traj=predD,
        pinc_traj=predP,
        dt=dt,
        seconds=PLOT_FIG_SECONDS,
        save_path="media/true_vs_4models_2D.png",
    )

    # ------------------------------------------------------------------
    #  9) Optional: print first 200 predicted vs true (Koopman only)
    # ------------------------------------------------------------------
    print("\nFirst 200 predicted vs. true body positions (m) & orientations (deg) [Koopman]:")
    for k in range(min(200, horizon)):
        kx, ky, kz = predK[k, 0:3]
        tx, ty, tz = true_traj[k, 0:3]
        Kang = np.rad2deg(predK[k, 3:6])
        Tang = np.rad2deg(true_traj[k, 3:6])
        print(
            f"t={k*dt:4.2f}s: "
            f"K pred=({kx: .3f}, {ky: .3f}, {kz: .3f}) "
            f"ang=({Kang[0]: .2f}, {Kang[1]: .2f}, {Kang[2]: .2f}) | "
            f"true=({tx: .3f}, {ty: .3f}, {tz: .3f}) "
            f"ang=({Tang[0]: .2f}, {Tang[1]: .2f}, {Tang[2]: .2f})"
        )


if __name__ == "__main__":
    main()