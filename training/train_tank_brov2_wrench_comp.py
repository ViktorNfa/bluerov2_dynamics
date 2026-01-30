"""
End-to-end example (wrench dataset):
  1) Load the newest rosbags/**/koopman_dataset_50Hz_with_wrench.csv
  2) Fit an EDMDc Koopman model (RBF dictionary) with 6D wrench input
  3) Compare against:
       - BlueROV2 physics model with direct wrench input
       - A simple Double Integrator (learned from wrench → accel)
     using:
       - One-step RMSE
       - Multi-step RMSE (H=10 and H=100)
  4) Show a short multi-step open-loop prediction vs. ground truth
     for ALL (Koopman, Fossen, Double Integrator) on the test split
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path

from Koopman.koopmanEDMDc import KoopmanEDMDc
from fossen.BlueROV2_thrust import BlueROV2


# ------------------------------------------------------------------
#  0)  Config
# ------------------------------------------------------------------
DATASET_NAME = "koopman_dataset_50Hz_with_wrench.csv"
TRAIN_SPLIT = 0.80
N_RBFS = 500
GAMMA = 3.0
RIDGE = 1e-1
OPEN_LOOP_STEPS = 500


# ----------------------- Side-by-side animation helper -----------------------
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

def animate_xy_four(
    true_traj: np.ndarray,
    koop_traj: np.ndarray,
    fossen_traj: np.ndarray,
    di_traj: np.ndarray,
    dt: float,
    save_path: str | None = None,   # e.g. "media/true_vs_models.gif"
    title: str = "Recorded CSV: True vs. Models (top view)",
    tail_secs: float = 10.0,
    speed: float = 1.0,
    dpi: int = 120
):
    """
    2x2 top-view animation: TRUE | KOOPMAN
                           FOSSEN | DOUBLE INTEGRATOR
    All trajectories must be the same shape (T, >=6).
    """
    assert true_traj.shape == koop_traj.shape == fossen_traj.shape == di_traj.shape
    T, n = true_traj.shape
    assert n >= 6, "Expected at least 6D state with psi at index 5."

    # Shared limits across all four
    xs = np.concatenate([true_traj[:, 0], koop_traj[:, 0], fossen_traj[:, 0], di_traj[:, 0]])
    ys = np.concatenate([true_traj[:, 1], koop_traj[:, 1], fossen_traj[:, 1], di_traj[:, 1]])
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    pad_x = 0.10 * max(1e-6, x_max - x_min)
    pad_y = 0.10 * max(1e-6, y_max - y_min)
    x_lim = (x_min - pad_x, x_max + pad_x)
    y_lim = (y_min - pad_y, y_max + pad_y)

    tail = max(1, int(tail_secs / max(dt, 1e-9)))

    fig, axs = plt.subplots(2, 2, figsize=(10, 9), dpi=dpi, constrained_layout=True)
    fig.suptitle(title)

    panels = [
        (axs[0,0], "TRUE (Recorded)",  true_traj,  "C0"),
        (axs[0,1], "KOOPMAN",          koop_traj,  "C1"),
        (axs[1,0], "FOSSEN (BlueROV2)",fossen_traj,"C2"),
        (axs[1,1], "DOUBLE INTEGRATOR",di_traj,    "C3"),
    ]

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
        dot, = ax.plot([], [], "o", ms=6, color=color)
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
            x = traj[:, 0]; y = traj[:, 1]; z = traj[:, 2]; psi = traj[:, 5]
            path.set_data(x[s:i+1], y[s:i+1])
            dot.set_data([x[i]], [y[i]])
            x0, y0, yaw = x[i], y[i], psi[i]
            x1, y1 = x0 + head_len * math.cos(yaw), y0 + head_len * math.sin(yaw)
            arrow.set_positions((x0, y0), (x1, y1))
            txt.set_text(f"t = {i*dt:5.2f} s\nz = {z[i]:.2f} m")
            outs.extend([path, dot, arrow, txt])
        return tuple(outs)

    interval_ms = int(max(1, 1000.0 * dt / max(speed, 1e-6)))
    ani = FuncAnimation(fig, update, frames=T, init_func=init, blit=True, interval=interval_ms)

    if save_path is None:
        plt.show()
    else:
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
# -------------------------------------------------------------------------------


# ------------------------------------------------------------------
#  Helpers
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

    # State and input columns for wrench dataset
    STATE_COLS = ["x","y","z","phi","theta","psi","u","v","w","p","q","r"]
    INPUT_COLS = ["Fx","Fy","Fz","Mx","My","Mz"]

    # Sanity checks / fill missing input columns with zeros if needed
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

    X = df[STATE_COLS].to_numpy(dtype=float)
    U = df[INPUT_COLS].to_numpy(dtype=float)

    t = df["t"].to_numpy(dtype=float)
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.05

    print(f"[i] Samples: {len(df)} | median dt ≈ {dt:.5f}s (~{1.0/max(dt,1e-9):.2f} Hz)")
    return X, U, dt

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ----------------------- Physics (BlueROV2 wrench) -----------------------
def simulate_physics(x0: np.ndarray, U_seq: np.ndarray, dt: float, rov: BlueROV2):
    """
    Explicit-Euler rollout of the BlueROV2 wrench-input model using recorded wrenches.
    Returns trajectory of shape (len(U_seq)+1, 12).
    """
    H = len(U_seq)
    traj = np.zeros((H + 1, x0.shape[0]))
    traj[0] = x0
    x = x0.copy()
    for k in range(H):
        dx = rov.dynamics(x, U_seq[k], dt)  # tau_body = U_seq[k]
        x = x + dt * dx
        traj[k + 1] = x
    return traj

def one_step_rmse_physics(X_test: np.ndarray, U_test: np.ndarray, dt: float) -> float:
    rov = BlueROV2()
    preds = np.zeros_like(X_test)
    preds[0] = X_test[0]
    for k in range(len(X_test) - 1):
        dx = rov.dynamics(X_test[k], U_test[k], dt)
        preds[k + 1] = X_test[k] + dt * dx
    return rmse(X_test[1:], preds[1:])

def multistep_rmse_endpoint_physics(X_test: np.ndarray,
                                    U_test: np.ndarray,
                                    H: int,
                                    dt: float) -> float:
    T = len(X_test)
    n_states = X_test.shape[1]
    n_start = T - H
    if n_start <= 0:
        return float("nan")

    rov = BlueROV2()
    se_total = 0.0
    for k in range(n_start):
        x0 = X_test[k]
        U_seq = U_test[k:k+H]
        x_end = simulate_physics(x0, U_seq, dt, rov)[-1]
        err = x_end - X_test[k + H]
        se_total += float(np.dot(err, err))
    return float(np.sqrt(se_total / (n_start * n_states)))


# ----------------------- Double Integrator (wrench-driven) -----------------------
def _euler_to_R_b2n(phi, theta, psi):
    """Rotation body->world using Z(psi) Y(theta) X(phi)."""
    cph, sph = math.cos(phi), math.sin(phi)
    cth, sth = math.cos(theta), math.sin(theta)
    cps, sps = math.cos(psi), math.sin(psi)
    Rz = np.array([[cps, -sps, 0.0],
                   [sps,  cps, 0.0],
                   [0.0,  0.0,  1.0]])
    Ry = np.array([[ cth, 0.0, sth],
                   [ 0.0, 1.0, 0.0],
                   [-sth, 0.0, cth]])
    Rx = np.array([[1.0, 0.0,  0.0],
                   [0.0, cph, -sph],
                   [0.0, sph,  cph]])
    return Rz @ Ry @ Rx

def estimate_di_gains(X_train: np.ndarray,
                      U_train: np.ndarray,
                      dt: float,
                      ridge: float = 1e-3):
    """
    Learn linear maps (from 6D wrench to body accelerations):
        dv_body ≈ U @ K_lin    (K_lin: 6x3)
        dw_body ≈ U @ K_ang    (K_ang: 6x3)

    using forward differences on [u,v,w,p,q,r].
    """
    V = X_train[:, 6:9]    # body linear velocities
    W = X_train[:, 9:12]   # body angular rates
    dV = (V[1:] - V[:-1]) / max(dt, 1e-9)   # (N-1,3)
    dW = (W[1:] - W[:-1]) / max(dt, 1e-9)   # (N-1,3)
    G  = U_train[:-1]                       # (N-1,6) wrench

    GTG = G.T @ G
    I = np.eye(GTG.shape[0])
    K_lin = np.linalg.solve(GTG + ridge * I, G.T @ dV)   # (6,3)
    K_ang = np.linalg.solve(GTG + ridge * I, G.T @ dW)   # (6,3)
    return K_lin, K_ang

def simulate_double_integrator(x0: np.ndarray,
                               U_seq: np.ndarray,
                               dt: float,
                               K_lin: np.ndarray,
                               K_ang: np.ndarray):
    """
    Discrete DI in body frame, driven by wrench (6D):

      a_body  = U_k @ K_lin   (3,)
      alpha   = U_k @ K_ang   (3,)

      v_{k+1} = v_k + dt * a_body
      w_{k+1} = w_k + dt * alpha

      x_{k+1} = x_k + dt * (R_b2n(phi,theta,psi) @ v_k)
      ang_{k+1} = ang_k + dt * w_k  (small-angle approx)
    """
    H = len(U_seq)
    traj = np.zeros((H + 1, x0.shape[0]))
    traj[0] = x = x0.copy()

    for k in range(H):
        pos = x[0:3]
        ang = x[3:6]
        v = x[6:9]
        w = x[9:12]

        # Input (wrench) → accelerations in body frame
        a_body = U_seq[k] @ K_lin   # (3,)
        alpha = U_seq[k] @ K_ang    # (3,)

        # Integrate velocities
        v_next = v + dt * a_body
        w_next = w + dt * alpha

        # Integrate pose
        phi, theta, psi = ang
        Rb2n = _euler_to_R_b2n(phi, theta, psi)
        pos_next = pos + dt * (Rb2n @ v)
        ang_next = ang + dt * w   # small-angle approx

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
#  Main
# ------------------------------------------------------------------
def main():
    # 1) Load dataset
    repo_root = find_project_root(Path(__file__).parent)
    # Adjust this path if needed
    rosbags_dir = repo_root / "rosbags/rosbag2_2025_11_06/rosbag2_2025_11_06-manual"
    csv_path = find_latest_csv(rosbags_dir, DATASET_NAME)

    X, U, dt = load_dataset(csv_path)
    N = len(X)
    if N < 3:
        raise RuntimeError("Not enough samples to train/evaluate.")

    # 2) Train / test split
    split = int(TRAIN_SPLIT * N)
    X_train, U_train = X[:split], U[:split]
    X_test, U_test = X[split-1:], U[split-1:]

    print(f"[i] Train: {len(X_train)} | Test: {len(X_test)}")

    # 3) Fit Koopman EDMDc with 6D input (wrench)
    model = KoopmanEDMDc(
        state_dim=12,
        input_dim=6,
        n_rbfs=N_RBFS,
        gamma=GAMMA,
        ridge=RIDGE
    )
    model.fit(X_train, U_train)
    print("[ok] Koopman model fitted (wrench input).")

    # 3b) Learn Double Integrator gains from TRAIN split (wrench -> accel)
    K_lin, K_ang = estimate_di_gains(X_train, U_train, dt, ridge=1e-3)
    print("[ok] Double Integrator gains learned (6x3 for linear & angular).")

    # ------------------------------------------------------------------
    #  3c)  Metrics: identical evaluator (endpoint H-step RMSE)
    # ------------------------------------------------------------------
    rmse_1_kgen = model.multistep_rmse(X_test, U_test, H=1)
    rmse_10_kgen = model.multistep_rmse(X_test, U_test, H=10)
    rmse_100_kgen = model.multistep_rmse(X_test, U_test, H=100)

    rmse_1_phys = multistep_rmse_endpoint_physics(X_test, U_test, H=1,   dt=dt)
    rmse_10_phys = multistep_rmse_endpoint_physics(X_test, U_test, H=10,  dt=dt)
    rmse_100_phys = multistep_rmse_endpoint_physics(X_test, U_test, H=100, dt=dt)

    rmse_1_di = multistep_rmse_endpoint_di(X_test, U_test, H=1,   dt=dt, K_lin=K_lin, K_ang=K_ang)
    rmse_10_di = multistep_rmse_endpoint_di(X_test, U_test, H=10,  dt=dt, K_lin=K_lin, K_ang=K_ang)
    rmse_100_di = multistep_rmse_endpoint_di(X_test, U_test, H=100, dt=dt, K_lin=K_lin, K_ang=K_ang)

    print("\n[metrics] Side-by-side using identical evaluator (endpoint RMSE):")
    print("  Model               | 1-step RMSE | 10-step RMSE | 100-step RMSE")
    print("  ------------------- | ----------- | ------------ | -------------")
    print(f"  Koopman (wrench)    | {rmse_1_kgen:11.6f} | {rmse_10_kgen:12.6f} | {rmse_100_kgen:13.6f}")
    print(f"  Fossen (BlueROV2)   | {rmse_1_phys:11.6f} | {rmse_10_phys:12.6f} | {rmse_100_phys:13.6f}")
    print(f"  Double Integrator   | {rmse_1_di:11.6f} | {rmse_10_di:12.6f} | {rmse_100_di:13.6f}")

    # ------------------------------------------------------------------
    #  4)  Short open-loop demo (Koopman vs Fossen vs DI vs True)
    # ------------------------------------------------------------------
    horizon = min(OPEN_LOOP_STEPS, len(X_test) - 1)
    start = int(0 * (len(X_test) - horizon))
    x0 = X_test[start]
    U_seq = U_test[start:start+horizon]

    # Koopman rollout
    predK = model.simulate(x0, U_seq)                       # (horizon+1, 12)
    # Fossen/BlueROV2 rollout
    rov = BlueROV2()
    predF = simulate_physics(x0, U_seq, dt, rov)            # (horizon+1, 12)
    # Double Integrator rollout
    predD = simulate_double_integrator(x0, U_seq, dt, K_lin, K_ang)

    # Truth
    true_traj = X_test[start:start+horizon+1]

    Path("media").mkdir(exist_ok=True)
    animate_xy_four(
        true_traj=true_traj,
        koop_traj=predK,
        fossen_traj=predF,
        di_traj=predD,
        dt=dt,
        save_path="media/csv_true_vs_models_wrench.gif",
        title="Recorded CSV (wrench): True vs. Koopman / Fossen / Double Integrator",
        tail_secs=10.0,
        speed=1.0,
        dpi=130
    )

    print("\nFirst 200 predicted vs. true body positions (m) & orientations (deg) [Koopman]:")
    for k in range(min(200, horizon)):
        kx, ky, kz = predK[k, 0:3]
        tx, ty, tz = true_traj[k, 0:3]
        Kang = np.rad2deg(predK[k, 3:6])
        Tang = np.rad2deg(true_traj[k, 3:6])
        print(
            f"t={k*dt:4.2f}s: "
            f"K pred=({kx: .3f}, {ky: .3f}, {kz: .3f}) ang=({Kang[0]: .2f}, {Kang[1]: .2f}, {Kang[2]: .2f}) | "
            f"true=({tx: .3f}, {ty: .3f}, {tz: .3f}) ang=({Tang[0]: .2f}, {Tang[1]: .2f}, {Tang[2]: .2f})"
        )


if __name__ == "__main__":
    main()