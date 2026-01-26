"""
End-to-end example (recorded data):
  1) Load the newest rosbags/**/koopman_dataset_50Hz.csv
  2) Fit an EDMDc Koopman model (RBF dictionary)
  3) Compare against the BlueROV2 physics model:
       - One-step RMSE
       - Multi-step RMSE (H=10 and H=100) using the same evaluator
  4) Show a short multi-step open-loop prediction vs. ground truth
     for BOTH (Koopman and Physics) on the test split
"""

import numpy as np
import pandas as pd
from pathlib import Path

from Koopman.koopmanEDMDc import KoopmanEDMDc
from fossen.BlueROV2 import BlueROV2


# ------------------------------------------------------------------
#  0)  Config
# ------------------------------------------------------------------
DATASET_NAME = "koopman_dataset_50Hz.csv"
TRAIN_SPLIT = 0.80
N_RBFS = 500
GAMMA = 3.0
RIDGE = 1e-1
OPEN_LOOP_STEPS = 500


# ----------------------- Side-by-side animation helper -----------------------
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

def animate_xy_true_vs_koopman(
    true_traj: np.ndarray,
    koop_traj: np.ndarray,
    dt: float,
    save_path: str | None = None,   # e.g. "media/koopman_vs_true.gif" or None to just show
    title: str = "True vs. Koopman (top view)",
    tail_secs: float = 10.0,        # trailing path length
    speed: float = 1.0,             # >1.0 to speed up playback
    dpi: int = 120
):
    """
    Minimal dependencies: matplotlib (+ Pillow ONLY to save GIF).
    Shows/saves a side-by-side x-y animation with a heading indicator from ψ (yaw).
    State layout assumed: [x,y,z, phi,theta,psi, u,v,w, p,q,r].
    """
    assert true_traj.shape == koop_traj.shape
    T, n = true_traj.shape
    assert n >= 6, "Expected 12D state with psi at index 5."

    # Compute shared limits across both trajectories for consistent scale
    xs = np.concatenate([true_traj[:, 0], koop_traj[:, 0]])
    ys = np.concatenate([true_traj[:, 1], koop_traj[:, 1]])
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    # pad by 10%
    pad_x = 0.10 * max(1e-6, x_max - x_min)
    pad_y = 0.10 * max(1e-6, y_max - y_min)
    x_lim = (x_min - pad_x, x_max + pad_x)
    y_lim = (y_min - pad_y, y_max + pad_y)

    # trail length in samples
    tail = max(1, int(tail_secs / max(dt, 1e-9)))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 5), dpi=dpi, constrained_layout=True)
    fig.suptitle(title)

    for ax, name in [(axL, "TRUE (BlueROV2)"), (axR, "KOOPMAN PREDICTION")]:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(name)

    # Left (TRUE)
    true_path, = axL.plot([], [], lw=2, alpha=0.9)
    true_dot, = axL.plot([], [], "o", ms=6)
    true_arrow = FancyArrowPatch((0, 0), (0, 0),
                                 arrowstyle='-|>', mutation_scale=12,
                                 lw=2, color="C0", zorder=5)
    axL.add_patch(true_arrow)
    time_textL = axL.text(0.02, 0.98, "", transform=axL.transAxes, va="top")

    # Right (KOOPMAN)
    koop_path, = axR.plot([], [], lw=2, alpha=0.9)
    koop_dot,  = axR.plot([], [], "o", ms=6)
    koop_arrow = FancyArrowPatch((0, 0), (0, 0),
                                 arrowstyle='-|>', mutation_scale=12,
                                 lw=2, color="C1", zorder=5)
    axR.add_patch(koop_arrow)
    time_textR = axR.text(0.02, 0.98, "", transform=axR.transAxes, va="top")

    # choose neutral colors (matplotlib default theme)
    true_path.set_color("C0"); true_dot.set_color("C0"); true_arrow.set_color("C0")
    koop_path.set_color("C1"); koop_dot.set_color("C1"); koop_arrow.set_color("C1")

    head_len = 0.1 * max(x_max - x_min, y_max - y_min) if (x_max > x_min or y_max > y_min) else 1.0

    def init():
        true_path.set_data([], [])
        true_dot.set_data([], [])
        true_arrow.set_positions((0, 0), (0, 0))
        koop_path.set_data([], [])
        koop_dot.set_data([], [])
        koop_arrow.set_positions((0, 0), (0, 0))
        time_textL.set_text("")
        time_textR.set_text("")
        return (true_path, true_dot, true_arrow, koop_path, koop_dot, koop_arrow, time_textL, time_textR)

    def update(i):
        # trailing window
        s = max(0, i - tail)
        # TRUE
        tx, ty = true_traj[s:i+1, 0], true_traj[s:i+1, 1]
        true_path.set_data(tx, ty)
        true_dot.set_data([true_traj[i, 0]], [true_traj[i, 1]])
        x0, y0, psi = true_traj[i, 0], true_traj[i, 1], true_traj[i, 5]
        x1, y1 = x0 + head_len * math.cos(psi), y0 + head_len * math.sin(psi)
        true_arrow.set_positions((x0, y0), (x1, y1))
        time_textL.set_text(f"t = {i*dt:5.2f} s\nz = {true_traj[i,2]:.2f} m")

        # KOOPMAN
        kx, ky = koop_traj[s:i+1, 0], koop_traj[s:i+1, 1]
        koop_path.set_data(kx, ky)
        koop_dot.set_data([koop_traj[i, 0]], [koop_traj[i, 1]])
        x0k, y0k, psik = koop_traj[i, 0], koop_traj[i, 1], koop_traj[i, 5]
        x1k, y1k = x0k + head_len * math.cos(psik), y0k + head_len * math.sin(psik)
        koop_arrow.set_positions((x0k, y0k), (x1k, y1k))
        time_textR.set_text(f"t = {i*dt:5.2f} s\nz = {koop_traj[i,2]:.2f} m")

        return (true_path, true_dot, true_arrow, koop_path, koop_dot, koop_arrow, time_textL, time_textR)

    interval_ms = int(max(1, 1000.0 * dt / max(speed, 1e-6)))
    ani = FuncAnimation(fig, update, frames=T, init_func=init, blit=True, interval=interval_ms)

    if save_path is None:
        plt.show()
    else:
        try:
            if save_path.lower().endswith(".gif"):
                # Pillow is optional; only needed to write GIF
                from matplotlib.animation import PillowWriter
                ani.save(save_path, writer=PillowWriter(fps=int(round(1.0/dt*speed))), dpi=dpi)
            else:
                # Try mp4 (requires ffmpeg installed in PATH)
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

    # Expected columns
    STATE_COLS = ["x","y","z","phi","theta","psi","u","v","w","p","q","r"]
    INPUT_COLS = [f"u{i}" for i in range(1,9)]

    # Sanity checks / fill missing input columns with zeros if needed
    for c in STATE_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing state column: {c}")
    for c in INPUT_COLS:
        if c not in df.columns:
            df[c] = 0.0

    # Sort by time and clean
    if "t" not in df.columns:
        raise ValueError("CSV must contain a 't' time column.")
    df = df.sort_values("t").drop_duplicates(subset="t")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=STATE_COLS)

    # Arrays
    X = df[STATE_COLS].to_numpy(dtype=float)
    U = df[INPUT_COLS].to_numpy(dtype=float)

    # Median dt (for info + physics baseline integrator)
    t = df["t"].to_numpy(dtype=float)
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.05

    print(f"[i] Samples: {len(df)} | median dt ≈ {dt:.5f}s (~{1.0/max(dt,1e-9):.2f} Hz)")
    return X, U, dt

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

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

def one_step_rmse_physics(X_test: np.ndarray, U_test: np.ndarray, dt: float) -> float:
    """
    Physics 1-step: predict X_{k+1} from (X_k, U_k) for all k, Euler step.
    """
    rov = BlueROV2(dt=dt)   # use same dt so thruster lags discretise consistently
    preds = np.zeros_like(X_test)
    preds[0] = X_test[0]
    for k in range(len(X_test) - 1):
        dx = rov.dynamics(X_test[k], U_test[k], dt)
        preds[k + 1] = X_test[k] + dt * dx
    return rmse(X_test[1:], preds[1:])

def multistep_rmse_endpoint_koopman(model: KoopmanEDMDc,
                                    X_test: np.ndarray,
                                    U_test: np.ndarray,
                                    H: int) -> float:
    """
    Strict H-step-ahead RMSE (endpoint only), identical to KoopmanEDMDc.multistep_rmse.
    """
    return model.multistep_rmse(X_test, U_test, H=H)

def multistep_rmse_endpoint_physics(X_test: np.ndarray,
                                    U_test: np.ndarray,
                                    H: int,
                                    dt: float) -> float:
    """
    Strict H-step-ahead RMSE (endpoint only) for the BlueROV2 baseline.
    For each start index k, simulate H steps from X[k] with U[k:k+H],
    compare only the final state to X[k+H].
    """
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
        x_end = simulate_physics(x0, U_seq, dt, rov)[-1]   # endpoint only
        err = x_end - X_test[k + H]
        se_total += float(np.dot(err, err))

    return float(np.sqrt(se_total / (n_start * n_states)))


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------
def main():
    # 1) Load dataset
    repo_root = find_project_root(Path(__file__).parent)
    rosbags_dir = repo_root / "rosbags/rosbag2_2025_11_06/rosbag2_2025_11_06-manual"
    csv_path = find_latest_csv(rosbags_dir, DATASET_NAME)

    X, U, dt = load_dataset(csv_path)
    N = len(X)
    if N < 3:
        raise RuntimeError("Not enough samples to train/evaluate.")

    # 2) Train / test split  (causal: start test one sample earlier)
    split = int(TRAIN_SPLIT * N)
    X_train, U_train = X[:split], U[:split]
    X_test, U_test = X[split-1:], U[split-1:]

    print(f"[i] Train: {len(X_train)} | Test: {len(X_test)}")

    # 3) Fit Koopman EDMDc
    model = KoopmanEDMDc(
        state_dim=12,
        input_dim=8,
        n_rbfs=N_RBFS,
        gamma=GAMMA,
        ridge=RIDGE
    )
    model.fit(X_train, U_train)
    print("[ok] Koopman model fitted.")

    # ------------------------------------------------------------------
    #  3a)  Metrics (native Koopman + side-by-side comparable metrics)
    # ------------------------------------------------------------------

    # Physics + Koopman with the SAME evaluator for a fair comparison
    rmse_1_kgen = multistep_rmse_endpoint_koopman(model, X_test, U_test, H=1)  # 1-step
    rmse_10_kgen = multistep_rmse_endpoint_koopman(model, X_test, U_test, H=10)
    rmse_100_kgen = multistep_rmse_endpoint_koopman(model, X_test, U_test, H=100)

    print("\n[metrics] Side-by-side using identical evaluator:")
    print("  Model          | 1-step RMSE | 10-step RMSE | 100-step RMSE")
    print("  -------------- | ----------- | ------------ | -------------")
    print(f"  Koopman        | {rmse_1_kgen:11.6f} | {rmse_10_kgen:12.6f} | {rmse_100_kgen:13.6f}")

    # ------------------------------------------------------------------
    #  4)  Short open-loop demo (Koopman vs Physics vs True)
    # ------------------------------------------------------------------
    start = 0
    max_h = (len(X_test) - 1) - start           # how many steps remain from start
    horizon = min(OPEN_LOOP_STEPS, max_h)       # number of control steps
    end = start + horizon

    x0 = X_test[start]
    U_seq = U_test[start:end]                   # length == horizon

    # Koopman rollout
    predK = model.simulate(x0, U_seq)           # (horizon+1, 12)

    # Truth
    true_traj = X[start:end + 1]

    Path("media").mkdir(exist_ok=True)
    animate_xy_true_vs_koopman(
        true_traj=true_traj,
        koop_traj=predK,
        dt=dt,
        save_path="media/csv_true_vs_koopman.gif",
        title="Recorded CSV: True vs. Koopman (top view)",
        tail_secs=10.0,
        speed=1.0,
        dpi=130
    )

    print("\nFirst 500 predicted vs. true body positions (m) & orientations (deg):")
    for k in range(horizon):
        # positions
        kx, ky, kz = predK[k, 0:3]
        # px, py, pz = predP[k, 0:3]
        tx, ty, tz = true_traj[k, 0:3]
        # angles -> deg for readability
        Kang = np.rad2deg(predK[k, 3:6])
        # Pang = np.rad2deg(predP[k, 3:6])
        Tang = np.rad2deg(true_traj[k, 3:6])

        print(
            f"t={k*dt:4.2f}s: "
            f"K pred=({kx: .3f}, {ky: .3f}, {kz: .3f}) ang=({Kang[0]: .2f}, {Kang[1]: .2f}, {Kang[2]: .2f}) | "
            f"true=({tx: .3f}, {ty: .3f}, {tz: .3f}) ang=({Tang[0]: .2f}, {Tang[1]: .2f}, {Tang[2]: .2f})"
        )

    # for k in range(horizon):
    #     print(
    #         f"t={k*dt:4.2f}s: "
    #         f"K pred=[{predK[k,6]: .3f}, {predK[k,7]: .3f}, {predK[k,8]: .3f}, {predK[k,9]: .3f}, {predK[k,10]: .3f}, {predK[k,11]: .3f}], "
    #         f"true=[{true_traj[k,6]: .3f}, {true_traj[k,7]: .3f}, {true_traj[k,8]: .3f}, {true_traj[k,9]: .3f}, {true_traj[k,10]: .3f}, {true_traj[k,11]: .3f}]"
    #     )


if __name__ == "__main__":
    main()