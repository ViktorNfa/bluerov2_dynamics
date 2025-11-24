"""
End-to-end example:
  1) Simulate the BlueROV2 physics model to generate data
  2) Fit a EDMDc Koopman model with 200 RBFs
  3) Report one-step and multi-step RMSE on a held-out test set
  4) Show a short multi-step prediction versus ground truth
"""

import numpy as np
from Koopman.koopmanEDMDc import KoopmanEDMDc
from fossen.BlueROV2 import BlueROV2

from pathlib import Path


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
    true_dot,  = axL.plot([], [], "o", ms=6)
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
#  1)  Generate a data set
# ------------------------------------------------------------------
np.random.seed(42)
dt = 0.05                   # 20 Hz sampling
T_total = 12000.0           # 10 min rollout
N = int(T_total / dt)

rov = BlueROV2(dt=dt)

# random but smooth thruster commands
def random_input(prev):
    alpha = 0.98
    noise = 0.02 * np.random.randn(rov.n_thrusters)
    return np.clip(alpha * prev + noise, -1.0, 1.0)

states_true = np.zeros((N, 12))     # ground-truth (noiseless)
states = np.zeros((N, 12))          # noisy measurements
inputs = np.zeros((N, 8))

x = np.zeros(12)                    # initial state (rest at origin)
u_prev = np.zeros(8)

# define realistic noise scales (tunable)
pos_noise_std = 0.0005      # m
vel_noise_std = 0.0005      # m/s
ang_noise_std = 0.001       # rad
angvel_noise_std = 0.001    # rad/s

for k in range(N):
    u = random_input(u_prev)
    dx = rov.dynamics(x, u, dt)
    x = x + dt * dx

    # store ground truth
    states_true[k] = x

    # add sensor-like Gaussian noise
    noisy_state = x.copy()
    noisy_state[0:3]  += pos_noise_std     * np.random.randn(3)  # position
    noisy_state[3:6]  += ang_noise_std     * np.random.randn(3)  # Euler angles (rad)
    noisy_state[6:9]  += vel_noise_std     * np.random.randn(3)  # linear velocity
    noisy_state[9:12] += angvel_noise_std  * np.random.randn(3)  # angular velocity


    states[k] = noisy_state
    inputs[k] = u
    u_prev = u

# ------------------------------------------------------------------
#  2)  Train / test split
# ------------------------------------------------------------------
split = int(0.8 * N)
X_train, U_train = states[:split], inputs[:split]
X_test,  U_test  = states[split-1:], inputs[split-1:]   # -1 for causality

model = KoopmanEDMDc(
    state_dim=12,
    input_dim=8,
    n_rbfs=200,
    gamma=1.0,
    ridge=1e-3
)

model.fit(X_train, U_train)
print("Model fitted!")

# ------------------------------------------------------------------
#  3)  Quantitative accuracy
# ------------------------------------------------------------------
rmse = model.evaluate(X_test, U_test)
print(f"One-step RMSE on test set: {rmse:.4f}")

horizon_steps = 10
rmse_H = model.multistep_rmse(X_test, U_test, H=horizon_steps)
print(f"{horizon_steps}-step RMSE on test set: {rmse_H:.4f}")

horizon_steps = 100
rmse_H = model.multistep_rmse(X_test, U_test, H=horizon_steps)
print(f"{horizon_steps}-step RMSE on test set: {rmse_H:.4f}")

# ------------------------------------------------------------------
#  4)  Multi-step open-loop demo
# ------------------------------------------------------------------
horizon = 200               # 10 s
x0 = X_test[0]
U_seq = U_test[:horizon]

pred_traj = model.simulate(x0, U_seq)
true_traj = states[split-1: split-1+horizon+1]

# ---- Make animation (side-by-side) ----
Path("media").mkdir(exist_ok=True)
animate_xy_true_vs_koopman(
    true_traj=true_traj,
    koop_traj=pred_traj,
    dt=dt,
    save_path="media/sim_true_vs_koopman.gif",  # or None to just show
    title="Simulation: True vs. Koopman (top view)",
    tail_secs=10.0,
    speed=2.0,   # 2x speed playback
    dpi=130
)

print("\nFirst 200 predicted vs. true body positions (m) & orientations (rad):")
for k in range(200):
    print(
        f"t={k*dt:4.2f}s: "
        f"pred=[{pred_traj[k,0]: .3f}, {pred_traj[k,1]: .3f}, {pred_traj[k,2]: .3f}, "
        f"{pred_traj[k,3]: .3f}, {pred_traj[k,4]: .3f}, {pred_traj[k,5]: .3f}], "
        f"true=[{true_traj[k,0]: .3f}, {true_traj[k,1]: .3f}, {true_traj[k,2]: .3f}, "
        f"{true_traj[k,3]: .3f}, {true_traj[k,4]: .3f}, {true_traj[k,5]: .3f}]"
    )
    # print(
    #     f"t={k*dt:4.2f}s: "
    #     f"pred=[{pred_traj[k,6]: .3f}, {pred_traj[k,7]: .3f}, {pred_traj[k,8]: .3f}, "
    #     f"{pred_traj[k,9]: .3f}, {pred_traj[k,10]: .3f}, {pred_traj[k,11]: .3f}], "
    #     f"true=[{true_traj[k,6]: .3f}, {true_traj[k,7]: .3f}, {true_traj[k,8]: .3f}, "
    #     f"{true_traj[k,9]: .3f}, {true_traj[k,10]: .3f}, {true_traj[k,11]: .3f}]"
    # )