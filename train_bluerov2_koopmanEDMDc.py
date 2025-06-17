"""
End-to-end example:
  1) Simulate the BlueROV2 physics model to generate data
  2) Fit a EDMDc Koopman model with 200 RBFs
  3) Report one-step RMSE on a held-out test set
  4) Show a short multi-step prediction versus ground truth
"""

import numpy as np
from Koopman.koopmanEDMDc import KoopmanEDMDc
from BlueROV2 import BlueROV2


# ------------------------------------------------------------------
#  1)  Generate a data set
# ------------------------------------------------------------------
np.random.seed(42)
dt = 0.05                   # 20 Hz sampling
T_total = 1200.0            # 20 min rollout
N = int(T_total / dt)

rov = BlueROV2(dt=dt)

# random but smooth thruster commands
def random_input(prev):
    alpha = 0.98
    noise = 0.02 * np.random.randn(rov.n_thrusters)
    return np.clip(alpha * prev + noise, -1.0, 1.0)

states = np.zeros((N, 12))
inputs = np.zeros((N, 8))

x = np.zeros(12)            # initial state (rest at origin)
u_prev = np.zeros(8)

for k in range(N):
    u = random_input(u_prev)
    dx = rov.dynamics(x, u, dt)
    x = x + dt * dx
    states[k] = x
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
    ridge=1e-8
)

model.fit(X_train, U_train)

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

print("\nFirst 200 predicted vs. true body positions (m) & orientations (deg):")
for k in range(200):
    # print(f"t={k*dt:4.2f}s: pred=[{pred_traj[k,0]: .3f}, {pred_traj[k,1]: .3f}, {pred_traj[k,2]: .3f}, {pred_traj[k,3]: .3f}, {pred_traj[k,4]: .3f}, {pred_traj[k,5]: .3f}], true=[{true_traj[k,0]: .3f}, {true_traj[k,1]: .3f}, {true_traj[k,2]: .3f}, {true_traj[k,3]: .3f}, {true_traj[k,4]: .3f}, {true_traj[k,5]: .3f}]")
    print(
        f"t={k*dt:4.2f}s: "
        f"pred=[{pred_traj[k,0]: .3f}, {pred_traj[k,1]: .3f}, {pred_traj[k,2]: .3f}, "
        f"{pred_traj[k,3]: .3f}, {pred_traj[k,4]: .3f}, {pred_traj[k,5]: .3f}], "
        f"true=[{true_traj[k,0]: .3f}, {true_traj[k,1]: .3f}, {true_traj[k,2]: .3f}, "
        f"{true_traj[k,3]: .3f}, {true_traj[k,4]: .3f}, {true_traj[k,5]: .3f}]"
    )