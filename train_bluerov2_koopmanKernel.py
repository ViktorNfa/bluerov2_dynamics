"""
Uses the kernel-Koopman class to learn a bilinear model of the BlueROV2
dynamics and evaluates one-, ten- and 100-step prediction accuracy.
The script mirrors your previous EDMDc example - only two lines change.
"""

import numpy as np
from Koopman.koopmanKernel import KoopmanKernel
from BlueROV2 import BlueROV2

# ------------------------------------------------------------------#
# 1) generate data
# ------------------------------------------------------------------#
np.random.seed(42)
dt = 0.05
T_total = 1200.0
N = int(T_total / dt)

rov = BlueROV2(dt=dt)

def random_input(prev):
    alpha = 0.98
    noise = 0.02 * np.random.randn(rov.n_thrusters)
    return np.clip(alpha * prev + noise, -1.0, 1.0)

states = np.zeros((N, 12))
inputs = np.zeros((N, 8))

x = np.zeros(12)
u_prev = np.zeros(8)

for k in range(N):
    u = random_input(u_prev)
    dx = rov.dynamics(x, u, dt)
    x = x + dt * dx
    states[k] = x
    inputs[k] = u
    u_prev = u

# ------------------------------------------------------------------#
# 2) train / test split
# ------------------------------------------------------------------#
split = int(0.8 * N)
X_train, U_train = states[:split], inputs[:split]
X_test, U_test = states[split - 1 :], inputs[split - 1 :]  # -1 for causality

# ------------------------------------------------------------------#
# 3) fit kernel-Koopman model
# ------------------------------------------------------------------#
model = KoopmanKernel(
    state_dim=12,
    input_dim=8,
    n_inducing=300,      # m
    gamma=1.0,
    ridge=1e-9,
)

model.fit(X_train, U_train)

# ------------------------------------------------------------------#
# 4) quantitative accuracy
# ------------------------------------------------------------------#
print(f"One-step RMSE on test set : {model.evaluate(X_test, U_test):.4f}")

for H in (10, 100):
    print(f"{H:3d}-step RMSE on test set : {model.multistep_rmse(X_test, U_test, H):.4f}")

# ------------------------------------------------------------------#
# 5) open-loop forecast demo
# ------------------------------------------------------------------#
horizon = 200          # 10 s
x0 = X_test[0]
U_seq = U_test[:horizon]

pred_traj = model.simulate(x0, U_seq)
true_traj = states[split - 1 : split - 1 + horizon + 1]

print("\nFirst 20 predicted vs. true states (pos & yaw):")
for k in range(20):
    print(
        f"t={k*dt:4.2f}s "
        f"pred=[{pred_traj[k,0]: .3f}, {pred_traj[k,1]: .3f}, {pred_traj[k,2]: .3f}, {pred_traj[k,5]: .3f}] "
        f"true=[{true_traj[k,0]: .3f}, {true_traj[k,1]: .3f}, {true_traj[k,2]: .3f}, {true_traj[k,5]: .3f}]"
    )