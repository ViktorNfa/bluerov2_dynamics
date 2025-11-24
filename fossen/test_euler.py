import numpy as np
from BlueROV2 import BlueROV2

# 1) Create ROV
dt = 0.01
rov = BlueROV2(dt=dt)
# rov = BlueROV2(dt=dt, input_mode='voltage', use_thruster_lag=True)

# 2) ROV’s initial state [eta(6), nu(6)], e.g. also put the ROV at z=5 in navigation (n) frame
x = np.zeros(12)
x[2] = 5.0

# 3) Tether calculation does not work in Euler integration, so disable it for now (default)
# rov.use_tether = False

# 4) Some thruster command (the input is voltage normalized to [-1,1])
u_thrusters = np.array([0.1, 0.1, 0.1, 0.0, 0.5, 0.5, 0.5, 0.5])

# 5) Simple Euler integration parameters
t_end = 5.0
n_steps = int(t_end / dt)

print(f"Starting Euler integration for t=[0...{t_end}] at dt={dt}")

# 6) Euler Integration Loop
xdot = np.zeros(12)  # Initialize state derivative
for step in range(n_steps):
    # 6a) Get state derivative
    xdot = rov.dynamics(x, u_thrusters, dt)
    # 6b) Euler update
    x += dt * xdot

    # 6c) Print 
    t = step*dt
    print(f"Time={t:.2f}, pos=({x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f}, {x[3]:.2f}, {x[4]:.2f}, {x[5]:.2f})")