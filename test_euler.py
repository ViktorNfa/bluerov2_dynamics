import numpy as np
from BlueROV2 import BlueROV2, Tether

# 1) Create ROV
rov = BlueROV2()

# 2) ROV’s initial state [eta(6), nu(6)], e.g. also put the ROV at z=5 in navigation (n) frame
x = np.zeros(12)
x[2] = 5.0

# 3) Optionally enable tether
use_tether = True
if use_tether:
    rov.use_tether = True
    rov.tether = Tether(n_segments=5, length=20.0)
    rov.anchor_pos = np.array([0.0, 0.0, 0.0])  # anchor in NED
    rov_start_ned = x[0:3]  # ROV start in NED

    # Initialize tether nodes along a straight line, zero velocity
    x_teth_init = rov.tether.init_nodes_line(rov.anchor_pos, rov_start_ned)
    rov.tether_state = x_teth_init

# 4) Some thruster command (the input is voltage normalized to [-1,1])
u_thrusters = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])

# 5) Simple Euler integration parameters
dt = 0.01
t_end = 5.0
n_steps = int(t_end / dt)

print(f"Starting Euler integration for t=[0..{t_end}] at dt={dt}")

# 6) Euler Integration Loop
for step in range(n_steps):
    # 6a) Get state derivative
    xdot = rov.dynamics(x, u_thrusters, dt)
    # 6b) Euler update
    x += dt * xdot

    # 6c) Print 
    t = step*dt
    # print(f"Time={t:.2f}, pos=({x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f}, {x[3]:.2f}, {x[4]:.2f}, {x[5]:.2f})")