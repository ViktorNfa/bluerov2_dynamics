import numpy as np
from scipy.integrate import solve_ivp
from BlueROV2 import BlueROV2, Tether

# 1) Create ROV
rov = BlueROV2()

# 2) ROV’s initial state [eta(6), nu(6)], e.g. also put the ROV at z=5
x0 = np.zeros(12)
x0[2] = 5.0

# 3) Optionally enable tether
use_tether = True
if use_tether:
    rov.use_tether = True
    rov.tether = Tether(n_segments=5, length=20.0)
    rov.anchor_pos = np.array([0.0, 0.0, 0.0])  # anchor in NED
    rov_start_ned = x0[0:3]  # ROV start in NED

    # Initialize tether nodes along a straight line, zero velocity
    x_teth_init = rov.tether.init_nodes_line(rov.anchor_pos, rov_start_ned)
    rov.tether_state = x_teth_init

# 4) Some thruster command (the input is voltage normalized to [-1,1])
u_thrusters = np.array([0.1, 0.1, 0.1, 0.0, 0.5, 0.5, 0.5, 0.5])

# 5) Solve
def ode_func(t, x):
    # dt=0.0 => solver manages timesteps
    return rov.dynamics(x, u_thrusters, dt=0.0)

t_end = 5.0
t_eval = np.arange(0, t_end, 0.01)

print(f"Starting ODE integration for t=[0..{t_end}] at dt={0.01}")

sol = solve_ivp(
    fun=ode_func,
    t_span=(0,t_end),
    y0=x0,
    t_eval=t_eval,
    method='BDF',
    rtol=1e-6, atol=1e-8
)

if not sol.success:
    print("Integration failed:", sol.message)

for i, t in enumerate(sol.t):
    x_, y_, z_, phi_, theta_, psi_ = sol.y[0,i], sol.y[1,i], sol.y[2,i], sol.y[3,i], sol.y[4,i], sol.y[5,i]
    print(f"Time={t:.2f}, pos=({x_:.2f}, {y_:.2f}, {z_:.2f}, {phi_:.2f}, {theta_:.2f}, {psi_:.2f})")