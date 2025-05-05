#!/usr/bin/env python3
# demo_bluerov2_matrices.py
#
# Run a short BlueROV2 simulation and print MRB, MA, CRB, CA, C, D.

import numpy as np
from BlueROV2 import (
    BlueROV2, Tether, quat_from_euler, euler_from_quat
)

# -------------------------------------------------------------------------
# 0) helper to recover CRB and CA separately (BlueROV2._coriolis returns sum)
# -------------------------------------------------------------------------
def compute_crb_ca(rov: BlueROV2, nu):
    """
    Re-implement the formulas in BlueROV2._coriolis but return CRB *and* CA.
    """
    u, v, w, p, q, r = nu
    m, Ix, Iy, Iz = rov.m, rov.Ix, rov.Iy, rov.Iz

    # ---------- rigid-body part ------------------------------------------
    CRB = np.zeros((6, 6), float)
    CRB[0,4] =  m*w
    CRB[0,5] = -m*v
    CRB[1,3] = -m*w
    CRB[1,5] =  m*u
    CRB[2,3] =  m*v
    CRB[2,4] = -m*u
    CRB[3,1] =  m*w
    CRB[3,2] = -m*v
    CRB[3,4] = -Iz*r
    CRB[3,5] = -Iy*q
    CRB[4,0] = -m*w
    CRB[4,2] =  m*u
    CRB[4,3] =  Iz*r
    CRB[4,5] =  Ix*p
    CRB[5,0] =  m*v
    CRB[5,1] = -m*u
    CRB[5,3] =  Iy*q
    CRB[5,4] = -Ix*p

    # ---------- added-mass part ------------------------------------------
    Xu, Yv, Zw = rov.Xu_dot, rov.Yv_dot, rov.Zw_dot
    Kp, Mq, Nr = rov.Kp_dot, rov.Mq_dot, rov.Nr_dot

    CA = np.zeros((6, 6), float)
    CA[0,4] = -Zw*w
    CA[0,5] =  Yv*v
    CA[1,3] =  Zw*w
    CA[1,5] = -Xu*u
    CA[2,3] = -Yv*v
    CA[2,4] =  Xu*u
    CA[3,1] = -Zw*w
    CA[3,2] =  Yv*v
    CA[3,4] = -Nr*r
    CA[3,5] =  Mq*q
    CA[4,0] =  Zw*w
    CA[4,2] = -Xu*u
    CA[4,3] =  Nr*r
    CA[4,5] = -Kp*p
    CA[5,0] = -Yv*v
    CA[5,1] =  Xu*u
    CA[5,3] = -Mq*q
    CA[5,4] =  Kp*p

    return CRB, CA


# -------------------------------------------------------------------------
# 1) ROV + (optional) tether
# -------------------------------------------------------------------------
rov = BlueROV2()

use_tether = False                       # leave off for this demo
if use_tether:
    rov.use_tether = True
    rov.tether     = Tether(n_segments=5, length=20.0)
    rov.anchor_pos = np.array([0.0, 0.0, 0.0])
    rov.tether_state = rov.tether.init_nodes_line(
        rov.anchor_pos, np.array([0.0, 0.0, 5.0])
    )

# -------------------------------------------------------------------------
# 2) initial state  (z = 5 m depth, level attitude, rest)
# -------------------------------------------------------------------------
x = np.zeros(13)
x[2]      = 5.0
x[3:7]    = quat_from_euler(0, 0, 0)

# -------------------------------------------------------------------------
# 3) thruster command  (mixed: 4 × horiz at 0, 4 × vert ±0.5)
# -------------------------------------------------------------------------
u_thrusters = np.array([0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5])

# -------------------------------------------------------------------------
# 4) integrate for 5 s with dt = 0.01
# -------------------------------------------------------------------------
dt, t_end = 0.01, 0.1
n_steps   = int(t_end / dt)

for _ in range(n_steps):
    xdot = rov.dynamics(x, u_thrusters, dt)
    x   += dt * xdot

# -------------------------------------------------------------------------
# 5) gather matrices after motion built up
# -------------------------------------------------------------------------
nu_now   = x[7:13]
CRB, CA  = compute_crb_ca(rov, nu_now)
C        = CRB + CA
D        = rov._damping(nu_now)          # current_speed is zero → ν_r = ν

# -------------------------------------------------------------------------
# 6) pretty-print
# -------------------------------------------------------------------------
np.set_printoptions(precision=4, suppress=True)

print("\n=== Rigid-body mass matrix  MRB ===")
print(rov.MRB, "\n")

print("=== Added-mass matrix       MA  ===")
print(rov.MA,  "\n")

print("=== Coriolis (rigid body)   CRB ===")
print(CRB,     "\n")

print("=== Coriolis (added mass)   CA  ===")
print(CA,      "\n")

print("=== Full Coriolis           C   ===")
print(C,       "\n")

print("=== Non-linear damping      D   ===")
print(D,       "\n")
