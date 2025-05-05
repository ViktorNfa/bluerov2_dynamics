#!/usr/bin/env python3
# demo_sam_matrices_iter.py
#
# Initialise a SAM AUV, run a few integration steps with non-zero controls,
# and print MRB, MA, CRB, CA, C, and D.

import numpy as np
from SAM import SAM
from gnc import m2c                     # Coriolis helper

# --------------------------------------------------------------------------
# 1. Vehicle + helpers
# --------------------------------------------------------------------------
sam = SAM(dt=0.01)                      # 10 ms timestep

# state vector x = [eta(7), nu(6), u_actuators(6)]
x = np.zeros(19)
x[3] = 1.0                              # identity quaternion

# simple constant command:
# [x_vbs  x_lcg  δ_s  δ_r  rpm1  rpm2]
u_cmd = np.array([0, 0, 0, 0,  800, -800], dtype=float)

# --------------------------------------------------------------------------
# 2. Integrate for N steps (forward Euler is fine for a quick demo)
# --------------------------------------------------------------------------
N = 10
for _ in range(N):
    x_dot = sam.dynamics(x, u_cmd)      # updates sam’s internal fields too
    x    += sam.dt * x_dot

# --------------------------------------------------------------------------
# 3. Extract matrices after motion has built up a bit
# --------------------------------------------------------------------------
CRB = m2c(sam.MRB, sam.nu_r)            # rigid-body Coriolis
CA  = m2c(sam.MA,  sam.nu_r)            # added-mass Coriolis

np.set_printoptions(precision=4, suppress=True)

print("=== Rigid-body mass matrix  MRB ===")
print(sam.MRB, "\n")

print("=== Added-mass matrix       MA  ===")
print(sam.MA,  "\n")

print("=== Coriolis (rigid body)   CRB ===")
print(CRB,     "\n")

print("=== Coriolis (added mass)   CA  ===")
print(CA,      "\n")

print("=== Full Coriolis           C   ===")
print(sam.C,   "\n")

print("=== Non-linear damping      D   ===")
print(sam.D,   "\n")