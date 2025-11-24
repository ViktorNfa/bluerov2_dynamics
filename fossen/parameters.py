''' Parameters '''

m = 11.4    # BlueROV2 mass (kg) 
g = 9.82  # gravitational field strength (m/s^2)

F_bouy = 1026 * 0.0115 * g # Bouyancy force (N)

X_ud = -2.6 # Added mass in x direction (kg)
Y_vd = -18.5 # Added mass in y direction (kg)
Z_wd = -13.3 # Added mass in z direction (kg)
K_pd = -0.054 # Added mass for rotation about x direction (kg)
M_qd = -0.0173 # Added mass for rotation about y direction (kg)
N_rd = -0.28  # Added mass for rotation about z direction (kg)

I_xx = 0.21 # Moment of inertia (kg.m^2)
I_yy = 0.245 # Moment of inertia (kg.m^2)
I_zz = 0.245 # Moment of inertia (kg.m^2)

X_u  = -0.09 # Linear damping coefficient in x direction (N.s/m)
Y_v  = -0.26 # Linear damping coefficient  in y direction (N.s/m)
Z_w = -0.19 # Linear damping coefficient  in z direction (N.s/m)
K_p = -0.895 # Linear damping coefficient for rotation about z direction (N.s/rad)
M_q =  -0.287 # Linear damping coefficient for rotation about z direction (N.s/rad)
N_r  = -4.64 # Linear damping coefficient for rotation about z direction (N.s/rad)

X_uc = -34.96  # quadratic damping coefficient in x direction (N.s^2/m^2)
Y_vc = -103.25 # quadratic damping coefficient  in y direction (N.s^2/m^2)
Z_wc = -74.23 # quadratic damping coefficient  in z direction (N.s^2/m^2)
K_pc = -0.084 # quadratic damping coefficient for rotation about x direction (N.s^2/rad^2)
M_qc = -0.028 # quadratic damping coefficient for rotation about y direction (N.s^2/rad^2)
N_rc = - 0.43 # quadratic damping coefficient for rotation about z direction (N.s^2/rad^2)

z_b = -0.1 # Distance between cb and cg along the z-axis.