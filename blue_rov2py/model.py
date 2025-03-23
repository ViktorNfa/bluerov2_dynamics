#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BlueROV2.py:
    Python class modeling a BlueROV2 (heavy configuration) ROV as per:
      - von Benzon, M.; Sørensen, F.F.; Uth, E.; Jouffroy, J.; Liniger, J.; Pedersen, S.
        "An Open-Source Benchmark Simulator: Control of a BlueROV2 Underwater Robot."
        J. Mar. Sci. Eng. 2022, 10, 1898. https://doi.org/10.3390/jmse10121898
      - T.I. Fossen. "Handbook of Marine Craft Hydrodynamics and Motion Control", 2nd ed. Wiley, 2021.

    This version adds an optional tether model that can be turned on/off. See Tether class below.

Author: Victor Nan Fernandez-Ayala
Date:   2025
"""

import numpy as np

def svec(x):
    """Skew-symmetric matrix from 3-vector x."""
    return np.array([
        [0,     -x[2],  x[1]],
        [x[2],   0,    -x[0]],
        [-x[1],  x[0],  0   ]
    ], dtype=float)

def rotation_matrix(phi, theta, psi):
    """
    Basic Z-Y-X Euler-angles rotation matrix R_{n->b}.
    If you store orientation as [phi, theta, psi], define R_{b->n} = R^T.
    """
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth  = np.cos(theta)
    sth  = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    # R_{n->b} = Rz(psi)*Ry(theta)*Rx(phi)
    R = np.array([
        [ cth*cpsi,  cth*spsi, -sth    ],
        [ sphi*sth*cpsi - cphi*spsi, sphi*sth*spsi + cphi*cpsi, sphi*cth ],
        [ cphi*sth*cpsi + sphi*spsi, cphi*sth*spsi - sphi*cpsi, cphi*cth ]
    ], dtype=float)
    return R

def euler_kinematics_matrix(phi, theta):
    """
    Kinematic transformation from body-rates [p, q, r] to Euler-angle rates [phi_dot, theta_dot, psi_dot].
    """
    sphi = np.sin(phi)
    cphi = np.cos(phi)
    tth  = np.tan(theta)
    cth  = np.cos(theta)
    if abs(cth) < 1e-7:
        cth = 1e-7*np.sign(cth)

    return np.array([
        [1.0, sphi*tth,  cphi*tth],
        [0.0, cphi,     -sphi    ],
        [0.0, sphi/cth,  cphi/cth]
    ], dtype=float)


class BlueROV2:
    """
    BlueROV2 heavy configuration dynamic model.

    By default, it does *not* use tether. If you want tether forces:
      1) Set self.use_tether = True,
      2) Assign self.tether = Tether(...),
      3) Assign self.tether_state = <some initial array of shape 6*(n-1)>,
      4) Provide self.anchor_pos as [x0, y0, z0] in NED.

    Then each time you call dynamics(..., dt), the tether solver is stepped and
    a tension force is added in body-frame.
    """

    def __init__(self, rho=1000.0):
        # Physical parameters from the paper's Table A1 (heavy config).
        self.rho = rho
        self.g   = 9.82
        self.m   = 13.5
        self.volume = 0.0134
        self.W = self.m * self.g
        self.B = self.rho*self.g*self.volume

        # CG/CB
        self.xg = 0.0
        self.yg = 0.0
        self.zg = 0.0
        self.xb = 0.0
        self.yb = 0.0
        self.zb = -0.01

        # Inertias
        self.Ix = 0.26
        self.Iy = 0.23
        self.Iz = 0.37

        # Rigid-body mass matrix
        self.MRB = np.zeros((6,6), float)
        self.MRB[0,0] = self.m
        self.MRB[1,1] = self.m
        self.MRB[2,2] = self.m
        self.MRB[3,3] = self.Ix
        self.MRB[4,4] = self.Iy
        self.MRB[5,5] = self.Iz

        # Added mass
        self.Xu_dot =  6.36
        self.Yv_dot =  7.12
        self.Zw_dot = 18.68
        self.Kp_dot =  0.189
        self.Mq_dot =  0.135
        self.Nr_dot =  0.222
        self.MA = np.zeros((6,6), float)
        self.MA[0,0] = self.Xu_dot
        self.MA[1,1] = self.Yv_dot
        self.MA[2,2] = self.Zw_dot
        self.MA[3,3] = self.Kp_dot
        self.MA[4,4] = self.Mq_dot
        self.MA[5,5] = self.Nr_dot

        self.M = self.MRB + self.MA
        self.Minv = np.linalg.inv(self.M)

        # Damping
        self.Xu =   13.7
        self.Xu_abs = 141.0
        self.Yv =    0.0
        self.Yv_abs = 217.0
        self.Zw =   33.0
        self.Zw_abs =190.0
        self.Kp =    0.0
        self.Kp_abs =1.19
        self.Mq =    0.8
        self.Mq_abs =0.47
        self.Nr =    0.0
        self.Nr_abs =1.5

        # Thrusters
        self.n_thrusters = 8
        self.thrusters_r = self._define_thruster_placements()

        # Current in NED
        self.current_speed = np.array([0.0, 0.0, 0.0])

        # ---------------------------------------------------------------------
        # Tether fields (default off).
        self.use_tether    = False
        self.tether        = None      # a Tether object
        self.tether_state  = None      # shape (n-1)*6
        self.anchor_pos    = np.zeros(3)  # top side anchor in NED
        # ---------------------------------------------------------------------

    def _define_thruster_placements(self):
        thruster_list = []

        # Horizontal-plane thrusters T1..T4
        thruster_list.append({
            'r':  np.array([ 0.156,  0.111,  0.0]),
            'dir':np.array([ 1.0/np.sqrt(2), -1.0/np.sqrt(2), 0.0])
        })
        thruster_list.append({
            'r':  np.array([ 0.156, -0.111,  0.0]),
            'dir':np.array([ 1.0/np.sqrt(2),  1.0/np.sqrt(2), 0.0])
        })
        thruster_list.append({
            'r':  np.array([-0.156,  0.111,  0.0]),
            'dir':np.array([-1.0/np.sqrt(2), -1.0/np.sqrt(2), 0.0])
        })
        thruster_list.append({
            'r':  np.array([-0.156, -0.111,  0.0]),
            'dir':np.array([-1.0/np.sqrt(2),  1.0/np.sqrt(2), 0.0])
        })

        # Vertical thrusters T5..T8
        thruster_list.append({
            'r':  np.array([ 0.120,  0.218,  0.0]),
            'dir':np.array([ 0.0,    0.0,   -1.0])
        })
        thruster_list.append({
            'r':  np.array([ 0.120, -0.218,  0.0]),
            'dir':np.array([ 0.0,    0.0,   -1.0])
        })
        thruster_list.append({
            'r':  np.array([-0.120,  0.218,  0.0]),
            'dir':np.array([ 0.0,    0.0,   -1.0])
        })
        thruster_list.append({
            'r':  np.array([-0.120, -0.218,  0.0]),
            'dir':np.array([ 0.0,    0.0,   -1.0])
        })
        return thruster_list

    def _thruster_force_from_input(self, V):
        """Polynomial from the paper for T200 thrusters."""
        V3 = V**3
        V5 = V**5
        V7 = V**7
        V9 = V**9
        return -140.3*V9 + 389.9*V7 - 404.1*V5 + 176.0*V3 + 8.9*V

    def compute_thruster_forces(self, u_thrust):
        tau = np.zeros(6, dtype=float)
        for i in range(self.n_thrusters):
            F_i = self._thruster_force_from_input(u_thrust[i])
            dvec = self.thrusters_r[i]['dir']
            rvec = self.thrusters_r[i]['r']
            f_xyz = F_i * dvec
            m_xyz = np.cross(rvec, f_xyz)
            tau[0:3] += f_xyz
            tau[3:6] += m_xyz
        return tau

    def _coriolis(self, nu):
        u, v, w, p, q, r = nu
        m  = self.m
        Ix = self.Ix
        Iy = self.Iy
        Iz = self.Iz

        CRB = np.zeros((6,6), float)
        # Rigid-body part
        CRB[0,4] =  m*w
        CRB[0,5] = -m*v
        CRB[1,3] = -m*w
        CRB[1,5] =  m*u
        CRB[2,3] =  m*v
        CRB[2,4] = -m*u
        CRB[3,4] =  Iz*r
        CRB[3,5] = -Iy*q
        CRB[4,3] = -Iz*r
        CRB[4,5] =  Ix*p
        CRB[5,3] =  Iy*q
        CRB[5,4] = -Ix*p

        CA = np.zeros((6,6), float)
        Xudot = self.Xu_dot
        Yvdot = self.Yv_dot
        Zwdot = self.Zw_dot
        Kpdot = self.Kp_dot
        Mqdot = self.Mq_dot
        Nrdot = self.Nr_dot

        CA[0,4] = -Zwdot*w
        CA[0,5] =  Yvdot*v
        CA[1,3] =  Zwdot*w
        CA[1,5] = -Xudot*u
        CA[2,3] = -Yvdot*v
        CA[2,4] =  Xudot*u
        CA[3,4] = -Nrdot*r
        CA[3,5] =  Mqdot*q
        CA[4,3] =  Nrdot*r
        CA[4,5] = -Kpdot*p
        CA[5,3] = -Mqdot*q
        CA[5,4] =  Kpdot*p

        return CRB + CA

    def _damping(self, nu_r):
        u, v, w, p, q, r = nu_r
        D = np.zeros((6,6), float)
        D[0,0] = self.Xu + self.Xu_abs*abs(u)
        D[1,1] = self.Yv + self.Yv_abs*abs(v)
        D[2,2] = self.Zw + self.Zw_abs*abs(w)
        D[3,3] = self.Kp + self.Kp_abs*abs(p)
        D[4,4] = self.Mq + self.Mq_abs*abs(q)
        D[5,5] = self.Nr + self.Nr_abs*abs(r)
        return D

    def _restoring(self, phi, theta, psi):
        W_minus_B = self.W - self.B
        gvec = np.zeros(6, float)
        gvec[0] =  W_minus_B * np.sin(theta)
        gvec[1] = -W_minus_B * np.cos(theta)*np.sin(phi)
        gvec[2] = -W_minus_B * np.cos(theta)*np.cos(phi)

        phi_s = np.sin(phi)
        phi_c = np.cos(phi)
        th_s  = np.sin(theta)
        th_c  = np.cos(theta)

        gvec[3] = - (self.yb*self.B)*th_c*phi_c + ( self.zb*self.B)*th_c*phi_s
        gvec[4] = - (self.zb*self.B)*th_s - (self.xb*self.B)*th_c*phi_c
        gvec[5] =   (self.xb*self.B)*th_c*phi_s + (self.yb*self.B)*th_s
        return gvec

    def dynamics(self, x, u_thrust, dt=0.01):
        """
        Main 6-DOF ODE step:
          x = [x, y, z, phi, theta, psi,  u, v, w, p, q, r]
        u_thrust in R^8 => normalized thruster commands in [-1..1].
        dt is included so we can also integrate tether if needed.

        Returns xdot of the same dimension (12,).
        """
        # 1) unpack
        eta = x[0:6]
        nu  = x[6:12]
        phi, theta, psi = eta[3:6]

        # 2) transforms
        R_n2b = rotation_matrix(phi, theta, psi)
        R_b2n = R_n2b.T
        J2    = euler_kinematics_matrix(phi, theta)

        # 3) relative velocity
        v_c_b = R_n2b.dot(self.current_speed)
        nu_r = np.copy(nu)
        nu_r[:3] -= v_c_b

        # 4) system matrices
        Cmat = self._coriolis(nu_r)
        Dmat = self._damping(nu_r)
        gvec = self._restoring(phi, theta, psi)

        # 5) thrusters
        tau_thr = self.compute_thruster_forces(u_thrust)
        tau_ext = np.copy(tau_thr)  # we can add tether or anything else to this

        # 6) Tether logic (optional)
        if self.use_tether and (self.tether is not None) and (self.tether_state is not None):
            # The ROV attachment is at x,y,z from "eta[:3]"
            rov_pos_ned = eta[0:3]
            dx_t, F_teth_ned = self.tether.dynamics(
                self.tether_state,
                self.anchor_pos,
                rov_pos_ned,
                self.current_speed,
                dt
            )
            self.tether_state += dt * dx_t
            # Convert that force to body frame and add
            F_teth_b = R_n2b.dot(F_teth_ned)
            # (For minimal example, we just add force in body; if tether attaches off-CG,
            #  you'd also add a moment = cross(r_attach, F_teth_b).)
            tau_ext[0:3] += F_teth_b

        # 7) solve for nu_dot
        rhs = tau_ext - Cmat.dot(nu_r) - Dmat.dot(nu_r) - gvec
        nu_dot = self.Minv.dot(rhs)

        # 8) compute eta_dot
        p_dot_n = R_b2n.dot(nu[0:3])  # position
        eul_rates = J2.dot(nu[3:6])   # orientation
        eta_dot = np.concatenate([p_dot_n, eul_rates])

        # 9) pack
        x_dot = np.concatenate([eta_dot, nu_dot])
        return x_dot


###############################################################################
# Tether class with a simple lumped-mass solver (optional)
###############################################################################

class Tether:
    """
    Lumped-mass tether model from von Benzon et al. references.
    Node 0: anchor at anchor_pos (fixed).
    Node n: ROV at rov_pos (fixed).
    The internal nodes 1..n-1 are states in x_teth.

    x_teth shape = (n-1)*6 => [p1..p_{n-1}, v1..v_{n-1}],
    each p_i,v_i is a 3D vector in NED.
    
    Example usage:
      tether = Tether(n_segments=5, length=20.0)
      x_teth_init = tether.init_nodes_line(anchor=..., rovpos=...)
      # then store x_teth_init in your main code, e.g. rov.tether_state = x_teth_init
    """

    def __init__(self,
                 n_segments=10,
                 length=35.0,
                 tether_diameter=0.0075,
                 E_modulus=6.437e10,
                 drag_normal=1.2,
                 drag_tangent=0.01,
                 c_internal=100.0,
                 mass_per_length=0.043,
                 rho=1000.0):
        self.n = n_segments
        self.L = length
        self.dtet = tether_diameter
        self.Across = np.pi*(0.5*self.dtet)**2
        self.Et = E_modulus
        self.Cn = drag_normal
        self.Ct = drag_tangent
        self.c_internal = c_internal
        self.mpl = mass_per_length
        self.rho = rho

        self.l0 = self.L / float(self.n)
        self.node_mass = self.mpl * self.l0

    def init_nodes_line(self, anchor, rovpos):
        """
        Place the n-1 internal nodes in a straight line between anchor and rovpos, 
        with zero initial velocities.

        Returns x_teth (np.ndarray) of shape ((n-1)*6).
         - If n < 2, returns an empty array (no internal nodes).
        """
        n_i = self.n - 1
        if n_i < 1:
            return np.zeros(0, dtype=float)

        # Build the list of positions
        p_array = []
        v_array = []
        for i in range(1, self.n):
            # fraction of the way from anchor to rovpos
            alpha = i / float(self.n)
            p_i = anchor + alpha*(rovpos - anchor)
            p_array.append(p_i)
            v_array.append([0.0, 0.0, 0.0])

        p_flat = np.array(p_array).ravel()
        v_flat = np.array(v_array).ravel()
        return np.concatenate([p_flat, v_flat])

    def dynamics(self, x_teth, anchor_pos, rov_pos, current_ned, dt=0.01):
        """
        anchor_pos: (3,) top-side in NED
        rov_pos:    (3,) ROV in NED
        current_ned:(3,) current velocity in NED
        returns: dx_teth, F_teth_ned
        """
        if (self.n < 2):
            # no internal nodes => no tension
            return np.zeros_like(x_teth), np.zeros(3)

        num_i = self.n - 1
        p_list = []
        v_list = []
        for i in range(num_i):
            ip = 3*i
            iv = 3*num_i + 3*i
            pi = x_teth[ip:ip+3]
            vi = x_teth[iv:iv+3]
            p_list.append(pi)
            v_list.append(vi)

        def _get_pos(i):
            if i == 0:
                return anchor_pos
            elif i == self.n:
                return rov_pos
            else:
                return p_list[i-1]

        def _get_vel(i):
            if (i == 0) or (i == self.n):
                return np.zeros(3)
            else:
                return v_list[i-1]

        # Example tension law that clamps to zero if dist < l0
        def _tension(r):
            dist = np.linalg.norm(r)
            if dist < 1e-9:
                return np.zeros(3)
            # If segment is shorter than nominal => slack => no tension
            if dist <= self.l0:
                return np.zeros(3)
            scale = self.Et*self.Across/self.l0*(1.0 - self.l0/dist)
            return scale*r

        def _drag_force(vi):
            vrel = vi - current_ned
            spd  = np.linalg.norm(vrel)
            if spd<1e-9:
                return np.zeros(3)
            area = self.dtet*self.l0
            Cd   = self.Cn
            F = 0.5*self.rho*Cd*area*(spd**2)
            return F*(vrel/spd)

        def _damping(vi):
            return -self.c_internal*vi

        dp_list = []
        dv_list = []
        for i in range(1, self.n):
            if i == self.n:
                continue
            pi   = _get_pos(i)
            vi   = _get_vel(i)
            pim1 = _get_pos(i-1)
            pip1 = _get_pos(i+1)

            r_i_im1 = pi - pim1
            Ti = _tension(r_i_im1)
            r_ip1_i = pip1 - pi
            Tip1 = _tension(r_ip1_i)

            F_net = (Tip1 - Ti) - _drag_force(vi) + _damping(vi)
            a_i   = F_net / self.node_mass
            dp_list.append(vi)
            dv_list.append(a_i)

        dp_flat = np.array(dp_list).ravel()
        dv_flat = np.array(dv_list).ravel()
        dx_teth = np.concatenate([dp_flat, dv_flat])

        # Tension on ROV => from segment n
        p_n_1 = _get_pos(self.n - 1)
        p_n   = _get_pos(self.n)
        r_n_1_n = p_n - p_n_1
        Tn = _tension(r_n_1_n)
        return dx_teth, Tn