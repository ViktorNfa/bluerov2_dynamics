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


def rotation_matrix(phi, theta, psi):
    """
    Basic Z-Y-X Euler-angles rotation matrix R_{b->n}.
    Storing orientation as [phi, theta, psi]. Note that R_{n->b} = R^T.
    """
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth  = np.cos(theta)
    sth  = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    # R_{b->n} = Rz(psi)*Ry(theta)*Rx(phi)
    R = np.array([
        [cpsi*cth, -spsi*cphi + cpsi*sth*sphi, spsi*sphi + cpsi*cphi*sth],
        [spsi*cth, cpsi*cphi + sphi*sth*spsi, -cpsi*sphi + sth*spsi*cphi],
        [-sth, cth*sphi, cth*cphi]
    ], dtype=float)
    return R

def euler_kinematics_matrix(phi, theta, eps=1e-7):
    """
    Kinematic transformation from body-rates [p, q, r] to Euler-angle rates [phi_dot, theta_dot, psi_dot].
    """
    sphi = np.sin(phi)
    cphi = np.cos(phi)
    sth  = np.sin(theta)
    cth  = np.cos(theta)

    # Clamp cosine before using it
    if abs(cth) < 1e-7:
        cth = 1e-7*np.sign(cth)
        
    tth  = sth / cth

    return np.array([
        [1.0, sphi*tth, cphi*tth],
        [0.0, cphi, -sphi],
        [0.0, sphi/cth, cphi/cth]
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

    def __init__(self, rho=1000.0, current_speed=np.array([0.0, 0.0, 0.0])):
        # Physical parameters from the paper's Table A1 (heavy config).
        self.rho = rho
        self.g = 9.82
        self.m = 13.5
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

        # Added mass (paper forgets to add a minus sign for these terms)
        self.Xu_dot = -6.36
        self.Yv_dot = -7.12
        self.Zw_dot = -18.68
        self.Kp_dot = -0.189
        self.Mq_dot = -0.135
        self.Nr_dot = -0.222
        self.MA = np.zeros((6,6), float)
        self.MA[0,0] = -self.Xu_dot
        self.MA[1,1] = -self.Yv_dot
        self.MA[2,2] = -self.Zw_dot
        self.MA[3,3] = -self.Kp_dot
        self.MA[4,4] = -self.Mq_dot
        self.MA[5,5] = -self.Nr_dot

        self.M = self.MRB + self.MA
        self.Minv = np.linalg.inv(self.M)

        # Damping (paper forgets to add a minus sign for these terms)
        self.Xu = -13.7
        self.Xu_abs = -141.0
        self.Yv = -0.0
        self.Yv_abs = -217.0
        self.Zw = -33.0
        self.Zw_abs = -190.0
        self.Kp = -0.0
        self.Kp_abs = -1.19
        self.Mq = -0.8
        self.Mq_abs = -0.47
        self.Nr = -0.0
        self.Nr_abs = -1.5

        # Thrusters
        self.n_thrusters = 8
        self.thrusters_r = self._define_thruster_placements()

        # Current in NED (assume irrotational, constant speed)
        self.current_speed = current_speed

        # ---------------------------------------------------------------------
        # Tether fields (default off).
        self.use_tether    = False
        self.tether        = None           # a Tether object
        self.tether_state  = None           # shape (n-1)*6
        self.anchor_pos    = np.zeros(3)    # top side anchor in NED
        # ---------------------------------------------------------------------

    def _thruster_rotational_matrix(self, alpha):
        """
        The thrusters are located in a circular pattern and the rotation matrix used is denoted.
        """
        salp = np.sin(alpha)
        calp = np.cos(alpha)

        return np.array([
            [calp, -salp, 0.0],
            [salp, calp, 0.0],
            [0.0, 0.0,  1.0]
        ], dtype=float)

    def _define_thruster_placements(self):
        thruster_list = []

        # Thrusters positions
        r1234 = np.array([0.156, 0.111, 0.085])
        r5678 = np.array([0.12, 0.218, 0.0])

        # Thrusters orientations
        e1234 = np.array([1.0/np.sqrt(2), -1.0/np.sqrt(2), 0.0])

        # Thrusters rotational matrices
        J3_r1 = self._thruster_rotational_matrix(0.0)
        J3_r2 = self._thruster_rotational_matrix(5.05)
        J3_r3 = self._thruster_rotational_matrix(1.91)
        J3_r4 = self._thruster_rotational_matrix(np.pi)
        J3_r5 = self._thruster_rotational_matrix(0.0)
        J3_r6 = self._thruster_rotational_matrix(4.15)
        J3_r7 = self._thruster_rotational_matrix(1.01)
        J3_r8 = self._thruster_rotational_matrix(np.pi)

        J3_e1 = self._thruster_rotational_matrix(0.0)
        J3_e2 = self._thruster_rotational_matrix(np.pi/2)
        J3_e3 = self._thruster_rotational_matrix(3*np.pi/2) # paper has the wrong angle
        J3_e4 = self._thruster_rotational_matrix(np.pi)

        # Horizontal-plane thrusters T1..T4
        thruster_list.append({
            'r':    np.dot(J3_r1, r1234),
            'dir':  np.dot(J3_e1, e1234)
        })
        thruster_list.append({
            'r':    np.dot(J3_r2, r1234),
            'dir':  np.dot(J3_e2, e1234)
        })
        thruster_list.append({
            'r':    np.dot(J3_r3, r1234),
            'dir':  np.dot(J3_e3, e1234)
        })
        thruster_list.append({
            'r':    np.dot(J3_r4, r1234),
            'dir':  np.dot(J3_e4, e1234)
        })

        # Vertical thrusters T5..T8
        thruster_list.append({
            'r':    np.dot(J3_r5, r5678),
            'dir':  np.array([0.0, 0.0, -1.0])
        })
        thruster_list.append({
            'r':    np.dot(J3_r6, r5678),
            'dir':  np.array([0.0, 0.0, -1.0])
        })
        thruster_list.append({
            'r':    np.dot(J3_r7, r5678),
            'dir':  np.array([0.0, 0.0, -1.0])
        })
        thruster_list.append({
            'r':    np.dot(J3_r8, r5678),
            'dir':  np.array([0.0, 0.0, -1.0])
        })
        return thruster_list

    def _thruster_force_from_input(self, V):
        """
        Polynomial from the paper for T200 thrusters.
        Note that the input is the voltage V and it's normalized to [-1,1].
        """
        V3 = V**3
        V5 = V**5
        V7 = V**7
        V9 = V**9
        return -140.3*V9 + 389.9*V7 - 404.1*V5 + 176.0*V3 + 8.9*V

    def compute_thruster_forces(self, u_thrust):
        """
        The input is voltage per thruster normalized to [-1,1].
        """
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

        CRB = np.zeros((6,6), float)
        # Rigid-body part
        CRB[0,4] =  self.m*w
        CRB[0,5] = -self.m*v
        CRB[1,3] = -self.m*w
        CRB[1,5] =  self.m*u
        CRB[2,3] =  self.m*v
        CRB[2,4] = -self.m*u
        CRB[3,1] =  self.m*w
        CRB[3,2] = -self.m*v
        CRB[3,4] =  self.Iz*r # I think this term is wrong in the paper, based on Fossen Eq. 3.60
        CRB[3,5] = -self.Iy*q
        CRB[4,0] = -self.m*w
        CRB[4,2] =  self.m*u
        CRB[4,3] = -self.Iz*r # I think this term is wrong in the paper, based on Fossen Eq. 3.60
        CRB[4,5] =  self.Ix*p
        CRB[5,0] =  self.m*v
        CRB[5,1] = -self.m*u
        CRB[5,3] =  self.Iy*q
        CRB[5,4] = -self.Ix*p

        CA = np.zeros((6,6), float)
        # Hydrodynamics part
        CA[0,4] = -self.Zw_dot*w
        CA[0,5] =  self.Yv_dot*v
        CA[1,3] =  self.Zw_dot*w
        CA[1,5] = -self.Xu_dot*u
        CA[2,3] = -self.Yv_dot*v
        CA[2,4] =  self.Xu_dot*u
        CA[3,1] = -self.Zw_dot*w
        CA[3,2] =  self.Yv_dot*v
        CA[3,4] = -self.Nr_dot*r
        CA[3,5] =  self.Mq_dot*q
        CA[4,0] =  self.Zw_dot*w
        CA[4,2] = -self.Xu_dot*u
        CA[4,3] =  self.Nr_dot*r
        CA[4,5] = -self.Kp_dot*p
        CA[5,0] = -self.Yv_dot*v
        CA[5,1] =  self.Xu_dot*u
        CA[5,3] = -self.Mq_dot*q
        CA[5,4] =  self.Kp_dot*p

        return CRB + CA

    def _damping(self, nu_r):
        u_r, v_r, w_r, p_r, q_r, r_r = nu_r

        # Damping matrix
        D = np.zeros((6,6), float)
        D[0,0] = -self.Xu - self.Xu_abs*abs(u_r)
        D[1,1] = -self.Yv - self.Yv_abs*abs(v_r)
        D[2,2] = -self.Zw - self.Zw_abs*abs(w_r)
        D[3,3] = -self.Kp - self.Kp_abs*abs(p_r)
        D[4,4] = -self.Mq - self.Mq_abs*abs(q_r)
        D[5,5] = -self.Nr - self.Nr_abs*abs(r_r)
        return D

    def _restoring(self, phi, theta, psi):
        W_minus_B = self.W - self.B
        sphi = np.sin(phi)
        cphi = np.cos(phi)
        sth  = np.sin(theta)
        cth  = np.cos(theta)

        # Restoring forces and moments (assuming CG is at origin)
        gvec = np.zeros(6, float)
        gvec[0] =  W_minus_B * sth
        gvec[1] = -W_minus_B * cth*sphi
        gvec[2] = -W_minus_B * cth*cphi
        gvec[3] =  (self.yb*self.B)*cth*cphi - (self.zb*self.B)*cth*sphi
        gvec[4] = -(self.zb*self.B)*sth - (self.xb*self.B)*cth*cphi
        gvec[5] =  (self.xb*self.B)*cth*sphi + (self.yb*self.B)*sth
        return gvec

    def dynamics(self, x, u_thrust):
        """
        Main 6-DOF (with velocities) ODE step:
          x = [eta, nu] = [x, y, z, phi, theta, psi,  u, v, w, p, q, r]
        u_thrust in R^8 => normalized (voltage) thruster commands in [-1,1].

        Returns xdot of the same dimension (12,).
        """
        # 1) unpack
        eta = x[0:6]
        nu = x[6:12]
        phi, theta, psi = eta[3:6]

        # 2) transforms
        R_b2n = rotation_matrix(phi, theta, psi) # Also called J1 in the paper
        R_n2b = R_b2n.T
        J2 = euler_kinematics_matrix(phi, theta)

        # 3) relative velocity
        v_c_b = R_n2b.dot(self.current_speed)
        nu_r = np.copy(nu)
        nu_r[:3] -= v_c_b

        # 4) system matrices
        C = self._coriolis(nu)
        D = self._damping(nu_r)
        gvec = self._restoring(phi, theta, psi)

        # 5) thrusters
        tau_thr = self.compute_thruster_forces(u_thrust)
        tau_ext = np.copy(tau_thr)  # we can add anything else to this

        # 6) solve for nu_dot
        rhs = tau_ext - C.dot(nu) - D.dot(nu_r) - gvec
        nu_dot = self.Minv.dot(rhs)

        # 7) compute eta_dot
        p_dot_n = R_b2n.dot(nu[0:3])  # linear velocity in n-frame
        eul_rates = J2.dot(nu[3:6])   # orientation rates in n-frame
        eta_dot = np.concatenate([p_dot_n, eul_rates])

        # 8) pack
        x_dot = np.concatenate([eta_dot, nu_dot])
        return x_dot
    
    def _n_tether_states(self):
        return 0 if (self.tether is None or self.tether.n < 2) else (self.tether.n - 1) * 6
    
    def dynamics_with_tether(self, x, u_thrust):
        """
        x  = [eta, nu, tether_internal(6*(n-1))]
        u_thrust as before.
        Requires:  self.use_tether is True and self.tether is set.
        """
        nt = self._n_tether_states()
        assert self.use_tether and nt > 0, "tether not initialised"

        # 1) split the extended state vector
        x_body = x[:12]
        x_teth = x[12:] # (n-1)*6 <- internal nodes
        eta = x_body[:6]
        nu = x_body[6:]
        phi,theta,psi = eta[3:6]

        # 2) transforms
        R_b2n = rotation_matrix(phi, theta, psi) # Also called J1 in the paper
        R_n2b = R_b2n.T
        J2 = euler_kinematics_matrix(phi, theta)

        # 3) relative velocity
        v_c_b = R_n2b.dot(self.current_speed)
        nu_r = np.copy(nu)
        nu_r[:3] -= v_c_b

        # 4) system matrices
        C = self._coriolis(nu)
        D = self._damping(nu_r)
        gvec = self._restoring(phi, theta, psi)

        # 5) thrusters
        tau_thr = self.compute_thruster_forces(u_thrust)
        tau_ext = np.copy(tau_thr)

        # 6) Tether logic
        rov_pos_ned = eta[:3]
        rov_vel_ned = R_b2n.dot(nu[:3])
        dx_teth, F_teth_ned = self.tether.dynamics(
            x_teth, self.anchor_pos,
            rov_pos_ned, rov_vel_ned,
            self.current_speed
        )

        tau_ext[:3] += R_n2b.dot(F_teth_ned) # add force in body frame

        # 7) solve for nu_dot and eta_dot
        rhs = tau_ext - C.dot(nu) - D.dot(nu_r) - gvec
        nu_dot = self.Minv.dot(rhs)
        eta_dot = np.concatenate([R_b2n.dot(nu[:3]), J2.dot(nu[3:])])

        # 8) pack & return 
        return np.concatenate([eta_dot, nu_dot, dx_teth])


###############################################################################
# Tether class with a simple lumped-mass solver (optional)
###############################################################################

class Tether:
    """
    Lumped-mass tether model from von Benzon et al. references.
    Node 0: anchor at anchor_pos (fixed).
    Node n: ROV at rov_pos (fixed).
    The internal nodes 1...n-1 are states in x_teth.

    x_teth shape = (n-1)*6 => [p1...p_{n-1}, v1...v_{n-1}],
    each p_i or v_i is a 3D vector in NED.
    
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

    def dynamics(self, x_teth, anchor_pos, rov_pos, rov_vel, current_ned):
        """
        anchor_pos : (3,)  fixed top-side point  (node 0)
        rov_pos    : (3,)  ROV position         (node n)
        rov_vel    : (3,)  ROV velocity
        current_ned: (3,)  water-current velocity (expressed in NED)

        Returns
        --------
        dx_teth : time derivative of x_teth  (shape (n-1)*6)
        F_teth  : (3,) tether force on the ROV (τ_tet = T_{n-1})
        """
        # 0) no tether -> nothing to do
        if self.n < 2: 
            return np.zeros_like(x_teth), np.zeros(3)

        n_i = self.n - 1 # number of internal nodes

        # 1) unpack the flattened state
        p_int = x_teth[:3 * n_i].reshape((n_i, 3))
        v_int = x_teth[3 * n_i:].reshape((n_i, 3))

        # full node lists  (0 … n)
        pos = [anchor_pos, *p_int, rov_pos]
        vel = [np.zeros(3), *v_int, rov_vel]

        # 2) pre-compute segment quantities (k = 0 … n-1)
        T = []          # axial tension   (Eq. 36)
        P = []          # internal damping (Eq. 29)
        F = []          # hydrodynamic drag (Eqs. 30-34)

        for k in range(self.n):
            r_k     = pos[k + 1] - pos[k] # vector node k -> k+1
            L_k     = np.linalg.norm(r_k) + 1e-12 # avoid /0
            r_hat   = r_k / L_k

            # axial tension T_k (Eq. 36)
            if L_k > self.l0: # slack → no tension
                T_nominal = (self.Et * self.Across / self.l0) * (1 - self.l0 / L_k) * r_k
                # Lower limit to keep compression-free (no pushing)
                T_min = 0.0 # N  
                # Upper limit to prevent state explosion in euler integration
                T_max = 10000.0 # N
                # T_k = np.clip(T_nominal, T_min, T_max)
                T_k = T_nominal
            else:
                T_k = np.zeros(3)
            T.append(T_k)

            # internal damping P_k (Eq. 29) 
            v_rel_nodes = vel[k + 1] - vel[k]
            P_k = self.c_internal * (np.dot(v_rel_nodes, r_hat)) * r_hat
            P.append(P_k)

            # external drag F_k (Eqs. 31-34)
            v_rel_flow = current_ned - vel[k] # flow at node k
            v_perp     = np.dot(v_rel_flow, r_hat) * r_hat      # Eq. 33
            v_tan      = v_rel_flow - v_perp                    # Eq. 34

            sp_perp = np.linalg.norm(v_perp)
            sp_tan  = np.linalg.norm(v_tan)

            F_perp = 0.5 * self.rho * self.dtet * self.Cn * L_k * sp_perp * v_perp
            F_tan  = 0.5 * self.rho * self.dtet * self.Ct * L_k * sp_tan  * v_tan
            F.append(F_perp + F_tan)                            # Eq. 30

        # 3) node dynamics  (i = 1 … n-1)
        dp_list = []
        dv_list = []

        for i in range(1, self.n):  # internal nodes only
            # forces on node i (Eq. 26 ⟹  F_net = T_i − T_{i−1} + P_{i-1} − P_i + F_i)
            F_net = (
                T[i] - T[i - 1] +
                P[i - 1] - P[i] +
                F[i]
            )

            a_i = F_net / self.node_mass # Mt,i ≈ node_mass
            dp_list.append(vel[i]) # ṗ_i   = v_i
            dv_list.append(a_i) # v̇_i   = a_i

        dx_teth = np.concatenate([np.ravel(dp_list), np.ravel(dv_list)])

        # 4) tether force on the ROV 
        T_rovtet = T[-1] # T_{n-1}
        return dx_teth, T_rovtet