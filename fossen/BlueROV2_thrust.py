#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BlueROV2.py (wrench-input simplified version):

BlueROV2 heavy configuration dynamic model with direct 6D wrench input:
    tau = [Fx, Fy, Fz, Mx, My, Mz] in body frame.

Simplifications compared to the full version:
  - No thruster model (no geometry, no thrust curve, no lag).
  - No tether model.
  - Input is directly a body-frame wrench tau.
  - Hydrodynamics (added mass, damping, restoring) kept from von Benzon et al.
"""

import numpy as np


def rotation_matrix(phi, theta, psi):
    """
    Z-Y-X Euler-angles rotation matrix R_{b->n}.
    Orientation is [phi, theta, psi] (roll, pitch, yaw).
    R_{n->b} = R^T.
    """
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth  = np.cos(theta)
    sth  = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    # R_{b->n} = Rz(psi)*Ry(theta)*Rx(phi)
    R = np.array([
        [cpsi*cth, -spsi*cphi + cpsi*sth*sphi,  spsi*sphi + cpsi*cphi*sth],
        [spsi*cth,  cpsi*cphi + sphi*sth*spsi, -cpsi*sphi + sth*spsi*cphi],
        [-sth,      cth*sphi,                  cth*cphi]
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

    if abs(cth) < eps:
        cth = eps * np.sign(cth)

    tth = sth / cth

    return np.array([
        [1.0,  sphi*tth,  cphi*tth],
        [0.0,  cphi,     -sphi],
        [0.0,  sphi/cth,  cphi/cth]
    ], dtype=float)


class BlueROV2:
    """
    BlueROV2 heavy configuration dynamic model with direct wrench input.

    State:
        x = [eta, nu] in R^12
        eta = [x, y, z, phi, theta, psi]
        nu  = [u, v, w, p, q, r] (body-frame velocities)

    Input:
        tau = [Fx, Fy, Fz, Mx, My, Mz] in body frame (units consistent
        with the hydrodynamic parameters, typically N and N·m).

    Method:
        dynamics(x, tau, dt) -> xdot
    """

    def __init__(self, rho=1000.0, current_speed=None):
        # Physical parameters from von Benzon et al (heavy config).
        self.rho = rho
        self.g = 9.82
        self.m = 13.5
        self.volume = 0.0134
        self.W = self.m * self.g
        self.B = self.rho * self.g * self.volume

        # CG / CB
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
        self.MRB = np.zeros((6, 6), float)
        self.MRB[0, 0] = self.m
        self.MRB[1, 1] = self.m
        self.MRB[2, 2] = self.m
        self.MRB[3, 3] = self.Ix
        self.MRB[4, 4] = self.Iy
        self.MRB[5, 5] = self.Iz

        # Added mass (paper forgets to add a minus sign for these terms)
        self.Xu_dot = -6.36
        self.Yv_dot = -7.12
        self.Zw_dot = -18.68
        self.Kp_dot = -0.189
        self.Mq_dot = -0.135
        self.Nr_dot = -0.222

        self.MA = np.zeros((6, 6), float)
        self.MA[0, 0] = -self.Xu_dot
        self.MA[1, 1] = -self.Yv_dot
        self.MA[2, 2] = -self.Zw_dot
        self.MA[3, 3] = -self.Kp_dot
        self.MA[4, 4] = -self.Mq_dot
        self.MA[5, 5] = -self.Nr_dot

        self.M = self.MRB + self.MA
        self.Minv = np.linalg.inv(self.M)

        # Damping (linear + quadratic) (paper forgets to add a minus sign for these terms)
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

        # Current in NED (assume irrotational, constant speed)
        if current_speed is None:
            current_speed = np.zeros(3, dtype=float)
        self.current_speed = np.asarray(current_speed, dtype=float).reshape(3,)

    # ------------------------- internal helpers -------------------------
    def _coriolis(self, nu):
        """
        6x6 Coriolis-centripetal matrix for MRB + MA (approx).
        """
        u, v, w, p, q, r = nu

        CRB = np.zeros((6, 6), float)
        # Rigid-body part
        CRB[0, 4] =  self.m * w
        CRB[0, 5] = -self.m * v
        CRB[1, 3] = -self.m * w
        CRB[1, 5] =  self.m * u
        CRB[2, 3] =  self.m * v
        CRB[2, 4] = -self.m * u
        CRB[3, 1] =  self.m * w
        CRB[3, 2] = -self.m * v
        CRB[3, 4] =  self.Iz * r # I think this term is wrong in the paper, based on Fossen Eq. 3.60
        CRB[3, 5] = -self.Iy * q
        CRB[4, 0] = -self.m * w
        CRB[4, 2] =  self.m * u
        CRB[4, 3] = -self.Iz * r # I think this term is wrong in the paper, based on Fossen Eq. 3.60
        CRB[4, 5] =  self.Ix * p
        CRB[5, 0] =  self.m * v
        CRB[5, 1] = -self.m * u
        CRB[5, 3] =  self.Iy * q
        CRB[5, 4] = -self.Ix * p

        CA = np.zeros((6, 6), float)
        # Hydrodynamic part
        CA[0, 4] = -self.Zw_dot * w
        CA[0, 5] =  self.Yv_dot * v
        CA[1, 3] =  self.Zw_dot * w
        CA[1, 5] = -self.Xu_dot * u
        CA[2, 3] = -self.Yv_dot * v
        CA[2, 4] =  self.Xu_dot * u
        CA[3, 1] = -self.Zw_dot * w
        CA[3, 2] =  self.Yv_dot * v
        CA[3, 4] = -self.Nr_dot * r
        CA[3, 5] =  self.Mq_dot * q
        CA[4, 0] =  self.Zw_dot * w
        CA[4, 2] = -self.Xu_dot * u
        CA[4, 3] =  self.Nr_dot * r
        CA[4, 5] = -self.Kp_dot * p
        CA[5, 0] = -self.Yv_dot * v
        CA[5, 1] =  self.Xu_dot * u
        CA[5, 3] = -self.Mq_dot * q
        CA[5, 4] =  self.Kp_dot * p

        return CRB + CA

    def _damping(self, nu_r):
        """
        Diagonal linear+quadratic damping with relative velocity nu_r.
        """
        u_r, v_r, w_r, p_r, q_r, r_r = nu_r

        D = np.zeros((6, 6), float)
        D[0, 0] = -self.Xu - self.Xu_abs * abs(u_r)
        D[1, 1] = -self.Yv - self.Yv_abs * abs(v_r)
        D[2, 2] = -self.Zw - self.Zw_abs * abs(w_r)
        D[3, 3] = -self.Kp - self.Kp_abs * abs(p_r)
        D[4, 4] = -self.Mq - self.Mq_abs * abs(q_r)
        D[5, 5] = -self.Nr - self.Nr_abs * abs(r_r)
        return D

    def _restoring(self, phi, theta, psi):
        """
        Restoring forces/moments from hydrostatics.
        """
        W_minus_B = self.W - self.B
        sphi = np.sin(phi)
        cphi = np.cos(phi)
        sth  = np.sin(theta)
        cth  = np.cos(theta)

        gvec = np.zeros(6, float)
        gvec[0] =  W_minus_B * sth
        gvec[1] = -W_minus_B * cth * sphi
        gvec[2] = -W_minus_B * cth * cphi
        gvec[3] =  (self.yb*self.B)*cth*cphi - (self.zb*self.B)*cth*sphi
        gvec[4] = -(self.zb*self.B)*sth - (self.xb*self.B)*cth*cphi
        gvec[5] =  (self.xb*self.B)*cth*sphi + (self.yb*self.B)*sth
        return gvec

    # ------------------------------ API ------------------------------
    def dynamics(self, x, tau_body, dt):
        """
        Continuous-time dynamics:

            x = [x, y, z, phi, theta, psi, u, v, w, p, q, r]
            tau_body = [Fx, Fy, Fz, Mx, My, Mz] in body frame.

        Returns xdot of shape (12,).
        dt is not used (kept for API compatibility with old code).
        """
        x = np.asarray(x, dtype=float).reshape(12,)
        tau_body = np.asarray(tau_body, dtype=float).reshape(6,)

        # 1) unpack
        eta = x[0:6]
        nu  = x[6:12]
        phi, theta, psi = eta[3:6]

        # 2) transforms
        R_b2n = rotation_matrix(phi, theta, psi)
        R_n2b = R_b2n.T
        J2 = euler_kinematics_matrix(phi, theta)

        # 3) relative velocity (account for current)
        v_c_b = R_n2b.dot(self.current_speed)
        nu_r = nu.copy()
        nu_r[:3] -= v_c_b

        # 4) hydrodynamic terms
        C = self._coriolis(nu)
        D = self._damping(nu_r)
        gvec = self._restoring(phi, theta, psi)

        # 5) total external wrench = input wrench
        tau_ext = tau_body

        # 6) solve for nu_dot
        rhs = tau_ext - C.dot(nu) - D.dot(nu_r) - gvec
        nu_dot = self.Minv.dot(rhs)

        # 7) kinematics for eta_dot
        p_dot_n = R_b2n.dot(nu[0:3])
        eul_rates = J2.dot(nu[3:6])
        eta_dot = np.concatenate([p_dot_n, eul_rates])

        # 8) pack
        x_dot = np.concatenate([eta_dot, nu_dot])
        return x_dot