#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BlueROV2.py (wrench-input simplified version, quaternion state):

BlueROV2 heavy configuration dynamic model with direct 6D wrench input:
    tau = [Fx, Fy, Fz, Mx, My, Mz] in body frame.

Simplifications compared to the full version:
  - No thruster model (no geometry, no thrust curve, no lag).
  - No tether model.
  - Input is directly a body-frame wrench tau.
  - Hydrodynamics (added mass, damping, restoring) kept from von Benzon et al with some changes (see comments).

Some extra changes compared to full version:
  - Orientation is represented by a unit quaternion.
  - The main dynamics/kinematics do not reference Euler angles.
  - Conversion helpers are provided for external code that wants Euler<->quat utilities.
"""

import numpy as np


# ------------------------- quaternion utilities -------------------------

def quat_normalize(q, eps=1e-12):
    """
    Normalize quaternion q = [qw, qx, qy, qz].
    """
    q = np.asarray(q, dtype=float).reshape(4,)
    n = np.linalg.norm(q)
    if n < eps:
        # Fall back to identity if something went badly wrong
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def quat_to_rotation_matrix(q):
    """
    Quaternion -> rotation matrix R_{b->n}.
    Quaternion convention: q = [qw, qx, qy, qz] (scalar-first).
    R_{n->b} = R^T.
    """
    qw, qx, qy, qz = quat_normalize(q)

    # Standard direction cosine matrix (DCM) for scalar-first quaternion
    R = np.array([
        [1.0 - 2.0*(qy*qy + qz*qz),       2.0*(qx*qy - qz*qw),       2.0*(qx*qz + qy*qw)],
        [      2.0*(qx*qy + qz*qw), 1.0 - 2.0*(qx*qx + qz*qz),       2.0*(qy*qz - qx*qw)],
        [      2.0*(qx*qz - qy*qw),       2.0*(qy*qz + qx*qw), 1.0 - 2.0*(qx*qx + qy*qy)]
    ], dtype=float)
    return R


def quat_multiply(q1, q2):
    """
    Hamilton product q = q1 ⊗ q2 for scalar-first quaternions.
    q1, q2 are [qw, qx, qy, qz].
    """
    w1, x1, y1, z1 = np.asarray(q1, dtype=float).reshape(4,)
    w2, x2, y2, z2 = np.asarray(q2, dtype=float).reshape(4,)
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)


def quat_derivative(q, omega_body):
    """
    Quaternion kinematics with body rates omega = [p, q, r] in body frame.

    q_dot = 0.5 * ( q ⊗ [0, p, q, r] )
    """
    p, qrate, r = np.asarray(omega_body, dtype=float).reshape(3,)
    omega_quat = np.array([0.0, p, qrate, r], dtype=float)
    return 0.5 * quat_multiply(q, omega_quat)


# ------------------------- optional Euler conversion helpers -------------------------
# These are NOT used by the main class. They are provided so external code can
# convert datasets or interfaces that still use Euler angles.

def euler_to_quat(phi, theta, psi):
    """
    Z-Y-X (roll, pitch, yaw) Euler angles -> quaternion [qw, qx, qy, qz].
    """
    phi = float(phi)
    theta = float(theta)
    psi = float(psi)

    c1 = np.cos(phi * 0.5)
    s1 = np.sin(phi * 0.5)
    c2 = np.cos(theta * 0.5)
    s2 = np.sin(theta * 0.5)
    c3 = np.cos(psi * 0.5)
    s3 = np.sin(psi * 0.5)

    # q = qz(psi) ⊗ qy(theta) ⊗ qx(phi)
    qw = c3*c2*c1 + s3*s2*s1
    qx = c3*c2*s1 - s3*s2*c1
    qy = c3*s2*c1 + s3*c2*s1
    qz = s3*c2*c1 - c3*s2*s1
    return quat_normalize([qw, qx, qy, qz])


def quat_to_euler(q):
    """
    Quaternion [qw, qx, qy, qz] -> Z-Y-X (roll, pitch, yaw) Euler angles.

    Note: Euler angles have singularities; this is intended only as a convenience helper.
    """
    qw, qx, qy, qz = quat_normalize(q)

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw*qx + qy*qz)
    cosr_cosp = 1.0 - 2.0 * (qx*qx + qy*qy)
    phi = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (qw*qy - qz*qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    theta = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    psi = np.arctan2(siny_cosp, cosy_cosp)

    return phi, theta, psi

def quat_to_yaw(q):
    qw, qx, qy, qz = quat_normalize(q)
    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


class BlueROV2:
    """
    BlueROV2 heavy configuration dynamic model with direct wrench input.

    State:
        x = [eta, nu] in R^13
        eta = [x, y, z, qw, qx, qy, qz]  (unit quaternion, scalar-first)
        nu  = [u, v, w, p, q, r]         (body-frame velocities)

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
        CRB[3, 4] =  self.Iz * r  # This term is wrong in the paper, but I corrected it based on Fossen Eq. 3.60
        CRB[3, 5] = -self.Iy * q
        CRB[4, 0] = -self.m * w
        CRB[4, 2] =  self.m * u
        CRB[4, 3] = -self.Iz * r  # This term is wrong in the paper, but I corrected it based on Fossen Eq. 3.60
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

    def _restoring_from_R(self, R_b2n):
        """
        Restoring forces/moments from hydrostatics, computed from R_{b->n} directly.

        This matches the original closed-form expressions (which were written in terms of roll/pitch),
        but avoids referencing angles by extracting the required terms from the rotation matrix.

        For Z-Y-X convention, the third row of R_{b->n} is:
            [ -sin(theta),  cos(theta)sin(phi),  cos(theta)cos(phi) ]
        """
        W_minus_B = self.W - self.B

        # Extract the combinations used in the original expressions
        sth      = -R_b2n[2, 0]
        cth_sphi =  R_b2n[2, 1]
        cth_cphi =  R_b2n[2, 2]

        gvec = np.zeros(6, float)
        gvec[0] =  W_minus_B * sth
        gvec[1] = -W_minus_B * cth_sphi
        gvec[2] = -W_minus_B * cth_cphi

        # Moments: same structure as original, expressed with extracted terms.
        gvec[3] =  (self.yb*self.B)*cth_cphi - (self.zb*self.B)*cth_sphi
        gvec[4] = -(self.zb*self.B)*sth - (self.xb*self.B)*cth_cphi
        gvec[5] =  (self.xb*self.B)*cth_sphi + (self.yb*self.B)*sth
        return gvec

    # ------------------------------ API ------------------------------
    def dynamics(self, x, tau_body, dt=0.02):
        """
        Continuous-time dynamics:

            x = [x, y, z, qw, qx, qy, qz, u, v, w, p, q, r]
            tau_body = [Fx, Fy, Fz, Mx, My, Mz] in body frame.

        Returns xdot of shape (13,).
        dt is not used (kept for API compatibility with old code).
        """
        x = np.asarray(x, dtype=float).reshape(13,)
        tau_body = np.asarray(tau_body, dtype=float).reshape(6,)

        # 1) unpack
        pos = x[0:3]
        quat = quat_normalize(x[3:7])
        nu = x[7:13]   # [u, v, w, p, q, r]

        # 2) transforms
        R_b2n = quat_to_rotation_matrix(quat)
        R_n2b = R_b2n.T

        # 3) relative velocity (account for current)
        v_c_b = R_n2b.dot(self.current_speed)
        nu_r = nu.copy()
        nu_r[:3] -= v_c_b

        # 4) hydrodynamic terms
        C = self._coriolis(nu)
        D = self._damping(nu_r)
        gvec = self._restoring_from_R(R_b2n)

        # 5) total external wrench = input wrench
        tau_ext = tau_body

        # 6) solve for nu_dot
        rhs = tau_ext - C.dot(nu) - D.dot(nu_r) - gvec
        nu_dot = self.Minv.dot(rhs)

        # 7) kinematics for eta_dot (position + quaternion)
        pos_dot = R_b2n.dot(nu[0:3])
        quat_dot = quat_derivative(quat, nu[3:6])

        # 8) pack
        x_dot = np.concatenate([pos_dot, quat_dot, nu_dot])
        return x_dot