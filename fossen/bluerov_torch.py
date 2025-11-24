import torch
from fossen.parameters import (
    m, X_ud, Y_vd, Z_wd, I_zz, N_rd,
    X_u, X_uc, Y_v, Y_vc, Z_w, Z_wc,
    N_r, N_rc, g, F_bouy
)

def ssa(angle):
    """
    Computes the smallest signed angle (normalize angle to [-pi, pi]).
    
    Args:
        angle (Tensor): Angle in radians.
        
    Returns:
        Tensor: Normalized angle within [-pi, pi].
    """
    return angle - 2 * torch.pi * torch.floor_divide(angle + torch.pi, 2 * torch.pi)

def bluerov_compute(t, x_, u_):
    """
    Computes the time derivative of the state for the BlueROV underwater vehicle.
    
    Args:
        x_ (Tensor): State tensor of shape (batch_size, 9).
        u_ (Tensor): Control input tensor of shape (batch_size, 4).
        
    Returns:
        Tensor: Time derivative of the state, shape (batch_size, 9).
    """
    # Ensure x_ and u_ are at least 2D tensors with batch dimension as dim=0
    x_ = x_.unsqueeze(0) if x_.dim() == 1 else x_
    u_ = u_.unsqueeze(0) if u_.dim() == 1 else u_

    # State variables
    eta = x_[:, :5]  # Position and orientation
    nu = x_[:, 5:]   # Linear and angular velocities
    cos_psi = eta[:, 3]
    sin_psi = eta[:, 4]

    # Extract velocities and torques
    u, v, w, r = nu[:, 0], nu[:, 1], nu[:, 2], nu[:, 3]
    X, Y, Z, M_z = u_[:, 0], u_[:, 1], u_[:, 2], u_[:, 3]

    x_d = cos_psi * u - sin_psi * v
    y_d = sin_psi * u + cos_psi * v
    z_d = w

    # Correct derivatives of cos(psi) and sin(psi)
    cos_psi_d = -sin_psi * r
    sin_psi_d = cos_psi * r

    # Use the correct derivatives in eta_dot
    eta_dot = torch.stack([x_d, y_d, z_d, cos_psi_d, sin_psi_d], dim=1)

    # Compute nu_dot as before
    u_d = 1 / (m - X_ud) * (X + (m - Y_vd) * v * r + (X_u + X_uc * abs(u)) * u) # experiments was carried out with error on the sign of Y_vd - should have been negative but was positive in the experiments    
    v_d = 1 / (m - Y_vd) * (Y - (m - X_ud) * u  * r + (Y_v + Y_vc * abs(v)) * v) # experiments was carried out with error on the sign of X_ud - should have been negative but was positive in the experiments  
    w_d = 1 / (m - Z_wd) * (Z + (Z_w + Z_wc * abs(w)) * w + m * g - F_bouy)
    r_d = 1 / (I_zz - N_rd) * (M_z - (X_ud - Y_vd) * u * v + (N_r + N_rc * abs(r)) * r)

    nu_dot = torch.stack([u_d, v_d, w_d, r_d], dim=1)

    # Combine eta_dot and nu_dot
    x_dot = torch.cat([eta_dot, nu_dot], dim=1)

    return x_dot
