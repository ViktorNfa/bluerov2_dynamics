"""
Nonparametric Control Koopman Operator Regression (cKOR / Ny-cKOR).

Main class: KoopmanCKOR
--------------------------------------------------------------
Methods
    |- fit(X, U)                    learn cKOR predictor from trajectory data
    |- fit_multi(X_list, U_list)    learn from multiple independent trajectories
    |- evaluate(X, U)               one-step RMSE over a test set
    |- multistep_rmse(X, U, H)      strict H-step endpoint RMSE
    `- simulate(x0, U_seq)          multi-step prediction (open-loop)

Key hyper-parameters
    - state_gamma    - Gaussian state kernel width
    - ridge          - Tikhonov regularization coefficient
    - use_nystrom    - if True, fit sketched Ny-cKOR
    - n_nystrom      - nr. inducing points for Ny-cKOR
    - control_kernel - "linear" (default) or "rbf"
--------------------------------------------------------------

Dependencies: numpy, dataclasses, typing

Author: Victor Nan Fernandez-Ayala
Date: 2026
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
from dataclasses import dataclass
from numpy.linalg import pinv
from typing import Sequence


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
def _rbf_mat(X: np.ndarray, C: np.ndarray, gamma: float) -> np.ndarray:
    """
    Fast pair-wise Gaussian RBF between a batch of points X (N,n)
    and centres C (k,n) -> (N,k) matrix.
    """
    x2 = np.sum(X**2, axis=1)[:, None]              # (N,1)
    c2 = np.sum(C**2, axis=1)[None, :]              # (1,k)
    return np.exp(-gamma * (x2 + c2 - 2 * X @ C.T))


# -----------------------------------------------------------------------------#
#  Main class
# -----------------------------------------------------------------------------#
@dataclass
class KoopmanCKOR:
    state_dim: int                     # (n_x)
    input_dim: int                     # (n_u)

    # state/control kernel hyper-parameters
    state_gamma: float = 1.0
    control_kernel: str = "linear"    # {"linear", "rbf"}
    control_gamma: float = 1.0

    # regularisation and sketching
    ridge: float = 1e-8
    use_nystrom: bool = False
    n_nystrom: int = 200
    random_state: int = 0
    jitter: float = 1e-12

    # learned parameters (set by fit)
    A_: np.ndarray = None              # (d, d)
    C_: np.ndarray = None              # (n_x, d)
    lift_dim_: int = None              # (d)
    mode_: str = None                  # {"full", "nystrom"}

    # reference points used by the lifting map z(x,u)
    x_ref_: np.ndarray = None          # (d, n_x)
    u_ref_: np.ndarray = None          # (d, n_u)
    x_plus_ref_: np.ndarray = None     # (d, n_x)

    # bookkeeping
    n_transitions_: int = None
    nystrom_indices_: np.ndarray = None

    # ----------------------------------------------------------
    #  Public API
    # ----------------------------------------------------------
    def fit(self, X: np.ndarray, U: np.ndarray) -> None:
        """
        Learn cKOR predictor from a single rollout.

        Parameters
        ----------
        X : (N, n_x)   states at times 0 ... N-1
        U : (N, n_u)   inputs at times 0 ... N-1 (aligned with X)
        """
        X = np.asarray(X, dtype=float)
        U = np.asarray(U, dtype=float)

        N, n = X.shape
        assert n == self.state_dim, "X has wrong state dimension"
        assert U.shape[0] == N and U.shape[1] == self.input_dim, "U has wrong input shape"
        if N < 2:
            raise ValueError("Need at least 2 state samples to build transitions")

        X_now = X[:-1]
        U_now = U[:-1]
        X_plus = X[1:]
        self._fit_from_transitions(X_now, U_now, X_plus)

    def fit_multi(self, X_list: Sequence[np.ndarray], U_list: Sequence[np.ndarray]) -> None:
        """
        Fit cKOR from multiple independent trajectories.
        Each (X, U) is a bag/rollout. We never create cross-bag transitions.
        """
        assert len(X_list) == len(U_list) and len(X_list) > 0

        X_blocks, U_blocks, Xp_blocks = [], [], []
        for X, U in zip(X_list, U_list):
            X = np.asarray(X, dtype=float)
            U = np.asarray(U, dtype=float)

            assert X.ndim == 2 and U.ndim == 2
            assert X.shape[1] == self.state_dim, "X has wrong state dimension"
            assert U.shape[1] == self.input_dim, "U has wrong input dimension"
            assert X.shape[0] == U.shape[0], "X and U lengths must match"

            if len(X) < 2:
                continue

            X_blocks.append(X[:-1])
            U_blocks.append(U[:-1])
            Xp_blocks.append(X[1:])

        if not X_blocks:
            raise ValueError("No valid transitions found in input trajectories")

        X_now = np.vstack(X_blocks)
        U_now = np.vstack(U_blocks)
        X_plus = np.vstack(Xp_blocks)
        self._fit_from_transitions(X_now, U_now, X_plus)

    # ------------------------------------------------------------------
    #  Scoring
    # ------------------------------------------------------------------
    def evaluate(self, X: np.ndarray, U: np.ndarray) -> float:
        """
        Fast one-step RMSE in state space.
        """
        self._check_is_fitted()

        X = np.asarray(X, dtype=float)
        U = np.asarray(U, dtype=float)
        N = X.shape[0]
        assert X.shape[1] == self.state_dim
        assert U.shape == (N, self.input_dim)
        if N < 2:
            return float("nan")

        Z = self._lift(X[:-1], U[:-1])
        X_hat = self._lift_inverse(Z)
        return float(np.sqrt(np.mean((X[1:] - X_hat) ** 2)))

    def multistep_rmse(self, X: np.ndarray, U: np.ndarray, H: int = 10) -> float:
        """
        Strict H-step endpoint RMSE.

        For each start index k, simulate H steps from (X[k], U[k:k+H])
        and compare only the endpoint to X[k+H].
        """
        self._check_is_fitted()

        X = np.asarray(X, dtype=float)
        U = np.asarray(U, dtype=float)
        assert X.ndim == 2 and U.ndim == 2
        assert X.shape[1] == self.state_dim
        assert U.shape[1] == self.input_dim

        T = len(X)
        n_start = T - H
        if n_start <= 0:
            return float("nan")

        se_total = 0.0
        for k in range(n_start):
            x_end = self._simulate_endpoint(X[k], U[k:k + H])
            err = x_end - X[k + H]
            se_total += float(np.dot(err, err))

        return float(np.sqrt(se_total / (n_start * self.state_dim)))

    def simulate(self, x0: np.ndarray, U_seq: np.ndarray) -> np.ndarray:
        """
        Roll forward open-loop starting at x0 under prescribed inputs.

        Returns an array of predicted states (T+1, n_x).
        """
        self._check_is_fitted()

        x0 = np.asarray(x0, dtype=float)
        U_seq = np.asarray(U_seq, dtype=float)
        assert x0.shape == (self.state_dim,)
        if U_seq.ndim != 2 or U_seq.shape[1] != self.input_dim:
            raise ValueError("U_seq must have shape (T, input_dim)")

        T = len(U_seq)
        X_pred = np.zeros((T + 1, self.state_dim), dtype=float)
        X_pred[0] = x0

        if T == 0:
            return X_pred

        # Initial lifted state z_1 = z(x_0, u_0)
        z = self._lift(x0, U_seq[0])
        X_pred[1] = self._lift_inverse(z)

        # Recursion: z_{k+1} = (A + diag(v_k) A) z_k, with v_k = kU(u_k)
        for t in range(1, T):
            v = self._control_kernel_mat(U_seq[t:t + 1], self.u_ref_).ravel()
            Az = self.A_ @ z
            z = Az + v * Az
            X_pred[t + 1] = self._lift_inverse(z)

        return X_pred

    # ----------------------------------------------------------
    #  Core fitting routines
    # ----------------------------------------------------------
    def _fit_from_transitions(self, X_now: np.ndarray, U_now: np.ndarray, X_plus: np.ndarray) -> None:
        """Dispatch to full cKOR or Ny-cKOR estimator."""
        n = X_now.shape[0]
        if n < 1:
            raise ValueError("No transitions available for fitting")

        self.n_transitions_ = n
        self.nystrom_indices_ = None

        if self.use_nystrom:
            self._fit_nystrom(X_now, U_now, X_plus)
        else:
            self._fit_full(X_now, U_now, X_plus)

    def _fit_full(self, X_now: np.ndarray, U_now: np.ndarray, X_plus: np.ndarray) -> None:
        """Full cKOR (Algorithm 1 in the paper)."""
        n = X_now.shape[0]

        # Kernel blocks
        KX = self._state_kernel_mat(X_now, X_now)          # (n,n)
        KU = self._control_kernel_mat(U_now, U_now)        # (n,n)
        K_plus = self._state_kernel_mat(X_plus, X_now)     # (n,n)

        KZ = KX * (1.0 + KU)                               # (n,n)
        K_reg = KZ + n * self.ridge * np.eye(n)
        K_gamma_inv = pinv(K_reg)

        # Predictor matrices from Algorithm 1
        self.A_ = (K_gamma_inv @ K_plus).T                 # (n,n)
        self.C_ = (K_gamma_inv @ X_plus).T                 # (n_x,n)

        self.x_ref_ = X_now.copy()
        self.u_ref_ = U_now.copy()
        self.x_plus_ref_ = X_plus.copy()
        self.lift_dim_ = n
        self.mode_ = "full"

    def _fit_nystrom(self, X_now: np.ndarray, U_now: np.ndarray, X_plus: np.ndarray) -> None:
        """Sketched Ny-cKOR (Algorithm 3 in the paper)."""
        n = X_now.shape[0]
        m = int(np.clip(self.n_nystrom, 1, n))

        rng = np.random.default_rng(self.random_state)
        idx = np.sort(rng.choice(n, size=m, replace=False))

        ex = X_now[idx]                                     # inducing x
        eu = U_now[idx]                                     # inducing u
        ex_plus = X_plus[idx]                               # inducing x+

        # Kernel blocks from Algorithm 3 notation
        KeX = self._state_kernel_mat(ex, ex)               # (m,m)
        KX_eX = self._state_kernel_mat(X_now, ex)          # (n,m)

        KeU = self._control_kernel_mat(eu, eu)             # (m,m)
        KU_eU = self._control_kernel_mat(U_now, eu)        # (n,m)

        K_plus_eplus = self._state_kernel_mat(X_plus, ex_plus)   # (n,m)
        Ke_plus = self._state_kernel_mat(ex_plus, ex_plus)       # (m,m)

        KeZ = KeX * (1.0 + KeU)                            # (m,m)
        KZ_eZ = KX_eX * (1.0 + KU_eU)                      # (n,m)

        H = KZ_eZ.T @ KZ_eZ + n * self.ridge * KeZ

        # Equivalent of \tilde{K}^{-1}_gamma in Algorithm 3
        Ktilde_inv = (
            pinv(H + self.jitter * np.eye(m))
            @ KZ_eZ.T
            @ K_plus_eplus
            @ pinv(Ke_plus + self.jitter * np.eye(m))
        )                                                  # (m,m)

        # Predictor matrices from Algorithm 3
        self.A_ = (Ktilde_inv @ Ke_plus).T                # (m,m)
        self.C_ = (Ktilde_inv @ ex_plus).T                # (n_x,m)

        self.x_ref_ = ex.copy()
        self.u_ref_ = eu.copy()
        self.x_plus_ref_ = ex_plus.copy()
        self.lift_dim_ = m
        self.mode_ = "nystrom"
        self.nystrom_indices_ = idx

    # ----------------------------------------------------------
    #  Kernel + lifting helpers
    # ----------------------------------------------------------
    def _state_kernel_mat(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return _rbf_mat(X1, X2, self.state_gamma)

    def _control_kernel_mat(self, U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
        if self.control_kernel == "linear":
            return U1 @ U2.T
        if self.control_kernel == "rbf":
            return _rbf_mat(U1, U2, self.control_gamma)
        raise ValueError(f"Unsupported control_kernel='{self.control_kernel}'")

    def _lift(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        z(x,u) = kX(x, X_ref) ⊙ (1 + kU(u, U_ref)).

        Works with single sample (n_x,), (n_u,) or batches (N,n_x), (N,n_u).
        """
        if x.ndim == 1 and u.ndim == 1:
            kx = self._state_kernel_mat(x[None, :], self.x_ref_).ravel()
            ku = self._control_kernel_mat(u[None, :], self.u_ref_).ravel()
            return kx * (1.0 + ku)

        if x.ndim == 2 and u.ndim == 2:
            if x.shape[0] != u.shape[0]:
                raise ValueError("Batch x and u must have the same nr. rows")
            kx = self._state_kernel_mat(x, self.x_ref_)
            ku = self._control_kernel_mat(u, self.u_ref_)
            return kx * (1.0 + ku)

        raise ValueError("x/u must both be 1D or both be 2D")

    def _lift_inverse(self, z: np.ndarray) -> np.ndarray:
        """
        Decode lifted features to state space with y=id observable matrix C.
        """
        if z.ndim == 1:
            return self.C_ @ z
        if z.ndim == 2:
            return z @ self.C_.T
        raise ValueError("z must have ndim 1 or 2")

    # ----------------------------------------------------------
    #  Simulation helper
    # ----------------------------------------------------------
    def _simulate_endpoint(self, x0: np.ndarray, U_seg: np.ndarray) -> np.ndarray:
        """Return only x_{k+H} for strict H-step endpoint scoring."""
        H = len(U_seg)
        if H == 0:
            return np.asarray(x0, dtype=float)

        z = self._lift(np.asarray(x0, dtype=float), U_seg[0])
        x_curr = self._lift_inverse(z)

        for t in range(1, H):
            v = self._control_kernel_mat(U_seg[t:t + 1], self.u_ref_).ravel()
            Az = self.A_ @ z
            z = Az + v * Az
            x_curr = self._lift_inverse(z)

        return x_curr

    def _check_is_fitted(self) -> None:
        if self.A_ is None or self.C_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit() or fit_multi() first.")
