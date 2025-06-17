"""
Kernel-Koopman identification for control-affine (bilinear) models.
This follows the non-parametric “cKOR” framework of Bevanda et al.
(IEEE TAC, 2025) but keeps the implementation deliberately simple:

    - phi(x)    : RBF (or custom) kernel features wrt. m inducing points
    - psi(x,u)  : [phi(x) ; u_1 phi(x) ; ... ; u_r phi(x)]  --- bilinear lift
    - z+ = A z  +  sum_i B_i z · u_i                        --- learned dynamics

The class exposes the same public API as `KoopmanEDMDc`:
    fit / evaluate / multistep_rmse / simulate
so the training script needs only the import line changed.
--------------------------------------------------------------

Dependencies : numpy, scipy, dataclasses, typing, scikit-learn

Author: Victor Nan Fernandez-Ayala
Date:   2025
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import numpy as np
from dataclasses import dataclass
from typing import Callable, Sequence
from numpy.linalg import pinv
from sklearn.cluster import KMeans


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
Array = np.ndarray
KernelFn = Callable[[Array, Array], float]

def _rbf(x: Array, c: Array, gamma: float) -> float:
    """Gaussian RBF ||x - c||^2 -> R."""
    return np.exp(-gamma * np.linalg.norm(x - c) ** 2)


# -----------------------------------------------------------------------------#
#  Main class
# -----------------------------------------------------------------------------#
@dataclass
class KoopmanKernel:
    state_dim: int
    input_dim: int
    n_inducing: int = 300
    gamma: float = 1.0                  # width for RBF
    ridge: float = 1e-9                 # l2-regulariser (lambda in paper)
    kernel: KernelFn = None             # if None -> use RBF(gamma)

    # learned params
    centers_: Array = None              # (m,n)
    A_: Array = None                    # (m,m)
    B_: Sequence[Array] = None          # list of r matrices (m,m)

    # ------------------------------------------------------------------#
    # Public api
    # ------------------------------------------------------------------#
    def fit(self, X: Array, U: Array) -> None:
        """
        Learn bilinear operator  z+ = A z + sum_i B_i z · u_i
        from one long trajectory  {(x_k,u_k)}_{k=0}^{N-1}.
        """
        N, n = X.shape
        assert n == self.state_dim and U.shape == (N, self.input_dim)

        # 1) Pick inducing points (RBF centres)
        kmeans = KMeans(n_clusters=self.n_inducing, random_state=0).fit(X)
        self.centers_ = kmeans.cluster_centers_
        m = self.n_inducing

        # Choose kernel
        if self.kernel is None:
            self.kernel = lambda xa, xb: _rbf(xa, xb, self.gamma)

        # Helper: lift state to feature column (m,)
        def lift(x):
            return np.fromiter(
                (self.kernel(x, c) for c in self.centers_), dtype=float, count=m
            )

        # 2) Build lifted snapshot matrices
        phi = np.zeros((N - 1, m))
        psi = np.zeros((N - 1, m * (1 + self.input_dim)))
        for k in range(N - 1):
            z = lift(X[k])
            phi[k] = z                      # z_k
            block = [z] + [U[k, i] * z for i in range(self.input_dim)]
            psi[k] = np.hstack(block)       # psi(x_k,u_k)

        # Target = z_{k+1}
        phi_plus = np.vstack([lift(x) for x in X[1:]])

        # 3) Solve ridge-regularised least squares  phi+ ~ M psi^T
        G = psi                             # (N-1, d)
        Y = phi_plus                        # (N-1, m)
        M = pinv(G.T @ G + self.ridge * np.eye(G.shape[1])) @ G.T @ Y
        M = M.T                             # (m, d)

        # 4) Split M -> A , {B_i}
        d_per_block = m
        self.A_ = M[:, :d_per_block]
        self.B_ = [
            M[:, d_per_block * (i + 1) : d_per_block * (i + 2)]
            for i in range(self.input_dim)
        ]

        # Cache decoder (identity – first m components reproduce kernel coeffs)
        self._lift = lift                   # store for simulate/evaluate

    # ------------------------------------------------------------------#
    def _bilinear_step(self, z, u):
        """z -> A z + sum B_i z u_i"""
        z_next = self.A_ @ z
        for i in range(self.input_dim):
            z_next += self.B_[i] @ z * u[i]
        return z_next

    # ------------------------------------------------------------------#
    def evaluate(self, X: Array, U: Array) -> float:
        preds = []
        for x, u in zip(X[:-1], U[:-1]):
            z = self._lift(x)
            z_next = self._bilinear_step(z, u)
            x_pred = self._lift_inverse(z_next)
            preds.append(x_pred)
        preds = np.asarray(preds)
        return np.sqrt(np.mean((X[1:] - preds) ** 2))

    def multistep_rmse(self, X: Array, U: Array, H: int = 10) -> float:
        errs = []
        N = len(X)
        for k in range(N - H):
            z = self._lift(X[k])
            for t in range(H):
                z = self._bilinear_step(z, U[k + t])
            errs.append(X[k + H] - self._lift_inverse(z))
        return np.sqrt(np.mean(np.square(errs)))

    def simulate(self, x0: Array, U_seq: Array) -> Array:
        T = len(U_seq)
        X_pred = np.zeros((T + 1, self.state_dim))
        X_pred[0] = x0
        z = self._lift(x0)
        for t, u in enumerate(U_seq):
            z = self._bilinear_step(z, u)
            X_pred[t + 1] = self._lift_inverse(z)
        return X_pred

    # ----------------------------------------------------------
    #  Private helpers
    # ----------------------------------------------------------
    def _lift_inverse(self, z: Array) -> Array:
        """
        Reconstruct state from kernel feature vector by radial basis
        interpolation (weights = z, centres = self.centers_).
        For RBF kernels with gamma > 0 this is the native-space interpolant.
        """
        # simple Shepard-type reconstruction (weighted centroid)
        w = z / (z.sum() + 1e-12)
        return w @ self.centers_