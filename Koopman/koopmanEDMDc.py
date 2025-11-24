"""
EDMDc Koopman identification with Radial Basis Functions (RBFs).

Main class: KoopmanEDMDc
--------------------------------------------------------------
Methods
    ├─ fit(X, U)                learn A, B from trajectory data
    ├─ evaluate(X, U)           one-step RMSE over a test set
    ├─ multistep_rmse(X, U, H)  multi-step RMSE after H steps
    └─ simulate(x0, U_seq)      multi-step prediction (open-loop)

Key hyper-parameters
    - n_rbfs        - number of RBF dictionary elements (>= 1)
    - gamma         - RBF width (1/(2*sigma^2)); smaller -> wider bumps
--------------------------------------------------------------

Dependencies: numpy, scipy, dataclasses, typing, scikit-learn

Author: Victor Nan Fernandez-Ayala
Date: 2025
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
from numpy.linalg import pinv
from dataclasses import dataclass
from sklearn.cluster import KMeans
from typing import Sequence


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
def _rbf(x: np.ndarray, c: np.ndarray, gamma: float) -> float:
    """Gaussian RBF ||x - c||^2 -> R."""
    return np.exp(-gamma * np.linalg.norm(x - c) ** 2)

def _rbf_mat(X: np.ndarray, C: np.ndarray, gamma: float) -> np.ndarray:
    """
    Fast pair-wise RBF between a batch of points X (N,n)
    and centres C (k,n) -> (N,k) matrix.
    """
    x2 = np.sum(X**2, axis=1)[:, None]              # (N,1)
    c2 = np.sum(C**2, axis=1)[None, :]              # (1,k)
    return np.exp(-gamma * (x2 + c2 - 2*X @ C.T))   # (N,k)


# -----------------------------------------------------------------------------#
#  Main class
# -----------------------------------------------------------------------------#
@dataclass
class KoopmanEDMDc:
    state_dim: int                      # (n)
    input_dim: int                      # (r)
    n_rbfs: int = 200
    gamma: float = 1.0
    # regularisation for pseudo-inverse (ridge) – prevents blow-up
    ridge: float = 1e-8

    # learned parameters (set by fit)
    centers_: np.ndarray = None         # (n_rbfs, n)
    A_: np.ndarray = None               # (d, d)
    B_: np.ndarray = None               # (d, r)
    lift_dim_: int = None               # (d)

    # ----------------------------------------------------------
    #  Public API
    # ----------------------------------------------------------
    def fit(self, X: np.ndarray, U: np.ndarray) -> None:
        """
        Learn (A,B) from a single rollout or a batch of transitions.

        Parameters
        ----------
        X : (N, n)   state at times 0 ... N-1
        U : (N, r)   input at times 0 ... N-1   (aligned with X)
        """
        N, n = X.shape
        assert U.shape[0] == N and U.shape[1] == self.input_dim

        # 1) Pick RBF centres via k-means on the state cloud
        kmeans = KMeans(n_clusters=self.n_rbfs, n_init="auto", random_state=0).fit(X)
        self.centers_ = kmeans.cluster_centers_

        # 2) Build lifted snapshots  Z_t = phi(x_t)
        Z = self._lift(X[:-1])          # (N-1, d)
        Z_plus = self._lift(X[1:])      # (N-1, d)
        U_cut = U[:-1]                  # (N-1, r)

        # 3) Solve Z+ = A Z + B U via least squares
        G = np.hstack([Z, U_cut])       # (N-1, d+r)
        Y = Z_plus                      # (N-1, d)
        # Ridge-regularised normal-equation
        M = pinv(G.T @ G + self.ridge * np.eye(G.shape[1])) @ G.T @ Y
        M = M.T                         # (d, d+r)
        d = Z.shape[1]
        self.A_ = M[:, :d]
        self.B_ = M[:, d:]

        self.lift_dim_ = d

        # # 4) Learn a decoder to reconstruct x
        # Z_full = np.stack([self._lift(x) for x in X])
        # W = np.linalg.solve(
        #         Z_full.T @ Z_full + self.ridge * np.eye(Z_full.shape[1]),
        #         Z_full.T @ X
        #     )                                                   # (d, n)
        # self.decoder_ = W.T                                     # (n, d)

    def fit_multi(self, X_list, U_list):
        """
        Fit EDMDc from multiple independent trajectories.
        Each (X, U) is a bag/rollout. We never create cross-bag transitions.
        """
        assert len(X_list) == len(U_list) and len(X_list) > 0
        n = self.state_dim
        r = self.input_dim
        for X, U in zip(X_list, U_list):
            assert X.shape[1] == n and U.shape[1] == r

        # 1) Choose RBF centers from ALL training states
        X_all = np.vstack([X for X in X_list if len(X) > 0])
        kmeans = KMeans(n_clusters=self.n_rbfs, n_init="auto", random_state=0).fit(X_all)
        self.centers_ = kmeans.cluster_centers_

        # 2) Build stacked snapshots WITHOUT crossing boundaries
        Z_blocks, Zp_blocks, U_blocks = [], [], []
        for X, U in zip(X_list, U_list):
            if len(X) < 2:
                continue
            Z = self._lift(X[:-1])      # (T-1, d)
            Zp = self._lift(X[1:])      # (T-1, d)
            Z_blocks.append(Z)
            Zp_blocks.append(Zp)
            U_blocks.append(U[:-1])

        Z = np.vstack(Z_blocks)         # (N, d)
        Zp = np.vstack(Zp_blocks)       # (N, d)
        Uc = np.vstack(U_blocks)        # (N, r)

        # 3) Solve Z+ = A Z + B U (ridge-regularised normal equation)
        G = np.hstack([Z, Uc])          # (N, d+r)
        Y = Zp                          # (N, d)
        M = np.linalg.pinv(G.T @ G + self.ridge*np.eye(G.shape[1])) @ (G.T @ Y)
        M = M.T                         # (d, d+r)
        d = Z.shape[1]
        self.A_ = M[:, :d]
        self.B_ = M[:, d:]
        self.lift_dim_ = d

    # ------------------------------------------------------------------
    #  Scoring (fully vectorised)
    # ------------------------------------------------------------------
    def evaluate(self, X: np.ndarray, U: np.ndarray) -> float:
        """
        Fast one-step RMSE
        Return root-mean-square one-step prediction error (RMSE) in state space.
        Cost ~ one BLAS GEMM instead of O(N) Python loops.
        """
        # 1. Lift all states except the last one in a single shot
        Z = self._lift(X[:-1])                      # (N-1, d)
        # 2. One-step prediction in feature space Z+ ~ A Z + B U
        Z_hat = Z @ self.A_.T + U[:-1] @ self.B_.T  # (N-1, d)
        # 3. Decode -> state space
        X_hat = self._lift_inverse(Z_hat)           # (N-1, n)
        # 4. RMSE
        return np.sqrt(np.mean((X[1:] - X_hat) ** 2))

    def multistep_rmse(self, X: np.ndarray, U: np.ndarray, H: int = 10) -> float:
        """
        Vectorised H-step RMSE.
        Root-mean-square error after propagating the model H steps without re-initialising.
        Propagates all starting positions in parallel.

        Parameters:
        ----------
        X: (N, n)   full state roll-out
        U: (N, r)   aligned inputs
        H: horizon  (default 10 steps)

        Returns:
        -------
        scalar  RMSE over all k = 0 ... N-H-1
        """
        N = len(X)
        n_start = N - H                                                 # Nr. starting indices k
        # 1. Lift X_k  for all k = 0…N-H-1
        Z_batch = self._lift(X[:n_start])                               # (n_start, d)
        # 2. Slice the required input segments once to avoid inside-loop indexing
        U_seg = np.stack([U[k:k+H] for k in range(n_start)], axis=0)    # (n_start, H, r)
        # 3. Propagate H steps  (vectorised over the n_start roll-outs)
        A_T, B_T = self.A_.T, self.B_.T                                 # local views -> +10 % speed
        for t in range(H):
            Z_batch = Z_batch @ A_T + U_seg[:, t] @ B_T                 # (n_start, d)
        # 4. Decode and score
        X_hat = self._lift_inverse(Z_batch)                             # (n_start, n)
        return np.sqrt(np.mean((X[H:] - X_hat) ** 2))

    def simulate(self, x0: np.ndarray, U_seq: np.ndarray) -> np.ndarray:
        """
        Roll forward open-loop starting at x0 under prescribed inputs.

        Returns an array of predicted states (T+1, n).
        """
        T = len(U_seq)
        X_pred = np.zeros((T + 1, self.state_dim))
        X_pred[0] = x0.copy()
        z = self._lift(x0)
        for t, u in enumerate(U_seq):
            z = self.A_ @ z + self.B_ @ u
            x = self._lift_inverse(z)
            X_pred[t + 1] = x
        return X_pred
        
    # ----------------------------------------------------------
    #  Private helpers  (now batch-aware)
    # ----------------------------------------------------------
    def _lift(self, x: np.ndarray) -> np.ndarray:
        """
        phi(x) = [x,  RBF_1(x), ..., RBF_k(x)] in R^{n+k}.

        Works with a single sample (n,) or a batch (N,n).
        Returns (d,) or (N,d) with d = n + k.
        """
        if x.ndim == 1:                 # single state
            rbf = _rbf_mat(x[None, :], self.centers_, self.gamma).ravel()   # (k,)
            return np.hstack([x, rbf])

        if x.ndim == 2:                 # batch of states
            rbf = _rbf_mat(x, self.centers_, self.gamma)                    # (N,k)
            return np.hstack([x, rbf])

        raise ValueError("x must have ndim 1 or 2")

    def _lift_inverse(self, z: np.ndarray) -> np.ndarray:
        """
        Batch-aware inverse phi^-1(z) -> state.
        If we have a decoder network, use it to reconstruct the state.
        Otherwise, use a simple inverse that copies back original state coordinates
        """
        if hasattr(self, "decoder_"):
            return z @ self.decoder_.T  # works for both (d,) and (N,d)

        # Fallback: take first n coordinates
        return z[..., :self.state_dim]  # works for (d,) or (N,d)