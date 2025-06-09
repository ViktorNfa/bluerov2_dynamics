"""
Control-Coherent Koopman identification with Radial Basis Functions (RBFs).

Main class: KoopmanCCK
--------------------------------------------------------------
Methods
    ├─ fit(X, U)              learn A, B from trajectory data
    ├─ evaluate(X, U)         one-step RMSE over a test set
    ├─ simulate(x0, U_seq)    multi-step prediction (open-loop)
    └─ save / load            (optional) persistence helpers

Key hyper-parameters
    • n_rbfs        - number of RBF dictionary elements (≥ 1)
    • gamma         - RBF width (1/(2*sigma²)); smaller ⇒ wider bumps
    • pt_indices    - indices of “actuator-driven” state coords p
                        (rows allowed to receive direct control via B)
                        For BlueROV2 a good default is the 6 body-rate states
                        [6,7,8,9,10,11]  (u,v,w,p,q,r).
--------------------------------------------------------------

Dependencies : numpy, scipy, scikit-learn

Author: Victor Nan Fernandez-Ayala
Date:   2025
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
from numpy.linalg import pinv
from dataclasses import dataclass
from sklearn.cluster import KMeans
from typing import Sequence, Tuple

# import sys
# np.set_printoptions(threshold=sys.maxsize)


def _rbf(x: np.ndarray, c: np.ndarray, gamma: float) -> float:
    """Gaussian RBF ‖x - c‖² → R."""
    return np.exp(-gamma * np.linalg.norm(x - c) ** 2)


@dataclass
class KoopmanCCK:
    state_dim: int                      # (n)
    input_dim: int                      # (r)
    n_rbfs: int = 200
    gamma: float = 1.0
    pt_indices: Sequence[int] = tuple()
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
        X : (N, n)   state at times 0 … N-1
        U : (N, r)   input at times 0 … N-1   (aligned with X)
        """
        N, n = X.shape
        assert U.shape[0] == N and U.shape[1] == self.input_dim

        # 1) pick RBF centres via k-means on the state cloud
        kmeans = KMeans(n_clusters=self.n_rbfs, random_state=0).fit(X)
        self.centers_ = kmeans.cluster_centers_

        # 2) build lifted snapshots  Z_t = phi(x_t)
        Z = np.stack([self._lift(x) for x in X[:-1]])           # (N-1, d)
        Z_plus = np.stack([self._lift(x) for x in X[1:]])       # (N-1, d)
        U_cut = U[:-1]                                          # (N-1, r)

        # 3) solve   Z⁺ = A Z + B U    via least squares
        G = np.hstack([Z, U_cut])                               # (N-1, d+r)
        Y = Z_plus                                              # (N-1, d)
        # ridge-regularised normal-equation
        M = pinv(G.T @ G + self.ridge * np.eye(G.shape[1])) @ G.T @ Y
        M = M.T                                                 # (d, d+r)
        d = Z.shape[1]
        self.A_ = M[:, :d]
        B_hat = M[:, d:]                                        # unconstrained

        # 4) impose CCK structure: only p-rows are allowed in B
        self.B_ = np.zeros_like(B_hat)
        p_rows = np.array(self.pt_indices, dtype=int)
        self.B_[p_rows, :] = B_hat[p_rows, :]

        # print(self.A_)
        # print(self.B_)

        self.lift_dim_ = d

    def evaluate(self, X: np.ndarray, U: np.ndarray) -> float:
        """
        Return root-mean-square one-step prediction error (RMSE) in state space.
        """
        preds = []
        for x, u in zip(X[:-1], U[:-1]):
            z = self._lift(x)
            z_next_pred = self.A_ @ z + self.B_ @ u
            x_pred = self._lift_inverse(z_next_pred)
            preds.append(x_pred)
        preds = np.asarray(preds)
        rmse = np.sqrt(np.mean((X[1:] - preds) ** 2))
        return rmse

    def simulate(self, x0: np.ndarray, U_seq: np.ndarray) -> np.ndarray:
        """
        Roll forward open-loop starting at x0 under prescribed inputs.

        Returns an array of predicted states  (T+1, n).
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
    #  Private helpers
    # ----------------------------------------------------------
    def _lift(self, x: np.ndarray) -> np.ndarray:
        """phi(x) = [p(x);  RBF_1(x); … ; RBF_k(x)] ∈ ℝ^{m+k}."""
        p = x[self.pt_indices] if self.pt_indices else np.empty(0)
        rbf_vals = np.array([_rbf(x, c, self.gamma) for c in self.centers_])
        return np.hstack([p, rbf_vals])

    def _lift_inverse(self, z: np.ndarray) -> np.ndarray:
        """
        Simple inverse that copies back original state coordinates and
        *ignores* the RBF part (coarse but adequate for evaluation).

        Better reconstructions could learn a decoder network.
        """
        x_rec = np.zeros(self.state_dim)
        if self.pt_indices:
            x_rec[self.pt_indices] = z[:len(self.pt_indices)]
        # the remaining coordinates are unknown – leave zeros
        return x_rec