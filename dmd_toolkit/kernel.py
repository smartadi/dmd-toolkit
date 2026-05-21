"""Kernel DMD — DMD in a (reproducing-kernel) feature space.

References
----------
- Williams, Rowley, Kevrekidis — *J. Nonlinear Sci.* 25 (2015), "A
  kernel-based method for data-driven Koopman spectral analysis".
- Panda, Singh, Kutz (2025), arxiv:2505.06806, "Kernel Dynamic Mode
  Decomposition for Sparse Reconstruction of Closable Koopman Operators"
  — regularised pseudo-inverse + rank truncation on the kernel Gram
  matrix to stabilise the spectrum.

Idea
----
Instead of restricting the Koopman operator to a chosen polynomial
dictionary, kernel DMD lifts the snapshots through a (potentially infinite-
dimensional) feature map :math:`\\psi(x)`, fits a linear operator in feature
space, and uses the kernel trick
:math:`k(x, y) = \\langle\\psi(x), \\psi(y)\\rangle` so the lift never has to
be evaluated explicitly. Two Gram matrices

.. math::
    G_{ij} = k(x_i,\\, x_j), \\qquad A_{ij} = k(x_{i+1},\\, x_j)

define a reduced Koopman matrix
:math:`\\hat K = G^{+}\\, A`. Its eigenvalues approximate the Koopman
spectrum; the corresponding eigenfunctions are evaluated on any new point
:math:`x` via :math:`\\phi_j(x) = v_j^* (\\Sigma^{+}\\, Q^*\\, k_x)` with
:math:`k_x = (k(x_1, x), \\ldots, k(x_m, x))^*`.

Forecasting back to the original observable :math:`g(x) = x` uses the
Koopman-mode decomposition
:math:`x(t) \\approx \\sum_j \\xi_j\\, \\phi_j(x_0)\\, \\lambda_j^{t/\\Delta t}`
where :math:`\\xi_j` is the Koopman mode for the identity observable,
recovered by least-squares projection of the training snapshots onto the
eigenfunction matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from . import utils


Kernel = Literal["rbf", "linear", "poly"]


def _kernel(
    X: np.ndarray,
    Y: np.ndarray,
    kind: Kernel,
    gamma: float,
    degree: int,
    coef0: float,
) -> np.ndarray:
    """Compute K[i, j] = k(X[:, i], Y[:, j]). Inputs are (features, n) and
    (features, m); output is (n, m).
    """
    # Use real parts for kernels — RBF/poly with complex inputs is ill-defined
    Xr = np.real(X).astype(float)
    Yr = np.real(Y).astype(float)
    if kind == "linear":
        return Xr.T @ Yr
    if kind == "poly":
        return (gamma * (Xr.T @ Yr) + coef0) ** degree
    if kind == "rbf":
        # Squared Euclidean distance via expansion
        x2 = np.sum(Xr ** 2, axis=0)
        y2 = np.sum(Yr ** 2, axis=0)
        d2 = x2[:, None] + y2[None, :] - 2.0 * (Xr.T @ Yr)
        np.maximum(d2, 0.0, out=d2)
        return np.exp(-gamma * d2)
    raise ValueError(f"Unknown kernel {kind!r}")


@dataclass
class KernelDMD:
    """Kernel DMD (Williams 2015) with Panda/Singh/Kutz-style regularisation.

    Parameters
    ----------
    kernel : {"rbf", "linear", "poly"}
        Kernel to lift the snapshots.
    gamma : float | None
        Kernel bandwidth (RBF) or scale (poly). If None, ``rbf`` uses the
        median-heuristic ``1 / (2 * median_distance^2)`` and ``poly`` uses
        ``1 / n_features``.
    degree : int
        Polynomial degree.
    coef0 : float
        Polynomial bias.
    rank : int | float | None
        Truncation of the Gram-matrix eigen-spectrum. Same semantics as
        :func:`dmd_toolkit.utils.truncated_svd` (int / fraction / None).
    reg : float
        Tikhonov regularisation added to the diagonal of ``G`` before
        pseudo-inversion. Defaults to ``1e-10 * tr(G) / m``.
    dt : float
        Sampling interval.
    """

    kernel: Kernel = "rbf"
    gamma: float | None = None
    degree: int = 3
    coef0: float = 1.0
    rank: int | float | None = None
    reg: float | None = None
    dt: float = 1.0

    eigenvalues: np.ndarray = field(init=False, repr=False)
    omega: np.ndarray = field(init=False, repr=False)
    modes: np.ndarray = field(init=False, repr=False)
    amplitudes: np.ndarray = field(init=False, repr=False)
    eigenfunctions_train: np.ndarray = field(init=False, repr=False)
    _X1: np.ndarray = field(init=False, repr=False)
    _Q: np.ndarray = field(init=False, repr=False)
    _sigma_inv: np.ndarray = field(init=False, repr=False)
    _V: np.ndarray = field(init=False, repr=False)
    _x0: np.ndarray = field(init=False, repr=False)

    def fit(self, data: np.ndarray) -> "KernelDMD":
        X = utils.to_snapshots(data)
        X1, X2 = utils.split_snapshots(X)
        self._X1 = X1
        self._x0 = X1[:, 0]
        gamma = self._auto_gamma(X1)

        G = _kernel(X1, X1, self.kernel, gamma, self.degree, self.coef0)
        A = _kernel(X2, X1, self.kernel, gamma, self.degree, self.coef0)

        m = G.shape[0]
        reg = self.reg if self.reg is not None else 1e-10 * (np.trace(G) / m + 1.0)
        G = G + reg * np.eye(m)

        # Eigendecompose Hermitian PSD G = Q Σ^2 Q^T (use eigh — G is symmetric)
        eigG, Q = np.linalg.eigh(G)
        order = np.argsort(eigG)[::-1]
        eigG, Q = eigG[order], Q[:, order]
        # Truncate
        if self.rank is None:
            r = len(eigG)
        elif isinstance(self.rank, float) and 0 < self.rank < 1:
            cum = np.cumsum(eigG) / np.sum(eigG)
            r = int(np.searchsorted(cum, self.rank) + 1)
        else:
            r = min(int(self.rank), len(eigG))
        # Guard against negative numerical eigenvalues
        eigG = np.maximum(eigG[:r], 1e-14)
        Q = Q[:, :r]
        sigma = np.sqrt(eigG)
        sigma_inv = 1.0 / sigma

        # Reduced Koopman matrix: K_hat = Σ^{-1} Q^T A Q Σ^{-1}
        # (working with A as the cross-Gram; here Q^T A Q is r×r)
        K_hat = (sigma_inv[:, None] * (Q.T @ A @ Q)) * sigma_inv[None, :]

        eigvals, V = np.linalg.eig(K_hat)

        # Eigenfunctions evaluated at training points
        # φ(x_i) = V^T Σ^{-1} Q^T k(X1, x_i). For x_i ∈ training set the kernel
        # vector is the i-th column of G, so φ_train = V^T Σ^{-1} Q^T G = V^T Σ Q^T
        phi_train = (V.T @ (sigma[:, None] * Q.T)).T  # (m, r) columns are φ_j(X1)

        # Koopman modes for identity observable g(x) = x:
        # X1 ≈ Ξ φ_train^T  →  Ξ = X1 @ pinv(φ_train.T)
        modes, *_ = np.linalg.lstsq(phi_train.T @ phi_train, phi_train.T @ X1.T, rcond=None)
        modes = modes.T  # (n_features, r)

        # Amplitudes: φ_j(x_0). For x_0 = X1[:, 0] this is phi_train[0]
        amplitudes = phi_train[0]

        self._Q = Q
        self._sigma_inv = sigma_inv
        self._V = V
        self.eigenvalues = eigvals
        self.omega = np.log(eigvals.astype(complex)) / self.dt
        self.modes = modes
        self.amplitudes = amplitudes
        self.eigenfunctions_train = phi_train
        self._gamma = gamma
        return self

    # ------------------------------------------------------------------
    def _auto_gamma(self, X: np.ndarray) -> float:
        if self.gamma is not None:
            return float(self.gamma)
        if self.kernel == "rbf":
            Xr = np.real(X).astype(float)
            # Median pairwise squared distance, subsample for speed
            m = Xr.shape[1]
            if m > 200:
                idx = np.random.default_rng(0).choice(m, size=200, replace=False)
                Xs = Xr[:, idx]
            else:
                Xs = Xr
            x2 = np.sum(Xs ** 2, axis=0)
            d2 = x2[:, None] + x2[None, :] - 2.0 * (Xs.T @ Xs)
            d2 = d2[np.triu_indices_from(d2, k=1)]
            d2 = d2[d2 > 0]
            med = float(np.median(d2)) if d2.size else 1.0
            return 1.0 / (2.0 * max(med, 1e-12))
        return 1.0 / X.shape[0]

    # ------------------------------------------------------------------
    def eigenfunctions(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the Koopman eigenfunctions at new points.

        Parameters
        ----------
        X : (n_features, n_points) array

        Returns
        -------
        phi : (n_points, rank) array — φ_j(x_i).
        """
        X = utils.to_snapshots(X)
        k = _kernel(self._X1, X, self.kernel, self._gamma, self.degree, self.coef0)
        # φ(x) = V^T Σ^{-1} Q^T k(X1, x)
        return (self._V.T @ (self._sigma_inv[:, None] * (self._Q.T @ k))).T

    def reconstruct(self, t: np.ndarray) -> np.ndarray:
        return utils.reconstruct(
            self.modes, self.eigenvalues, self.amplitudes, t, dt=self.dt
        )

    def predict(self, n_steps: int) -> np.ndarray:
        t = np.arange(1, n_steps + 1) * self.dt
        return utils.reconstruct(
            self.modes, self.eigenvalues, self.amplitudes, t, dt=self.dt
        )

    @property
    def frequencies(self) -> np.ndarray:
        return np.imag(self.omega) / (2 * np.pi)

    @property
    def growth_rates(self) -> np.ndarray:
        return np.real(self.omega)


def kernel_dmd(
    data: np.ndarray,
    kernel: Kernel = "rbf",
    rank: int | float | None = None,
    dt: float = 1.0,
    **kwargs,
) -> KernelDMD:
    """Convenience: fit and return a KernelDMD instance."""
    return KernelDMD(kernel=kernel, rank=rank, dt=dt, **kwargs).fit(data)
