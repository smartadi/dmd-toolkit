"""Physics-informed DMD (piDMD).

Reference: Baddoo, Herrmann, McKeon, Kutz, Brunton — *Proc. R. Soc. A* 479
(2023). "Physics-informed dynamic mode decomposition (piDMD)."

Standard DMD finds the best-fit linear operator A with no constraints. piDMD
solves a *constrained* least-squares problem
    min_{A in C}  || X2 - A X1 ||_F
where the constraint set C encodes a known physical structure (energy
preservation, self-adjointness, locality, translation invariance, ...).
The right constraint regularises in low-data / noisy regimes and yields
physically meaningful spectra.

Constraints implemented here
----------------------------
- ``"unitary"``       — A^* A = I (energy-preserving / Hamiltonian flow).
                         Closed form via the orthogonal Procrustes problem.
- ``"symmetric"``     — A = A^* (self-adjoint, real-spectrum systems).
                         Solved via a Lyapunov equation.
- ``"skew_symmetric"``— A = -A^* (purely-imaginary spectrum, conservative).
                         Solved via a Sylvester equation.
- ``"diagonal"``      — A is diagonal (channel-wise decoupled dynamics).
- ``"circulant"``     — A is circulant (1D translation-invariant systems);
                         fitted diagonally in the DFT basis.

The "non-local" constraints (unitary / symmetric / skew_symmetric) act on
the SVD-reduced operator and return DMD-style ``modes`` lifted back to
full space. The "local" constraints (diagonal / circulant) act directly on
the full feature space and return identity-like modes (one mode per feature
or per DFT bin).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.linalg import solve_continuous_lyapunov, solve_sylvester

from . import utils


CONSTRAINTS = ("unitary", "symmetric", "skew_symmetric", "diagonal", "circulant")
Constraint = Literal["unitary", "symmetric", "skew_symmetric", "diagonal", "circulant"]


@dataclass
class piDMD:
    """Physics-informed DMD.

    Parameters
    ----------
    constraint : str
        One of :data:`CONSTRAINTS`.
    rank : int | float | None
        SVD truncation for the non-local constraints. Ignored by the local
        ones (diagonal, circulant) which act on the full state.
    dt : float
        Sampling interval, used to convert discrete eigenvalues
        :math:`\\lambda` to continuous-time :math:`\\omega = \\log\\lambda / dt`.
    """

    constraint: Constraint = "unitary"
    rank: int | float | None = None
    dt: float = 1.0

    A: np.ndarray = field(init=False, repr=False)
    modes: np.ndarray = field(init=False, repr=False)
    eigenvalues: np.ndarray = field(init=False, repr=False)
    omega: np.ndarray = field(init=False, repr=False)
    amplitudes: np.ndarray = field(init=False, repr=False)
    _x0: np.ndarray = field(init=False, repr=False)

    def fit(self, data: np.ndarray) -> "piDMD":
        if self.constraint not in CONSTRAINTS:
            raise ValueError(
                f"Unknown constraint {self.constraint!r}; pick one of {CONSTRAINTS}"
            )
        X = utils.to_snapshots(data).astype(complex)
        X1, X2 = utils.split_snapshots(X)
        self._x0 = X1[:, 0]

        if self.constraint == "unitary":
            self._fit_unitary(X1, X2)
        elif self.constraint == "symmetric":
            self._fit_symmetric(X1, X2)
        elif self.constraint == "skew_symmetric":
            self._fit_skew_symmetric(X1, X2)
        elif self.constraint == "diagonal":
            self._fit_diagonal(X1, X2)
        elif self.constraint == "circulant":
            self._fit_circulant(X1, X2)

        self.omega = np.log(self.eigenvalues.astype(complex)) / self.dt
        return self

    # ------------------------------------------------------------------
    # Non-local constraints (SVD-reduced)
    # ------------------------------------------------------------------
    def _project(self, X1: np.ndarray, X2: np.ndarray):
        U, s, Vh = utils.truncated_svd(X1, rank=self.rank)
        Y1 = U.conj().T @ X1
        Y2 = U.conj().T @ X2
        return U, Y1, Y2

    def _fit_unitary(self, X1: np.ndarray, X2: np.ndarray) -> None:
        """Orthogonal Procrustes: min ||Y2 - Q Y1|| s.t. Q^*Q = I."""
        U, Y1, Y2 = self._project(X1, X2)
        Up, _, Vph = np.linalg.svd(Y2 @ Y1.conj().T, full_matrices=False)
        Q = Up @ Vph
        eigvals, W = np.linalg.eig(Q)
        modes = U @ W
        self.A = Q
        self.eigenvalues = eigvals
        self.modes = modes
        self.amplitudes = self._amplitudes(modes, self._x0)

    def _fit_symmetric(self, X1: np.ndarray, X2: np.ndarray) -> None:
        """A = A^* via Lyapunov-style condition

            A G + G A = M + M^*,    G = Y1 Y1^*,  M = Y2 Y1^*.
        """
        U, Y1, Y2 = self._project(X1, X2)
        G = Y1 @ Y1.conj().T
        M = Y2 @ Y1.conj().T
        rhs = M + M.conj().T
        # solve_continuous_lyapunov solves G X + X G^* = -rhs; G is Hermitian
        # so G^* = G and we get G A + A G = -rhs  →  pass -rhs to get +rhs.
        A = solve_continuous_lyapunov(G, -rhs)
        A = (A + A.conj().T) / 2  # numerical Hermitian projection
        eigvals, W = np.linalg.eigh(A)
        modes = U @ W
        self.A = A
        self.eigenvalues = eigvals.astype(complex)
        self.modes = modes
        self.amplitudes = self._amplitudes(modes, self._x0)

    def _fit_skew_symmetric(self, X1: np.ndarray, X2: np.ndarray) -> None:
        """A = -A^* via Sylvester equation

            A G - G A = M - M^*,    G = Y1 Y1^*,  M = Y2 Y1^*.
        """
        U, Y1, Y2 = self._project(X1, X2)
        G = Y1 @ Y1.conj().T
        M = Y2 @ Y1.conj().T
        rhs = M - M.conj().T
        # solve_sylvester(A_, B_, C) solves A_ X + X B_ = C → use (G, -G, -rhs)
        A = solve_sylvester(-G, G, -rhs)
        A = (A - A.conj().T) / 2  # numerical skew projection
        eigvals, W = np.linalg.eig(A)
        modes = U @ W
        self.A = A
        self.eigenvalues = eigvals
        self.modes = modes
        self.amplitudes = self._amplitudes(modes, self._x0)

    # ------------------------------------------------------------------
    # Local constraints (full-space)
    # ------------------------------------------------------------------
    def _fit_diagonal(self, X1: np.ndarray, X2: np.ndarray) -> None:
        """Per-feature scalar fit: a_i = <x2_i, x1_i> / <x1_i, x1_i>."""
        num = np.sum(X2 * X1.conj(), axis=1)
        den = np.sum(np.abs(X1) ** 2, axis=1)
        a = num / np.where(den == 0, 1.0, den)
        self.A = np.diag(a)
        self.eigenvalues = a
        self.modes = np.eye(X1.shape[0], dtype=complex)
        self.amplitudes = self._x0.astype(complex)

    def _fit_circulant(self, X1: np.ndarray, X2: np.ndarray) -> None:
        """Circulant: diagonal in the DFT basis. Fit per Fourier bin."""
        F1 = np.fft.fft(X1, axis=0)
        F2 = np.fft.fft(X2, axis=0)
        num = np.sum(F2 * F1.conj(), axis=1)
        den = np.sum(np.abs(F1) ** 2, axis=1)
        a_hat = num / np.where(den == 0, 1.0, den)
        # Reconstruct circulant A = F^{-1} diag(a_hat) F
        n = X1.shape[0]
        F = np.fft.fft(np.eye(n), axis=0)
        Finv = F.conj().T / n
        self.A = Finv @ np.diag(a_hat) @ F
        self.eigenvalues = a_hat
        # Modes are the DFT basis columns
        self.modes = Finv
        # Amplitudes: project x0 onto DFT basis (i.e. forward FFT of x0)
        self.amplitudes = np.fft.fft(self._x0)

    # ------------------------------------------------------------------
    @staticmethod
    def _amplitudes(modes: np.ndarray, x0: np.ndarray) -> np.ndarray:
        b, *_ = np.linalg.lstsq(modes, x0, rcond=None)
        return b

    # ------------------------------------------------------------------
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


def pidmd(
    data: np.ndarray,
    constraint: Constraint = "unitary",
    rank: int | float | None = None,
    dt: float = 1.0,
) -> piDMD:
    """Convenience: fit and return a piDMD instance."""
    return piDMD(constraint=constraint, rank=rank, dt=dt).fit(data)
