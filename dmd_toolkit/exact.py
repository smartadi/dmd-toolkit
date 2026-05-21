"""Exact DMD (Tu, Rowley, Luchtenburg, Brunton, Kutz 2014)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import utils


@dataclass
class ExactDMD:
    """Exact DMD fitted on a snapshot matrix.

    Algorithm (Tu et al. 2014):
        1. SVD of X1 = U Σ V*
        2. Atilde = U* X2 V Σ^{-1}
        3. Eigendecomp Atilde W = W Λ
        4. Modes Φ = X2 V Σ^{-1} W   (exact DMD modes)
        5. Continuous-time eigenvalues ω = log(Λ) / dt
        6. Amplitudes b = Φ^† x_1
    """

    rank: int | float | None = None
    dt: float = 1.0

    modes: np.ndarray = field(init=False, repr=False)
    eigenvalues: np.ndarray = field(init=False, repr=False)
    omega: np.ndarray = field(init=False, repr=False)
    amplitudes: np.ndarray = field(init=False, repr=False)
    _x0: np.ndarray = field(init=False, repr=False)

    def fit(self, data: np.ndarray) -> "ExactDMD":
        X = utils.to_snapshots(data).astype(complex)
        X1, X2 = utils.split_snapshots(X)
        U, s, Vh = utils.truncated_svd(X1, rank=self.rank)
        V = Vh.conj().T
        s_inv = 1.0 / s
        Atilde = U.conj().T @ X2 @ V * s_inv
        eigvals, W = np.linalg.eig(Atilde)
        modes = X2 @ V * s_inv @ W
        b, *_ = np.linalg.lstsq(modes, X1[:, 0], rcond=None)
        self.modes = modes
        self.eigenvalues = eigvals
        self.omega = np.log(eigvals) / self.dt
        self.amplitudes = b
        self._x0 = X1[:, 0]
        return self

    def reconstruct(self, t: np.ndarray | None = None) -> np.ndarray:
        if t is None:
            raise ValueError("Pass t (array of times) to reconstruct.")
        return utils.reconstruct(
            self.modes, self.eigenvalues, self.amplitudes, t, dt=self.dt
        )

    def predict(self, n_steps: int) -> np.ndarray:
        """Forecast n_steps into the future from the last fitted snapshot."""
        t = np.arange(1, n_steps + 1) * self.dt
        return utils.reconstruct(
            self.modes, self.eigenvalues, self.amplitudes, t, dt=self.dt
        )

    @property
    def frequencies(self) -> np.ndarray:
        """Imaginary part of omega / (2π) — Hz if dt is in seconds."""
        return np.imag(self.omega) / (2 * np.pi)

    @property
    def growth_rates(self) -> np.ndarray:
        """Real part of omega."""
        return np.real(self.omega)


def exact_dmd(
    data: np.ndarray,
    rank: int | float | None = None,
    dt: float = 1.0,
) -> ExactDMD:
    """Convenience: fit and return an ExactDMD instance."""
    return ExactDMD(rank=rank, dt=dt).fit(data)
