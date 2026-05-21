"""Optimized DMD (Askham & Kutz 2018) — variable projection for nonlinear LS.

Solves: min_{omega, B}  || X - Phi(omega) B ||_F^2
where Phi(omega)_{ij} = exp(omega_j * t_i), and B (= amplitudes * modes) is
recovered by linear least-squares once omega is fixed. Variable projection
optimizes only over omega.

This implementation uses scipy.optimize.least_squares (Levenberg–Marquardt /
trust-region) on the variable-projected residual. It handles unevenly spaced
samples, which exact DMD cannot.

For the canonical reference implementation see Askham's `optdmd` (MATLAB) and
the PyDMD `BOPDMD` / `OptDMD` ports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from . import utils
from .exact import ExactDMD


@dataclass
class OptimizedDMD:
    """Optimized DMD with variable projection.

    Parameters
    ----------
    rank : int
        Number of modes / eigenvalues to fit. Required (no automatic choice).
    init : {"exact", "given"} or array
        Initial omega guess. "exact" uses exact DMD eigenvalues from evenly
        sampled data. An array of shape (rank,) provides explicit init.
    max_iter : int
        Max iterations for the nonlinear solver.
    tol : float
        Convergence tolerance.
    """

    rank: int
    init: str | np.ndarray = "exact"
    max_iter: int = 200
    tol: float = 1e-8

    modes: np.ndarray = field(init=False, repr=False)
    omega: np.ndarray = field(init=False, repr=False)
    amplitudes: np.ndarray = field(init=False, repr=False)
    converged: bool = field(init=False, default=False)

    def fit(self, data: np.ndarray, t: np.ndarray | None = None) -> "OptimizedDMD":
        X = utils.to_snapshots(data).astype(complex)
        n_features, n_times = X.shape
        if t is None:
            t = np.arange(n_times, dtype=float)
        t = np.asarray(t, dtype=float)
        if len(t) != n_times:
            raise ValueError("len(t) must equal number of snapshots")

        omega0 = self._initial_omega(X, t)

        def residual(omega_real):
            omega = omega_real[: self.rank] + 1j * omega_real[self.rank :]
            Phi_t = np.exp(np.outer(t, omega))
            B, *_ = np.linalg.lstsq(Phi_t, X.T, rcond=None)
            resid = (X.T - Phi_t @ B).ravel()
            return np.concatenate([resid.real, resid.imag])

        x0 = np.concatenate([omega0.real, omega0.imag])
        result = least_squares(
            residual,
            x0,
            method="lm",
            max_nfev=self.max_iter * (2 * self.rank),
            xtol=self.tol,
            ftol=self.tol,
        )
        self.converged = result.success
        omega = result.x[: self.rank] + 1j * result.x[self.rank :]

        Phi_t = np.exp(np.outer(t, omega))
        B, *_ = np.linalg.lstsq(Phi_t, X.T, rcond=None)
        B = B.T

        amplitudes = np.linalg.norm(B, axis=0)
        modes = B / np.where(amplitudes == 0, 1, amplitudes)

        self.omega = omega
        self.modes = modes
        self.amplitudes = amplitudes
        return self

    def _initial_omega(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        if isinstance(self.init, np.ndarray):
            if len(self.init) != self.rank:
                raise ValueError("init array length must equal rank")
            return self.init.astype(complex)
        if self.init == "exact":
            dt_est = float(np.mean(np.diff(t))) if len(t) > 1 else 1.0
            exact = ExactDMD(rank=self.rank, dt=dt_est).fit(X)
            return exact.omega
        raise ValueError(f"Unknown init mode: {self.init!r}")

    def reconstruct(self, t: np.ndarray) -> np.ndarray:
        time_dynamics = np.exp(np.outer(self.omega, t)) * self.amplitudes[:, None]
        return self.modes @ time_dynamics

    @property
    def frequencies(self) -> np.ndarray:
        return np.imag(self.omega) / (2 * np.pi)

    @property
    def growth_rates(self) -> np.ndarray:
        return np.real(self.omega)


def optimized_dmd(
    data: np.ndarray,
    rank: int,
    t: np.ndarray | None = None,
    **kwargs,
) -> OptimizedDMD:
    return OptimizedDMD(rank=rank, **kwargs).fit(data, t=t)
