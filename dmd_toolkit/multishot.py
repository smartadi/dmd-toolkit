"""Multi-shot DMD — shared-operator fit across many trials.

Data layout
-----------
Input ``X`` has shape ``(n_channels, n_trials, n_time)`` (e.g. neuroscience
trial-stratified recordings). Each trial is assumed to be drawn from the
*same* underlying dynamical system :math:`x_{t+1} = A x_t`, differing only
in initial condition / trial-specific noise.

Algorithm (Tu et al. 2014 + multi-shot extension)
-------------------------------------------------
Form trial-local snapshot pairs

.. math::
    X_1^{(k)} = X[:, k, 0{:}T-1], \\quad X_2^{(k)} = X[:, k, 1{:}T],

then concatenate **along the time/column axis** (never across the
trial-to-trial discontinuity)

.. math::
    \\mathbf X_1 = [X_1^{(1)} \\,|\\, X_1^{(2)} \\,|\\, \\cdots \\,|\\, X_1^{(K)}],
    \\quad
    \\mathbf X_2 = [X_2^{(1)} \\,|\\, X_2^{(2)} \\,|\\, \\cdots \\,|\\, X_2^{(K)}].

A single Exact DMD fit on :math:`(\\mathbf X_1, \\mathbf X_2)` produces one
shared operator :math:`A` whose eigenpairs :math:`(\\lambda_j, \\phi_j)`
characterise the *common* dynamics. Trial-to-trial variability lives in the
per-trial amplitudes

.. math::
    b^{(k)} = \\Phi^{+}\\, x^{(k)}(0).

This is the "data-stacked" or "multi-shot" DMD formulation used in fluids
(Tu 2013 thesis, Sec. 1.4.2) and adopted for neural ensembles in
Brunton et al. 2016 (Sci. Reports) and Kunert-Graf et al. 2019
(*Front. Comput. Neurosci.*) for trial-stratified ECoG / Neuropixels data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import utils


@dataclass
class MultiShotDMD:
    """DMD over many trials with a single shared operator.

    Parameters
    ----------
    rank : int | float | None
        SVD truncation. Same semantics as :func:`utils.truncated_svd`.
    dt : float
        Sampling interval.

    Attributes after :meth:`fit`
    ---------------------------
    modes : (n_channels, rank) complex
        Shared spatial modes.
    eigenvalues : (rank,) complex
        Discrete-time eigenvalues :math:`\\lambda_j`.
    omega : (rank,) complex
        Continuous-time eigenvalues :math:`\\omega_j = \\log\\lambda_j / \\Delta t`.
    amplitudes : (n_trials, rank) complex
        Per-trial amplitudes :math:`b^{(k)} = \\Phi^{+} x^{(k)}(0)`.
    """

    rank: int | float | None = None
    dt: float = 1.0

    modes: np.ndarray = field(init=False, repr=False)
    eigenvalues: np.ndarray = field(init=False, repr=False)
    omega: np.ndarray = field(init=False, repr=False)
    amplitudes: np.ndarray = field(init=False, repr=False)
    n_trials: int = field(init=False, default=0)
    n_time: int = field(init=False, default=0)

    def fit(self, data: np.ndarray) -> "MultiShotDMD":
        if data.ndim != 3:
            raise ValueError(
                f"multi-shot DMD expects 3D (channels, trials, time) array; "
                f"got shape {data.shape}"
            )
        X = np.asarray(data).astype(complex)
        n_ch, n_tr, n_t = X.shape
        if n_t < 2:
            raise ValueError("Each trial must have at least 2 time samples.")

        # Trial-local snapshot pairs, then concatenate along the time axis.
        # No pair ever crosses a trial boundary.
        X1 = X[:, :, :-1].reshape(n_ch, n_tr * (n_t - 1))
        X2 = X[:, :, 1:].reshape(n_ch, n_tr * (n_t - 1))

        U, s, Vh = utils.truncated_svd(X1, rank=self.rank)
        V = Vh.conj().T
        s_inv = 1.0 / s
        Atilde = U.conj().T @ X2 @ V * s_inv
        eigvals, W = np.linalg.eig(Atilde)
        modes = X2 @ V * s_inv @ W

        # Per-trial amplitudes from trial-zero snapshots.
        x0_per_trial = X[:, :, 0]                      # (n_ch, n_tr)
        B, *_ = np.linalg.lstsq(modes, x0_per_trial, rcond=None)
        amplitudes = B.T                               # (n_tr, rank)

        self.modes = modes
        self.eigenvalues = eigvals
        self.omega = np.log(eigvals.astype(complex)) / self.dt
        self.amplitudes = amplitudes
        self.n_trials = n_tr
        self.n_time = n_t
        return self

    # ------------------------------------------------------------------
    def reconstruct_trial(self, trial_idx: int, t: np.ndarray) -> np.ndarray:
        """Reconstruct a single trial at times ``t`` (relative to its t=0)."""
        b = self.amplitudes[trial_idx]
        return utils.reconstruct(
            self.modes, self.eigenvalues, b, t, dt=self.dt
        )

    def reconstruct_all(self, t: np.ndarray) -> np.ndarray:
        """Reconstruct every trial at times ``t``; returns ``(n_ch, n_tr, len(t))``."""
        out = np.empty((self.modes.shape[0], self.n_trials, len(t)), dtype=complex)
        for k in range(self.n_trials):
            out[:, k, :] = self.reconstruct_trial(k, t)
        return out

    def predict_trial(self, trial_idx: int, n_steps: int) -> np.ndarray:
        """Forecast ``n_steps`` past the end of a trial."""
        t = np.arange(1, n_steps + 1) * self.dt
        return self.reconstruct_trial(trial_idx, t)

    @property
    def frequencies(self) -> np.ndarray:
        return np.imag(self.omega) / (2 * np.pi)

    @property
    def growth_rates(self) -> np.ndarray:
        return np.real(self.omega)

    @property
    def trial_initial_conditions(self) -> np.ndarray:
        """Reconstructed initial conditions :math:`\\Phi b^{(k)}` per trial."""
        return self.modes @ self.amplitudes.T


def multi_shot_dmd(
    data: np.ndarray,
    rank: int | float | None = None,
    dt: float = 1.0,
) -> MultiShotDMD:
    """Convenience: fit and return a :class:`MultiShotDMD` instance.

    Parameters
    ----------
    data : (n_channels, n_trials, n_time) array
        Trial-stratified snapshots.
    rank : int | float | None
        SVD truncation. ``None`` keeps all modes.
    dt : float
        Sampling interval.
    """
    return MultiShotDMD(rank=rank, dt=dt).fit(data)
