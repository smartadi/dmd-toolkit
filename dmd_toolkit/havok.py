"""HAVOK — Hankel Alternative View Of Koopman (Brunton, Brunton, Proctor, Kutz 2017).

Decomposes a scalar chaotic signal into a linear system plus an intermittent
forcing term. Pipeline:

    1. Build Hankel matrix H from time-delay embeddings of x(t).
    2. SVD: H = U Σ V*.
    3. Treat the leading r-1 columns of V as a low-dim attractor v(t).
    4. Fit linear dynamics dv/dt ≈ A v on the first r-1 modes; the r-th mode
       v_r(t) acts as intermittent forcing.

Useful for chaotic / intermittent systems (Lorenz, etc.) where standard DMD
fails because the dynamics aren't linear in observable space.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import utils


@dataclass
class HAVOK:
    delays: int = 100
    rank: int = 15
    dt: float = 0.01

    U: np.ndarray = field(init=False, repr=False)
    s: np.ndarray = field(init=False, repr=False)
    V: np.ndarray = field(init=False, repr=False)
    A: np.ndarray = field(init=False, repr=False)
    B: np.ndarray = field(init=False, repr=False)
    forcing: np.ndarray = field(init=False, repr=False)
    embedding: np.ndarray = field(init=False, repr=False)

    def fit(self, x: np.ndarray) -> "HAVOK":
        x = np.asarray(x).ravel()
        H = utils.hankel(x, self.delays)
        U, s, Vh = np.linalg.svd(H, full_matrices=False)
        r = min(self.rank, len(s))
        self.U = U[:, :r]
        self.s = s[:r]
        self.V = Vh[:r, :].T  # shape (T_eff, r)

        # Finite-difference derivative of V (central difference on interior)
        dV = np.gradient(self.V, self.dt, axis=0)

        # Split state (first r-1) and forcing (last column)
        state = self.V[:, : r - 1]
        dstate = dV[:, : r - 1]
        forcing = self.V[:, r - 1 : r]  # column vector

        # Regress: dstate = state @ A.T + forcing @ B.T
        regressors = np.hstack([state, forcing])
        coeffs, *_ = np.linalg.lstsq(regressors, dstate, rcond=None)
        self.A = coeffs[: r - 1, :].T  # (r-1, r-1)
        self.B = coeffs[r - 1 :, :].T  # (r-1, 1)
        self.forcing = forcing.ravel()
        self.embedding = state
        return self

    def forcing_threshold(self, quantile: float = 0.95) -> float:
        """Heuristic threshold for tagging intermittent forcing events."""
        return float(np.quantile(np.abs(self.forcing), quantile))

    def forcing_events(self, quantile: float = 0.95) -> np.ndarray:
        """Indices where |forcing| exceeds the quantile threshold."""
        thresh = self.forcing_threshold(quantile)
        return np.where(np.abs(self.forcing) > thresh)[0]


def havok(
    x: np.ndarray,
    delays: int = 100,
    rank: int = 15,
    dt: float = 0.01,
) -> HAVOK:
    return HAVOK(delays=delays, rank=rank, dt=dt).fit(x)
