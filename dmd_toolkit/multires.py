"""Multi-resolution DMD (Kutz, Fu, Brunton 2016).

Recursively splits the snapshot matrix in time, applies DMD at each level,
and separates "slow" modes (low |omega|) from "fast" residuals which are
passed to the next finer resolution level.

The slow-mode threshold at level L is rho_L = (sample_rate / 2^L) * cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import utils
from .exact import ExactDMD


@dataclass
class MultiResolutionDMD:
    """Multi-Resolution DMD (Kutz, Fu, Brunton 2016).

    `slow_cutoff` is "cycles per window" — modes whose frequency × window
    is below this are captured at the current level. Default 5 means modes
    with up to 5 cycles inside the current window are deemed slow.
    """

    max_levels: int = 6
    rank: int | None = None
    slow_cutoff: float = 5.0
    dt: float = 1.0

    levels: list = field(init=False, default_factory=list, repr=False)

    def fit(self, data: np.ndarray) -> "MultiResolutionDMD":
        X = utils.to_snapshots(data).astype(complex)
        self.levels = []
        self._recurse(X, level=0, t_offset=0.0)
        return self

    def _recurse(self, X: np.ndarray, level: int, t_offset: float) -> None:
        if level >= self.max_levels or X.shape[1] < 4:
            return

        n_times = X.shape[1]
        window_duration = n_times * self.dt
        freq_cutoff = self.slow_cutoff / window_duration  # cycles/sec
        omega_cutoff = 2 * np.pi * freq_cutoff

        try:
            model = ExactDMD(rank=self.rank, dt=self.dt).fit(X)
        except np.linalg.LinAlgError:
            return
        slow_mask = np.abs(model.omega.imag) <= omega_cutoff

        slow_modes = model.modes[:, slow_mask]
        slow_omega = model.omega[slow_mask]
        slow_amps = model.amplitudes[slow_mask]

        self.levels.append(
            {
                "level": level,
                "t_offset": t_offset,
                "n_times": n_times,
                "modes": slow_modes,
                "omega": slow_omega,
                "amplitudes": slow_amps,
                "omega_cutoff": omega_cutoff,
            }
        )

        # Subtract slow contribution and recurse on halves
        t_local = np.arange(n_times) * self.dt
        slow_recon = utils.reconstruct(
            slow_modes, slow_omega, slow_amps, t_local, dt=None
        ) if slow_modes.size else np.zeros_like(X)
        residual = X - slow_recon

        mid = n_times // 2
        self._recurse(residual[:, :mid], level + 1, t_offset)
        self._recurse(residual[:, mid:], level + 1, t_offset + mid * self.dt)

    def reconstruct(self, n_times: int) -> np.ndarray:
        """Sum all level reconstructions onto a uniform grid of n_times samples."""
        full_t = np.arange(n_times) * self.dt
        if not self.levels:
            raise RuntimeError("Call fit() first.")
        n_features = self.levels[0]["modes"].shape[0]
        out = np.zeros((n_features, n_times), dtype=complex)
        for L in self.levels:
            if L["modes"].size == 0:
                continue
            start_idx = int(round(L["t_offset"] / self.dt))
            end_idx = start_idx + L["n_times"]
            if start_idx >= n_times:
                continue
            end_idx = min(end_idx, n_times)
            t_local = np.arange(end_idx - start_idx) * self.dt
            chunk = utils.reconstruct(
                L["modes"], L["omega"], L["amplitudes"], t_local, dt=None
            )
            out[:, start_idx:end_idx] += chunk
        return out


def mr_dmd(
    data: np.ndarray,
    max_levels: int = 4,
    rank: int | None = None,
    dt: float = 1.0,
    slow_cutoff: float = 0.5,
) -> MultiResolutionDMD:
    return MultiResolutionDMD(
        max_levels=max_levels, rank=rank, dt=dt, slow_cutoff=slow_cutoff
    ).fit(data)
