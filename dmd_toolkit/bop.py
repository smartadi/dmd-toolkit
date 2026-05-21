"""BOP-DMD (Sashidhar & Kutz 2022) — bagging + optimized DMD.

Idea: fit `n_trials` optimized DMD models on random column subsets of the
snapshot matrix, average omega across trials, and report std as uncertainty.

Reference: Sashidhar & Kutz, Phil. Trans. R. Soc. A 380:20210199 (2022).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import utils
from .optimized import OptimizedDMD


@dataclass
class BOPDMD:
    rank: int
    n_trials: int = 32
    subsample_fraction: float = 0.6
    rng_seed: int | None = None
    init: str | np.ndarray = "exact"

    omega_mean: np.ndarray = field(init=False, repr=False)
    omega_std: np.ndarray = field(init=False, repr=False)
    modes_mean: np.ndarray = field(init=False, repr=False)
    amplitudes_mean: np.ndarray = field(init=False, repr=False)
    trials: list = field(init=False, default_factory=list, repr=False)

    def fit(self, data: np.ndarray, t: np.ndarray | None = None) -> "BOPDMD":
        X = utils.to_snapshots(data).astype(complex)
        n_features, n_times = X.shape
        if t is None:
            t = np.arange(n_times, dtype=float)
        t = np.asarray(t, dtype=float)

        rng = np.random.default_rng(self.rng_seed)
        subsample_size = max(self.rank + 2, int(self.subsample_fraction * n_times))

        omegas, modeses, amps = [], [], []
        ref_init = self.init
        for _ in range(self.n_trials):
            idx = np.sort(rng.choice(n_times, size=subsample_size, replace=False))
            X_sub = X[:, idx]
            t_sub = t[idx]
            try:
                model = OptimizedDMD(rank=self.rank, init=ref_init).fit(X_sub, t=t_sub)
            except np.linalg.LinAlgError:
                continue
            omegas.append(model.omega)
            modeses.append(model.modes)
            amps.append(model.amplitudes)
            ref_init = model.omega  # warm-start next trial

        if not omegas:
            raise RuntimeError("All BOP-DMD trials failed.")

        aligned_omegas, aligned_modes, aligned_amps = _align_trials(
            omegas, modeses, amps
        )
        self.omega_mean = aligned_omegas.mean(axis=0)
        self.omega_std = aligned_omegas.std(axis=0)
        self.modes_mean = aligned_modes.mean(axis=0)
        self.amplitudes_mean = aligned_amps.mean(axis=0)
        self.trials = list(zip(omegas, modeses, amps))
        return self

    def reconstruct(self, t: np.ndarray) -> np.ndarray:
        time_dyn = np.exp(np.outer(self.omega_mean, t)) * self.amplitudes_mean[:, None]
        return self.modes_mean @ time_dyn

    def forecast_ensemble(self, t: np.ndarray) -> np.ndarray:
        """Return (n_trials, n_features, len(t)) ensemble reconstruction.

        Useful for uncertainty bands.
        """
        out = []
        for omega, modes, amps in self.trials:
            time_dyn = np.exp(np.outer(omega, t)) * amps[:, None]
            out.append(modes @ time_dyn)
        return np.array(out)


def _align_trials(omegas, modeses, amps):
    """Align eigenvalues across trials by matching to the first trial.

    Greedy nearest-neighbour in complex plane. Cheap and good enough for
    well-separated modes; misorders for near-degenerate eigenvalues.
    """
    ref = omegas[0]
    rank = len(ref)
    aligned_omega = [ref]
    aligned_modes = [modeses[0]]
    aligned_amps = [amps[0]]
    for omega, modes, amp in zip(omegas[1:], modeses[1:], amps[1:]):
        order = _match(ref, omega)
        aligned_omega.append(omega[order])
        aligned_modes.append(modes[:, order])
        aligned_amps.append(amp[order])
    return (
        np.array(aligned_omega),
        np.array(aligned_modes),
        np.array(aligned_amps),
    )


def _match(ref: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Greedy assignment of candidate indices to ref by min |diff|."""
    rank = len(ref)
    used = set()
    order = np.zeros(rank, dtype=int)
    for i, r in enumerate(ref):
        dists = np.abs(candidate - r)
        for j in used:
            dists[j] = np.inf
        pick = int(np.argmin(dists))
        order[i] = pick
        used.add(pick)
    return order


def bop_dmd(
    data: np.ndarray,
    rank: int,
    t: np.ndarray | None = None,
    **kwargs,
) -> BOPDMD:
    return BOPDMD(rank=rank, **kwargs).fit(data, t=t)
