"""Snapshot wrangling, SVD truncation, and reconstruction helpers."""

from __future__ import annotations

import numpy as np


def to_snapshots(data: np.ndarray) -> np.ndarray:
    """Coerce arbitrary input to a (features, time) snapshot matrix.

    - 1D input -> single-feature time series, shape (1, T).
    - 2D input -> assumed (features, T). If T < features and that looks wrong
      to the caller, transpose before passing in.
    - 3D+ input -> flattens all non-time leading axes; assumes last axis is time.
    """
    arr = np.asarray(data)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(-1, arr.shape[-1])


def split_snapshots(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (X1, X2) with X1 = X[:, :-1], X2 = X[:, 1:]."""
    return X[:, :-1], X[:, 1:]


def truncated_svd(
    X: np.ndarray,
    rank: int | float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD with optional rank truncation.

    rank semantics:
      - None -> no truncation (min(m, n) modes kept).
      - int  -> keep that many modes.
      - 0 < float < 1 -> keep enough modes to capture that fraction of energy.
    """
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    if rank is None:
        r = len(s)
    elif isinstance(rank, float) and 0 < rank < 1:
        cum = np.cumsum(s**2) / np.sum(s**2)
        r = int(np.searchsorted(cum, rank) + 1)
    else:
        r = min(int(rank), len(s))
    return U[:, :r], s[:r], Vh[:r, :]


def hankel(x: np.ndarray, delays: int) -> np.ndarray:
    """Build a Hankel (time-delay) matrix of shape (delays, T - delays + 1).

    x : 1D array of length T.
    """
    x = np.asarray(x).ravel()
    T = len(x)
    if delays >= T:
        raise ValueError(f"delays={delays} must be < signal length {T}")
    cols = T - delays + 1
    H = np.empty((delays, cols), dtype=x.dtype)
    for i in range(delays):
        H[i] = x[i : i + cols]
    return H


def reconstruct(
    modes: np.ndarray,
    eigenvalues: np.ndarray,
    amplitudes: np.ndarray,
    t: np.ndarray,
    dt: float | None = None,
) -> np.ndarray:
    """Reconstruct snapshots from DMD modes + eigenvalues + amplitudes.

    If `dt` is provided, eigenvalues are converted to continuous-time omega
    via omega = log(lambda) / dt and x(t) = Phi @ diag(exp(omega t)) @ b.
    Otherwise eigenvalues are treated as already-continuous omega.
    """
    if dt is not None:
        omega = np.log(eigenvalues.astype(complex)) / dt
    else:
        omega = eigenvalues.astype(complex)
    time_dynamics = np.exp(np.outer(omega, t)) * amplitudes[:, None]
    return modes @ time_dynamics


def sort_by_magnitude(
    modes: np.ndarray,
    eigenvalues: np.ndarray,
    amplitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort modes by |amplitude * mode_norm| descending."""
    mode_norms = np.linalg.norm(modes, axis=0)
    importance = np.abs(amplitudes) * mode_norms
    order = np.argsort(importance)[::-1]
    return modes[:, order], eigenvalues[order], amplitudes[order]
