"""Smoke tests for exact DMD and utilities."""

import numpy as np
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import dmd_toolkit as dmd
from dmd_toolkit.utils import hankel, to_snapshots, truncated_svd


def _clean_oscillator():
    # Two complex-exponential modes — data is exactly rank-2
    return dmd.data.two_mode_oscillator(nx=100, nt=200, dt=0.05, noise=0.0)


def test_exact_dmd_reconstruction():
    X, x, t = _clean_oscillator()
    ed = dmd.exact_dmd(X, rank=2, dt=0.05)  # rank-2 data → rank-2 DMD
    X_hat = ed.reconstruct(t)
    rel_err = np.linalg.norm(X - X_hat, "fro") / np.linalg.norm(X, "fro")
    assert rel_err < 0.01, f"Exact DMD rel error too high: {rel_err:.4f}"


def test_exact_dmd_frequencies():
    X, x, t = _clean_oscillator()
    ed = dmd.exact_dmd(X, rank=2, dt=0.05)
    freqs = np.sort(np.abs(ed.frequencies))
    assert np.any(np.abs(freqs - 1.0) < 0.1), f"Expected ~1 Hz mode, got {freqs}"
    assert np.any(np.abs(freqs - 5.5) < 0.1), f"Expected ~5.5 Hz mode, got {freqs}"


def test_exact_dmd_rank_fraction():
    X, x, t = _clean_oscillator()
    # 0.99 energy fraction on rank-2 data should give 2 modes
    ed = dmd.exact_dmd(X, rank=0.99, dt=0.05)
    assert ed.modes.shape[1] >= 2


def test_mr_dmd_runs():
    X, x, t = _clean_oscillator()
    mr = dmd.mr_dmd(X, max_levels=2, rank=4, dt=0.05)
    assert len(mr.levels) >= 1
    X_mr = mr.reconstruct(X.shape[1])
    assert X_mr.shape == X.shape


def test_havok_runs():
    _, t = dmd.data.lorenz(n_steps=3000, dt=0.01)
    lorenz, _ = dmd.data.lorenz(n_steps=3000, dt=0.01)
    hav = dmd.havok(lorenz[0], delays=50, rank=10, dt=0.01)
    assert hav.A.shape == (9, 9)
    assert len(hav.forcing) > 0


def test_hankel_shape():
    x = np.arange(100, dtype=float)
    H = hankel(x, delays=10)
    assert H.shape == (10, 91)


def test_to_snapshots_1d():
    x = np.ones(50)
    X = to_snapshots(x)
    assert X.shape == (1, 50)


def test_truncated_svd_fraction():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 100))
    U, s, Vh = truncated_svd(X, rank=0.9)
    assert len(s) < 50
    assert len(s) >= 1
