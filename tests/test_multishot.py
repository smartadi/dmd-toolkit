"""Tests for multi-shot DMD on trial-stratified data."""

import numpy as np
import pytest
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import dmd_toolkit as dmd


def _make_trials(n_ch=8, n_trials=10, n_time=120, dt=0.05, seed=0):
    """Generate K trials from the same generator A but with random ICs."""
    rng = np.random.default_rng(seed)
    # Spatial modes
    x = np.linspace(-5, 5, n_ch)
    Phi = np.column_stack([
        1.0 / np.cosh(x + 2),
        1.0 / np.cosh(x - 2),
    ]).astype(complex)  # (n_ch, 2)
    # Continuous-time eigenvalues
    omega = np.array([0.0 + 2j * np.pi * 1.0, -0.1 + 2j * np.pi * 4.0])
    t = np.arange(n_time) * dt
    time_dyn = np.exp(np.outer(omega, t))  # (2, n_time)

    X = np.empty((n_ch, n_trials, n_time), dtype=complex)
    for k in range(n_trials):
        b = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        X[:, k, :] = Phi @ (b[:, None] * time_dyn)
    return X, t


def test_multishot_recovers_shared_eigenvalues():
    X, t = _make_trials()
    m = dmd.multi_shot_dmd(X, rank=2, dt=0.05)
    freqs = np.sort(np.abs(m.frequencies))
    assert np.any(np.abs(freqs - 1.0) < 0.05), f"missed 1Hz: {freqs}"
    assert np.any(np.abs(freqs - 4.0) < 0.05), f"missed 4Hz: {freqs}"


def test_multishot_amplitudes_shape():
    X, _ = _make_trials(n_trials=7)
    m = dmd.multi_shot_dmd(X, rank=2, dt=0.05)
    assert m.amplitudes.shape == (7, 2)
    assert m.n_trials == 7


def test_multishot_per_trial_reconstruction():
    X, t = _make_trials(n_trials=5)
    m = dmd.multi_shot_dmd(X, rank=2, dt=0.05)
    for k in range(5):
        rec = m.reconstruct_trial(k, t)
        rel = np.linalg.norm(rec - X[:, k, :], "fro") / np.linalg.norm(X[:, k, :], "fro")
        assert rel < 0.01, f"trial {k} reconstruction error {rel:.3e}"


def test_multishot_reconstruct_all_shape():
    X, t = _make_trials(n_trials=4, n_time=80)
    m = dmd.multi_shot_dmd(X, rank=2, dt=0.05)
    all_rec = m.reconstruct_all(t)
    assert all_rec.shape == (X.shape[0], 4, len(t))


def test_multishot_rejects_2d():
    X = np.zeros((10, 50))
    with pytest.raises(ValueError, match="3D"):
        dmd.multi_shot_dmd(X)


def test_multishot_rejects_too_short():
    X = np.zeros((5, 3, 1))
    with pytest.raises(ValueError, match="at least 2"):
        dmd.multi_shot_dmd(X)


def test_multishot_no_boundary_pairs():
    """Sanity: per-trial reconstruction should beat naive concat-then-DMD when
    trials have unrelated initial conditions (the concat-DMD would try to fit
    a spurious step at every trial boundary).
    """
    X, t = _make_trials(n_trials=10)
    n_ch, n_tr, n_t = X.shape
    # Multi-shot
    ms = dmd.multi_shot_dmd(X, rank=2, dt=0.05)
    rec_ms = np.linalg.norm(ms.reconstruct_all(t) - X) / np.linalg.norm(X)
    # Naive concat: just flatten trial dim into time, fit one DMD
    Xflat = X.reshape(n_ch, n_tr * n_t)
    naive = dmd.exact_dmd(Xflat, rank=2, dt=0.05)
    rec_naive = naive.reconstruct(np.arange(n_tr * n_t) * 0.05)
    rec_naive = np.linalg.norm(rec_naive - Xflat) / np.linalg.norm(Xflat)
    assert rec_ms < rec_naive, (
        f"multi-shot ({rec_ms:.3e}) should beat naive concat ({rec_naive:.3e})"
    )
