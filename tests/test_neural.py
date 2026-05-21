"""Smoke tests for dmd_toolkit.neural — all five methods."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pytest
import dmd_toolkit as dmd
from dmd_toolkit.neural import (
    fit_shred, predict_shred, make_sensor_lag_dataset,
    fit_dpk,
    DiscrepancyModel,
    fit_sks,
    fit_koopman_ae,
)

RNG = np.random.default_rng(42)
DT = 0.05
NT = 300

@pytest.fixture(scope="module")
def multi_scale():
    X, x, t = dmd.data.multi_scale_signal(nx=20, nt=NT, dt=DT)
    return X.real.astype(np.float64), t


# ──────────────────────────────────────────────────
# SHRED
# ──────────────────────────────────────────────────
def test_shred_output_shape(multi_scale):
    X, t = multi_scale
    sensors = [0, 5, 10, 15]
    model, info = fit_shred(X, sensors, lags=20, epochs=5, batch_size=32,
                            hidden_size=16, decoder_hidden=(32, 32))
    window = X[sensors, :20].T  # (lags=20, n_sensors=4)
    pred = predict_shred(model, info, window)
    assert pred.shape == (X.shape[0],), f"expected ({X.shape[0]},), got {pred.shape}"


def test_shred_dataset_shapes(multi_scale):
    X, _ = multi_scale
    sensors = [0, 10]
    lags = 15
    inp, tgt = make_sensor_lag_dataset(X, sensors, lags)
    assert inp.shape == (NT - lags + 1, lags, len(sensors))
    assert tgt.shape == (NT - lags + 1, X.shape[0])


# ──────────────────────────────────────────────────
# Deep Probabilistic Koopman
# ──────────────────────────────────────────────────
def test_dpk_forecast_shape(multi_scale):
    X, t = multi_scale
    model, info = fit_dpk(X, latent_dim=4, horizon=8, epochs=5,
                          batches_per_epoch=4, batch_size=16,
                          encoder_hidden=(16, 16))
    mean, std = model.forecast(X[:, 0], n_steps=20)
    assert mean.shape == (21, X.shape[0])
    assert std.shape == (21, X.shape[0])
    assert np.all(std >= 0)


# ──────────────────────────────────────────────────
# Discrepancy model
# ──────────────────────────────────────────────────
def test_discrepancy_reconstruct_shape(multi_scale):
    X, t = multi_scale
    base = dmd.exact_dmd(X.astype(complex), rank=2, dt=DT)
    disc = DiscrepancyModel(base=base, hidden=(16, 16), epochs=20)
    disc.fit(X, t)
    X_hat = disc.reconstruct(t)
    assert X_hat.shape == X.shape


def test_discrepancy_improves_base(multi_scale):
    X, t = multi_scale
    base = dmd.exact_dmd(X.astype(complex), rank=1, dt=DT)  # deliberately low rank
    X_base = np.real(base.reconstruct(t))
    disc = DiscrepancyModel(base=base, hidden=(32, 32), epochs=200, lr=3e-3)
    disc.fit(X, t)
    X_disc = disc.reconstruct(t)
    err_base = np.linalg.norm(X - X_base, "fro") / np.linalg.norm(X, "fro")
    err_disc = np.linalg.norm(X - X_disc, "fro") / np.linalg.norm(X, "fro")
    assert err_disc < err_base, f"discrepancy {err_disc:.3f} >= base {err_base:.3f}"


# ──────────────────────────────────────────────────
# SINDy + Koopman + SHRED
# ──────────────────────────────────────────────────
def test_sks_latent_and_forecast(multi_scale):
    X, t = multi_scale
    sks = fit_sks(X, sensor_indices=[0, 5, 10, 15], lags=20, latent_dim=8,
                  decoder_hidden=(32, 32), epochs=5, dt=DT)
    assert sks.latent_traj.shape[1] == 8
    z_k = sks.forecast_latent(n_steps=10, mode="koopman")
    assert z_k.shape == (11, 8)
    z_s = sks.forecast_latent(n_steps=10, mode="sindy")
    assert z_s.shape == (11, 8)
    X_pred = sks.decode_latent(z_k)
    assert X_pred.shape[0] == X.shape[0]


def test_sks_sindy_sparsity(multi_scale):
    X, t = multi_scale
    sks = fit_sks(X, sensor_indices=[0, 10], lags=20, latent_dim=4,
                  decoder_hidden=(16, 16), epochs=3, sindy_threshold=1e6, dt=DT)
    # With a very high threshold all coefficients should be zeroed
    assert (sks.sindy_Xi == 0).all(), "expected all-zero Xi at extreme threshold"


# ──────────────────────────────────────────────────
# Koopman Autoencoder (invariant manifold)
# ──────────────────────────────────────────────────
def test_koopman_ae_eigenvalues(multi_scale):
    X, t = multi_scale
    model, info = fit_koopman_ae(X, latent_dim=4, horizon=3, epochs=5,
                                 batches_per_epoch=4, batch_size=16,
                                 encoder_hidden=(16, 16))
    evals = model.eigenvalues
    assert evals.shape == (4,)


def test_koopman_ae_eigenfunctions_shape(multi_scale):
    X, t = multi_scale
    model, info = fit_koopman_ae(X, latent_dim=4, horizon=3, epochs=5,
                                 batches_per_epoch=4, batch_size=16,
                                 encoder_hidden=(16, 16))
    phi = model.eigenfunctions(X[:, :50])
    assert phi.shape == (4, 50)


def test_koopman_ae_forecast_shape(multi_scale):
    X, t = multi_scale
    model, info = fit_koopman_ae(X, latent_dim=4, horizon=3, epochs=5,
                                 batches_per_epoch=4, batch_size=16,
                                 encoder_hidden=(16, 16))
    X_fcast = model.forecast(X[:, 0].astype(np.float32), n_steps=30)
    assert X_fcast.shape == (X.shape[0], 31)
