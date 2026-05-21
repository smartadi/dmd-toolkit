"""Demo of all five neural Kutz-group DMD/Koopman methods.

Run:
    cd ~/Documents/Yazdanlab/dmd-toolkit
    .venv/bin/python examples/neural_demo.py

Saves neural_demo.png.
"""

import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dmd_toolkit as dmd
from dmd_toolkit.neural import (
    fit_shred, predict_shred, make_sensor_lag_dataset,
    fit_dpk,
    DiscrepancyModel,
    fit_sks,
    fit_koopman_ae,
)

# ──────────────────────────────────────────────────
# Shared dataset: multi-scale signal (nx=40, nt=600)
# ──────────────────────────────────────────────────
X_ms, x_ms, t_ms = dmd.data.multi_scale_signal(nx=40, nt=600, dt=0.02)
X = X_ms.real
DT = 0.02
SENSORS = [0, 10, 20, 30]
LAGS = 30

print(f"Dataset: {X.shape}  (features × time), dt={DT}")
print("=" * 70)


# ──────────────────────────────────────────────────
# 1. SHRED-ROM
# ──────────────────────────────────────────────────
t0 = time.perf_counter()
shred_model, shred_info = fit_shred(
    X, SENSORS, lags=LAGS, hidden_size=64, decoder_hidden=(200, 200),
    epochs=200, lr=1e-3, verbose=False,
)
shred_time = time.perf_counter() - t0

windows, targets = make_sensor_lag_dataset(X, SENSORS, LAGS)
preds = np.stack([
    predict_shred(shred_model, shred_info, windows[i].astype(np.float32))
    for i in range(len(windows))
], axis=1)
shred_err = np.linalg.norm(X[:, LAGS - 1:] - preds, "fro") / np.linalg.norm(X[:, LAGS - 1:], "fro")
print(f"SHRED-ROM          fit {shred_time:.1f}s   recon err {shred_err:.3f}")


# ──────────────────────────────────────────────────
# 2. Deep Probabilistic Koopman
# ──────────────────────────────────────────────────
t0 = time.perf_counter()
dpk_model, dpk_info = fit_dpk(
    X, latent_dim=8, horizon=16, encoder_hidden=(64, 64),
    epochs=300, batches_per_epoch=32, batch_size=64, verbose=False,
)
dpk_time = time.perf_counter() - t0

dpk_mean, dpk_std = dpk_model.forecast(X[:, 0].astype(np.float32), n_steps=X.shape[1] - 1)
dpk_recon = dpk_mean.T  # (n_features, n_times)
# un-normalise using stored stats
dpk_recon_unscaled = dpk_recon * dpk_info.x_std[:, None] + dpk_info.x_mean[:, None]
dpk_err = np.linalg.norm(X - dpk_recon_unscaled, "fro") / np.linalg.norm(X, "fro")
print(f"Deep Prob Koopman  fit {dpk_time:.1f}s   recon err {dpk_err:.3f}")


# ──────────────────────────────────────────────────
# 3. Discrepancy modelling (exact DMD base, NN residual)
# ──────────────────────────────────────────────────
base_ed = dmd.exact_dmd(X.astype(complex), rank=2, dt=DT)
base_err = np.linalg.norm(X - np.real(base_ed.reconstruct(t_ms)), "fro") / np.linalg.norm(X, "fro")

t0 = time.perf_counter()
disc = DiscrepancyModel(base=base_ed, hidden=(64, 64), epochs=800, lr=2e-3, verbose=False)
disc.fit(X, t_ms)
disc_time = time.perf_counter() - t0

disc_recon = disc.reconstruct(t_ms)
disc_err = np.linalg.norm(X - disc_recon, "fro") / np.linalg.norm(X, "fro")
print(f"Discrepancy model  fit {disc_time:.1f}s   recon err {disc_err:.3f}  (base was {base_err:.3f})")


# ──────────────────────────────────────────────────
# 4. SINDy + Koopman + SHRED
# ──────────────────────────────────────────────────
t0 = time.perf_counter()
sks = fit_sks(
    X, sensor_indices=SENSORS, lags=LAGS, latent_dim=8,
    decoder_hidden=(200, 200), epochs=200, sindy_threshold=0.05, dt=DT, verbose=False,
)
sks_time = time.perf_counter() - t0

n_nonzero = int((sks.sindy_Xi != 0).sum())
z_fcast = sks.forecast_latent(n_steps=50, mode="koopman")
X_sks_fcast = sks.decode_latent(z_fcast)
print(f"SINDy+Koopman+SHRED fit {sks_time:.1f}s  SINDy nonzero terms {n_nonzero}/{sks.sindy_Xi.size}")
print(f"   Koopman eigenvalues: {np.round(np.abs(sks.K_eigvals), 3)} (|λ|)")


# ──────────────────────────────────────────────────
# 5. Koopman Autoencoder (invariant manifold)
# ──────────────────────────────────────────────────
t0 = time.perf_counter()
kae_model, kae_info = fit_koopman_ae(
    X, latent_dim=8, horizon=8, encoder_hidden=(64, 64),
    epochs=400, batches_per_epoch=32, batch_size=64, verbose=False,
)
kae_time = time.perf_counter() - t0

kae_fcast = kae_model.forecast(X[:, 0].astype(np.float32), n_steps=X.shape[1] - 1)
kae_recon_unscaled = kae_fcast * kae_info.x_std[:, None] + kae_info.x_mean[:, None]
kae_err = np.linalg.norm(X - kae_recon_unscaled, "fro") / np.linalg.norm(X, "fro")
evals = kae_model.eigenvalues
phi = kae_model.eigenfunctions(X)
print(f"Koopman Autoencoder fit {kae_time:.1f}s  recon err {kae_err:.3f}")
print(f"   Koopman eigenvalues: {np.round(np.abs(evals), 3)} (|λ|)")


# ──────────────────────────────────────────────────
# Visualisation — 6 panels
# ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

row = X[0, :]  # first spatial mode for 1D plots
feature_idx = 0

# Panel 1: SHRED reconstruction (first feature)
ax = axes[0, 0]
t_shred = t_ms[LAGS - 1:]
ax.plot(t_shred, X[feature_idx, LAGS - 1:], lw=0.8, label="truth")
ax.plot(t_shred, preds[feature_idx], "--", lw=0.8, label=f"SHRED (err {shred_err:.2f})")
ax.set_title("SHRED-ROM: sensor → full state")
ax.set_xlabel("t"); ax.legend(fontsize=8)

# Panel 2: DPK forecast mean ± 1σ
ax = axes[0, 1]
mu = dpk_mean[:, feature_idx] * dpk_info.x_std[feature_idx] + dpk_info.x_mean[feature_idx]
sigma = dpk_std[:, feature_idx] * dpk_info.x_std[feature_idx]
ax.plot(t_ms, X[feature_idx], lw=0.8, label="truth")
ax.plot(t_ms, mu, "--", lw=0.8, label=f"DPK (err {dpk_err:.2f})")
ax.fill_between(t_ms, mu - sigma, mu + sigma, alpha=0.25, label="±1σ")
ax.set_title("Deep Probabilistic Koopman")
ax.set_xlabel("t"); ax.legend(fontsize=8)

# Panel 3: Discrepancy model vs base DMD
ax = axes[0, 2]
ax.plot(t_ms, X[feature_idx], lw=0.8, label="truth")
ax.plot(t_ms, np.real(base_ed.reconstruct(t_ms))[feature_idx], ":", lw=0.8, label=f"ExactDMD (err {base_err:.2f})")
ax.plot(t_ms, disc_recon[feature_idx], "--", lw=0.8, label=f"+Discrepancy (err {disc_err:.2f})")
ax.set_title("Discrepancy Modeling")
ax.set_xlabel("t"); ax.legend(fontsize=8)

# Panel 4: SINDy+Koopman+SHRED — latent Koopman eigenvalue spectrum
ax = axes[1, 0]
theta = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.5, alpha=0.4)
ax.scatter(sks.K_eigvals.real, sks.K_eigvals.imag, s=80)
ax.set_aspect("equal"); ax.set_title("SKS: Latent Koopman eigenvalues")
ax.set_xlabel("Re"); ax.set_ylabel("Im")

# Panel 5: Koopman AE eigenvalues + loss curve
ax = axes[1, 1]
ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.5, alpha=0.4)
ax.scatter(evals.real, evals.imag, s=80, zorder=3)
ax.set_aspect("equal"); ax.set_title("KoopmanAE: eigenvalue spectrum")
ax.set_xlabel("Re"); ax.set_ylabel("Im")

# Panel 6: training loss curves (DPK, KAE)
ax = axes[1, 2]
ax.semilogy(dpk_info.losses, label="DPK loss", lw=0.8)
ax.semilogy(kae_info.losses, label="KAE loss", lw=0.8)
ax.semilogy(disc.losses, label="Discrepancy loss", lw=0.8)
ax.set_xlabel("epoch"); ax.set_ylabel("loss (log)"); ax.legend(fontsize=8)
ax.set_title("Training curves")

plt.tight_layout()
out = pathlib.Path(__file__).parent / "neural_demo.png"
plt.savefig(out, dpi=120)
print(f"\nSaved {out.name}")
