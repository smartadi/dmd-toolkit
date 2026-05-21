"""SINDy + Koopman + SHRED (Gao, Williams, Kutz — L4DC 2025).

Pipeline:
  1. Train a SHRED model that maps sparse sensor windows -> full state.
  2. Extract the LSTM hidden-state trajectory as a low-dim latent embedding.
  3. In that latent space, fit both:
       - a sparse-symbolic SINDy model  dz/dt = Theta(z) @ Xi
       - a linear Koopman operator       z[t+1] = K z[t]
  Either model lets you forecast in the latent and decode back through SHRED.

This is one concrete interpretation of the Gao/Williams/Kutz pipeline — using
SHRED's recurrent latent as the Koopman observable space.

Usage
-----
>>> sks = fit_sks(X, sensor_indices=[5, 25, 45], lags=40, latent_dim=8)
>>> z_pred = sks.forecast_latent(n_steps=100, mode="koopman")
>>> X_pred = sks.decode_latent(z_pred)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .shred import SHRED, SHREDInfo, fit_shred, make_sensor_lag_dataset


# ──────────────────────────────────────────────────
# SINDy helpers — polynomial library + STLSQ
# ──────────────────────────────────────────────────
def _poly_library(Z: np.ndarray, order: int = 2, include_bias: bool = True) -> tuple:
    """Polynomial feature library up to `order`. Returns (Theta, feature_names)."""
    n, d = Z.shape
    feats: list[np.ndarray] = []
    names: list[str] = []
    if include_bias:
        feats.append(np.ones((n, 1)))
        names.append("1")
    # order 1
    for i in range(d):
        feats.append(Z[:, i : i + 1])
        names.append(f"z{i}")
    # order >=2
    from itertools import combinations_with_replacement

    for o in range(2, order + 1):
        for combo in combinations_with_replacement(range(d), o):
            col = np.ones(n)
            for j in combo:
                col = col * Z[:, j]
            feats.append(col[:, None])
            names.append("*".join(f"z{j}" for j in combo))
    return np.concatenate(feats, axis=1), names


def _stlsq(Theta: np.ndarray, dZ: np.ndarray, threshold: float, n_iter: int = 10):
    """Sequential thresholded least squares: solve Theta @ Xi = dZ sparsely."""
    Xi, *_ = np.linalg.lstsq(Theta, dZ, rcond=None)
    for _ in range(n_iter):
        small = np.abs(Xi) < threshold
        Xi[small] = 0
        for k in range(dZ.shape[1]):
            big = ~small[:, k]
            if big.any():
                Xi[big, k], *_ = np.linalg.lstsq(Theta[:, big], dZ[:, k], rcond=None)
    return Xi


# ──────────────────────────────────────────────────
# Wrapper
# ──────────────────────────────────────────────────
@dataclass
class SINDyKoopmanSHRED:
    shred: SHRED
    info: SHREDInfo
    latent_traj: np.ndarray  # (n_samples, latent_dim)
    # SINDy
    sindy_Xi: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    sindy_features: list[str] = field(default_factory=list)
    sindy_order: int = 2
    sindy_threshold: float = 0.05
    # Koopman (DMD on latent)
    K: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    K_eigvals: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=complex))
    K_modes: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=complex))
    dt: float = 1.0
    device: str = "cpu"

    @property
    def latent_dim(self) -> int:
        return self.latent_traj.shape[1]

    def _theta(self, Z: np.ndarray) -> np.ndarray:
        Theta, _ = _poly_library(Z, order=self.sindy_order)
        return Theta

    def forecast_latent(
        self, n_steps: int, z0: np.ndarray | None = None, mode: str = "koopman"
    ) -> np.ndarray:
        """Roll the latent forward. Returns (n_steps+1, latent_dim)."""
        if z0 is None:
            z0 = self.latent_traj[-1].copy()
        out = np.empty((n_steps + 1, self.latent_dim))
        out[0] = z0
        if mode == "koopman":
            for i in range(n_steps):
                out[i + 1] = self.K @ out[i]
        elif mode == "sindy":
            # Forward Euler with the symbolic model
            for i in range(n_steps):
                Theta_i = self._theta(out[i : i + 1])  # (1, p)
                dz = (Theta_i @ self.sindy_Xi)[0]
                out[i + 1] = out[i] + self.dt * dz
        else:
            raise ValueError("mode must be 'koopman' or 'sindy'")
        return out

    def decode_latent(self, Z: np.ndarray) -> np.ndarray:
        """Decode latent trajectory (n_steps, latent_dim) -> full state via SHRED decoder.

        Returns (n_features, n_steps).
        """
        self.shred.eval()
        with torch.no_grad():
            z_t = torch.from_numpy(Z.astype(np.float32)).to(self.device)
            y = self.shred.decoder(z_t).cpu().numpy()  # (n_steps, n_features)
        return (y * self.info.x_std + self.info.x_mean).T


def _extract_latent(
    shred: SHRED, sensor_windows: np.ndarray, device: str = "cpu"
) -> np.ndarray:
    """Run sensor windows through the LSTM and return final hidden state per window."""
    shred.eval()
    with torch.no_grad():
        x = torch.from_numpy(sensor_windows.astype(np.float32)).to(device)
        out, _ = shred.lstm(x)
        z = out[:, -1, :].cpu().numpy()
    return z


def fit_sks(
    X: np.ndarray,
    sensor_indices: list[int] | np.ndarray,
    lags: int = 40,
    latent_dim: int = 8,
    decoder_hidden: tuple[int, int] = (128, 128),
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    sindy_order: int = 2,
    sindy_threshold: float = 0.05,
    dt: float = 1.0,
    device: str = "cpu",
    rng_seed: int = 0,
    verbose: bool = False,
) -> SINDyKoopmanSHRED:
    """End-to-end: train SHRED, extract latent, fit SINDy + Koopman in latent."""
    shred, info = fit_shred(
        X,
        sensor_indices=sensor_indices,
        lags=lags,
        hidden_size=latent_dim,
        n_layers=2,
        decoder_hidden=decoder_hidden,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        rng_seed=rng_seed,
        verbose=verbose,
    )

    sensor_windows, _ = make_sensor_lag_dataset(X, sensor_indices, lags)
    sensor_windows_n = (sensor_windows - info.sensor_mean) / info.sensor_std
    Z = _extract_latent(shred, sensor_windows_n, device=device)  # (n_samples, latent_dim)

    # ── SINDy on (Z, dZ/dt)
    dZ = np.gradient(Z, dt, axis=0)
    Theta, names = _poly_library(Z, order=sindy_order)
    Xi = _stlsq(Theta, dZ, threshold=sindy_threshold)

    # ── Koopman (DMD) on Z trajectory: Z2 ≈ K Z1
    Z1 = Z[:-1].T  # (latent_dim, n-1)
    Z2 = Z[1:].T
    K, *_ = np.linalg.lstsq(Z1.T, Z2.T, rcond=None)
    K = K.T  # so that z' = K z
    eigvals, modes = np.linalg.eig(K)

    return SINDyKoopmanSHRED(
        shred=shred,
        info=info,
        latent_traj=Z,
        sindy_Xi=Xi,
        sindy_features=names,
        sindy_order=sindy_order,
        sindy_threshold=sindy_threshold,
        K=K,
        K_eigvals=eigvals,
        K_modes=modes,
        dt=dt,
        device=device,
    )
