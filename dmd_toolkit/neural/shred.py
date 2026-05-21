"""SHRED — SHallow REcurrent Decoder.

Williams, Kutz et al. (Royal Society Proc A 2024; Nature Comm 2025 for ROM variant).

Reconstructs a full high-dimensional state from a sparse sensor time-window:
an LSTM ingests the past `lags` samples at `n_sensors` measurement points
and a shallow MLP decodes the LSTM's final hidden state to the full field.

Usage
-----
>>> import dmd_toolkit as dmd
>>> from dmd_toolkit.neural import fit_shred
>>> X, x, t = dmd.data.multi_scale_signal()
>>> sensors = [10, 30, 50, 70]
>>> model, info = fit_shred(X.real, sensors, lags=40, epochs=200)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class SHRED(nn.Module):
    def __init__(
        self,
        n_sensors: int,
        n_output: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        decoder_hidden: tuple[int, int] = (350, 400),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_sensors,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        h1, h2 = decoder_hidden
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, n_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lags, n_sensors)
        out, _ = self.lstm(x)
        # take final time step's hidden output
        return self.decoder(out[:, -1, :])


def make_sensor_lag_dataset(
    X: np.ndarray,
    sensor_indices: list[int] | np.ndarray,
    lags: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (windows, full_state_at_window_end) pairs.

    X : (n_features, n_times)
    sensor_indices : indices into the feature axis used as observations.
    lags : window length in time.

    Returns
    -------
    inputs : (n_samples, lags, n_sensors)
    targets : (n_samples, n_features)
    """
    X = np.asarray(X)
    n_features, n_times = X.shape
    sensor_indices = np.asarray(sensor_indices)
    sensor_data = X[sensor_indices, :].T  # (n_times, n_sensors)
    n_samples = n_times - lags + 1
    inputs = np.stack([sensor_data[i : i + lags] for i in range(n_samples)], axis=0)
    targets = X[:, lags - 1 :].T  # (n_samples, n_features)
    return inputs.astype(np.float32), targets.astype(np.float32)


@dataclass
class SHREDInfo:
    losses: list[float]
    val_losses: list[float]
    x_mean: np.ndarray
    x_std: np.ndarray
    sensor_mean: np.ndarray
    sensor_std: np.ndarray
    sensor_indices: np.ndarray
    lags: int


def fit_shred(
    X: np.ndarray,
    sensor_indices: list[int] | np.ndarray,
    lags: int = 40,
    hidden_size: int = 64,
    n_layers: int = 2,
    decoder_hidden: tuple[int, int] = (350, 400),
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    device: str = "cpu",
    rng_seed: int = 0,
    verbose: bool = False,
) -> tuple[SHRED, SHREDInfo]:
    """Train a SHRED model. Returns (model, info)."""
    torch.manual_seed(rng_seed)
    rng = np.random.default_rng(rng_seed)

    inputs, targets = make_sensor_lag_dataset(X, sensor_indices, lags)
    # Standardise (per-feature for targets, per-sensor for inputs)
    sensor_mean = inputs.mean(axis=(0, 1))
    sensor_std = inputs.std(axis=(0, 1)) + 1e-8
    x_mean = targets.mean(axis=0)
    x_std = targets.std(axis=0) + 1e-8
    inputs_n = (inputs - sensor_mean) / sensor_std
    targets_n = (targets - x_mean) / x_std

    n_samples = inputs.shape[0]
    perm = rng.permutation(n_samples)
    n_val = max(1, int(n_samples * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    X_tr = torch.from_numpy(inputs_n[train_idx]).to(device)
    Y_tr = torch.from_numpy(targets_n[train_idx]).to(device)
    X_val = torch.from_numpy(inputs_n[val_idx]).to(device)
    Y_val = torch.from_numpy(targets_n[val_idx]).to(device)

    model = SHRED(
        n_sensors=inputs.shape[2],
        n_output=targets.shape[1],
        hidden_size=hidden_size,
        n_layers=n_layers,
        decoder_hidden=decoder_hidden,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    losses, val_losses = [], []
    n_tr = X_tr.shape[0]
    for epoch in range(epochs):
        model.train()
        perm_e = torch.randperm(n_tr)
        ep_loss = 0.0
        for i in range(0, n_tr, batch_size):
            idx = perm_e[i : i + batch_size]
            opt.zero_grad()
            pred = model(X_tr[idx])
            loss = loss_fn(pred, Y_tr[idx])
            loss.backward()
            opt.step()
            ep_loss += loss.item() * idx.shape[0]
        ep_loss /= n_tr
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(X_val), Y_val).item()
        losses.append(ep_loss)
        val_losses.append(v)
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:4d}  train {ep_loss:.4e}  val {v:.4e}")

    info = SHREDInfo(
        losses=losses,
        val_losses=val_losses,
        x_mean=x_mean,
        x_std=x_std,
        sensor_mean=sensor_mean,
        sensor_std=sensor_std,
        sensor_indices=np.asarray(sensor_indices),
        lags=lags,
    )
    return model, info


def predict_shred(
    model: SHRED,
    info: SHREDInfo,
    sensor_window: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Decode a single sensor window (lags, n_sensors) -> full state (n_features,)."""
    model.eval()
    x = (sensor_window.astype(np.float32) - info.sensor_mean) / info.sensor_std
    x = torch.from_numpy(x[None, ...]).to(device)
    with torch.no_grad():
        y = model(x).cpu().numpy()[0]
    return y * info.x_std + info.x_mean
