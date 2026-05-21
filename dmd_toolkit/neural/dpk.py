"""Deep Probabilistic Koopman (Mallen & Kutz, Int. J. Forecasting 2024).

Long-horizon forecasting with calibrated uncertainty. Encodes state into a
latent observable space where dynamics are exactly linear, decodes back to a
Gaussian distribution over the original state. Loss combines reconstruction,
multi-step linearity in latent, and a Gaussian NLL.

For the most general Mallen-Kutz setting the linear operator K is replaced by
a time-periodic K(t) for periodic dynamics; the simplified default here is a
single learned K matrix, which is enough to demo the architecture and is what
the user wires up for non-periodic systems.

Usage
-----
>>> model, info = fit_dpk(X, latent_dim=8, epochs=300)
>>> mean, std = model.forecast(X[:, -1], n_steps=100)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def _mlp(dims: list[int], activation=nn.ReLU) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class DeepProbKoopman(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        encoder_hidden: tuple[int, ...] = (64, 64),
        min_logvar: float = -6.0,
        max_logvar: float = 4.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = _mlp([input_dim, *encoder_hidden, latent_dim])
        self.decoder_mean = _mlp([latent_dim, *encoder_hidden, input_dim])
        self.decoder_logvar = _mlp([latent_dim, *encoder_hidden, input_dim])
        K0 = torch.eye(latent_dim) + 0.01 * torch.randn(latent_dim, latent_dim)
        self.K = nn.Parameter(K0)
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        # Normalisation buffers — populated by fit_dpk; kept on model for inference
        self.register_buffer("x_mean", torch.zeros(input_dim))
        self.register_buffer("x_std", torch.ones(input_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def step(self, z: torch.Tensor, n: int = 1) -> torch.Tensor:
        # z: (..., latent_dim); apply K n times
        for _ in range(n):
            z = z @ self.K.T
        return z

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.decoder_mean(z)
        logvar = self.decoder_logvar(z).clamp(self.min_logvar, self.max_logvar)
        return mean, logvar

    def forward(
        self, x0: torch.Tensor, n_steps: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Roll forward from x0 for n_steps; return latent trajectory + decoded mean/logvar.

        x0 : (batch, input_dim)
        Returns
            z_traj : (batch, n_steps+1, latent_dim)
            mean   : (batch, n_steps+1, input_dim)
            logvar : (batch, n_steps+1, input_dim)
        """
        z = self.encode(x0)
        zs = [z]
        for _ in range(n_steps):
            z = z @ self.K.T
            zs.append(z)
        z_traj = torch.stack(zs, dim=1)
        mean, logvar = self.decode(z_traj)
        return z_traj, mean, logvar

    @torch.no_grad()
    def forecast(
        self, x0: np.ndarray, n_steps: int, device: str = "cpu"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return forecast (mean, std) of shape (n_steps+1, input_dim) in original scale."""
        self.eval()
        mu = self.x_mean.to(device)
        sigma = self.x_std.to(device)
        x0_n = (torch.from_numpy(x0.astype(np.float32)).to(device) - mu) / sigma
        _, mean_n, logvar = self.forward(x0_n[None, :], n_steps)
        std_n = torch.exp(0.5 * logvar)
        mean = mean_n[0] * sigma + mu
        std = std_n[0] * sigma
        return mean.cpu().numpy(), std.cpu().numpy()


@dataclass
class DPKInfo:
    losses: list[float]
    x_mean: np.ndarray
    x_std: np.ndarray


def _make_rollout_batches(
    X: np.ndarray, horizon: int, batch_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample random (start_idx, true_trajectory) batches for rollout training.

    X : (n_features, n_times)
    Returns x0 (batch, n_features) and target_traj (batch, horizon+1, n_features).
    """
    n_features, n_times = X.shape
    max_start = n_times - horizon - 1
    starts = rng.integers(0, max_start + 1, size=batch_size)
    x0 = X[:, starts].T.astype(np.float32)
    trajs = np.stack([X[:, s : s + horizon + 1].T for s in starts], axis=0).astype(
        np.float32
    )
    return x0, trajs


def fit_dpk(
    X: np.ndarray,
    latent_dim: int = 8,
    horizon: int = 16,
    encoder_hidden: tuple[int, ...] = (64, 64),
    epochs: int = 300,
    batches_per_epoch: int = 32,
    batch_size: int = 64,
    lr: float = 1e-3,
    w_recon: float = 1.0,
    w_linear: float = 1.0,
    w_nll: float = 1.0,
    device: str = "cpu",
    rng_seed: int = 0,
    verbose: bool = False,
) -> tuple[DeepProbKoopman, DPKInfo]:
    """Train a Deep Probabilistic Koopman model on a snapshot matrix X."""
    torch.manual_seed(rng_seed)
    rng = np.random.default_rng(rng_seed)

    n_features = X.shape[0]
    x_mean = X.mean(axis=1, keepdims=True)
    x_std = X.std(axis=1, keepdims=True) + 1e-8
    Xn = ((X - x_mean) / x_std).astype(np.float32)

    model = DeepProbKoopman(
        input_dim=n_features,
        latent_dim=latent_dim,
        encoder_hidden=encoder_hidden,
    ).to(device)
    model.x_mean.copy_(torch.from_numpy(x_mean.squeeze(-1).astype(np.float32)))
    model.x_std.copy_(torch.from_numpy(x_std.squeeze(-1).astype(np.float32)))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        ep_loss = 0.0
        for _ in range(batches_per_epoch):
            x0_np, traj_np = _make_rollout_batches(Xn, horizon, batch_size, rng)
            x0 = torch.from_numpy(x0_np).to(device)
            traj = torch.from_numpy(traj_np).to(device)  # (B, H+1, F)

            z_traj, mean, logvar = model(x0, horizon)

            # Reconstruction (one-step decode of true encoded points)
            with torch.no_grad():
                z_true = model.encode(traj.reshape(-1, n_features)).reshape(
                    traj.shape[0], traj.shape[1], -1
                )
            recon_loss = ((mean - traj) ** 2).mean()

            # Linearity: latent rollout matches latent of true trajectory
            linear_loss = ((z_traj - z_true) ** 2).mean()

            # NLL: Gaussian log-likelihood of true trajectory under predicted dist.
            inv_var = torch.exp(-logvar)
            nll = 0.5 * (logvar + (traj - mean) ** 2 * inv_var).mean()

            loss = w_recon * recon_loss + w_linear * linear_loss + w_nll * nll
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        ep_loss /= batches_per_epoch
        losses.append(ep_loss)
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:4d}  loss {ep_loss:.4e}")

    info = DPKInfo(losses=losses, x_mean=x_mean.squeeze(-1), x_std=x_std.squeeze(-1))
    return model, info
