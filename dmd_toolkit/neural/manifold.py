"""Invariant Manifolds + Koopman Eigenfunctions (Morrison & Kutz, SIAM JADS 2024).

Learns a nonlinear embedding x → z where:
  1. The latent z lives on an invariant manifold (enforced by reconstruction fidelity).
  2. Dynamics in z are exactly linear:  z_{t+1} = K z_t  (Koopman linearity).
  3. The eigenvectors of K composed with the encoder give Koopman eigenfunctions
     restricted to the manifold.

Architecture: deep Koopman autoencoder (Lusch, Brunton, Kutz — Nature Comm 2018,
extended by Morrison & Kutz 2024 for explicit invariant-manifold constraints).

Three training losses:
  L_recon  = ||decoder(encoder(x)) - x||^2             (manifold fidelity)
  L_linear = ||encoder(x_{t+1}) - K encoder(x_t)||^2   (Koopman linearity)
  L_pred   = ||decoder(K^h encoder(x_t)) - x_{t+h}||^2 (multi-step prediction)

After training:
  .K           learned linear Koopman operator (latent_dim × latent_dim)
  .eigenvalues complex Koopman eigenvalues
  .eigenfunctions(X)  evaluates eigenfunctions on new data

Usage
-----
>>> from dmd_toolkit.neural import fit_koopman_ae
>>> model, info = fit_koopman_ae(X, latent_dim=8, epochs=400, horizon=4)
>>> evals = model.eigenvalues               # Koopman spectrum
>>> phi = model.eigenfunctions(X[:, :50])   # (latent_dim, 50) eigenfunction values
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


def _mlp(dims: list[int], activation=nn.Tanh) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class KoopmanAutoencoder(nn.Module):
    """Deep Koopman autoencoder with a learned linear operator K in latent space."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        encoder_hidden: tuple[int, ...] = (64, 64),
        spectral_norm: bool = False,
    ):
        super().__init__()
        enc_dims = [input_dim, *encoder_hidden, latent_dim]
        dec_dims = [latent_dim, *reversed(encoder_hidden), input_dim]
        self.encoder = _mlp(enc_dims)
        self.decoder = _mlp(dec_dims)

        K0 = 0.9 * torch.eye(latent_dim) + 0.01 * torch.randn(latent_dim, latent_dim)
        self.K = nn.Parameter(K0)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.register_buffer("x_mean", torch.zeros(input_dim))
        self.register_buffer("x_std", torch.ones(input_dim))

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def step_latent(self, z: torch.Tensor, n: int = 1) -> torch.Tensor:
        for _ in range(n):
            z = z @ self.K.T
        return z

    def forward(
        self, x: torch.Tensor, horizon: int = 1
    ) -> dict[str, torch.Tensor]:
        """Return dict of tensors needed for computing the three losses."""
        z0 = self.encode(x[:, 0, :])
        x_recon = self.decode(z0)

        # Encode every snapshot in the trajectory
        B, H1, F = x.shape
        z_true = self.encode(x.reshape(B * H1, F)).reshape(B, H1, -1)

        # Koopman-stepped z from z0
        z_pred = torch.zeros_like(z_true)
        z_pred[:, 0, :] = z0
        z_k = z0
        for h in range(1, H1):
            z_k = z_k @ self.K.T
            z_pred[:, h, :] = z_k

        x_pred = self.decode(z_pred.reshape(B * H1, -1)).reshape(B, H1, F)

        return {
            "x": x,
            "x_recon": x_recon,
            "z_true": z_true,
            "z_pred": z_pred,
            "x_pred": x_pred,
        }

    # ------------------------------------------------------------------
    @property
    def eigenvalues(self) -> np.ndarray:
        K_np = self.K.detach().cpu().numpy()
        vals, _ = np.linalg.eig(K_np)
        return vals

    @property
    def eigenvectors(self) -> np.ndarray:
        K_np = self.K.detach().cpu().numpy()
        _, vecs = np.linalg.eig(K_np)
        return vecs

    @torch.no_grad()
    def eigenfunctions(self, X: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Evaluate Koopman eigenfunctions phi_j(x) = v_j · encoder(x).

        Returns (latent_dim, n_times) complex array.
        """
        self.eval()
        x_t = torch.from_numpy(X.T.astype(np.float32)).to(device)
        z = self.encoder(x_t).cpu().numpy()  # (n_times, latent_dim)
        _, V = np.linalg.eig(self.K.detach().cpu().numpy())
        return (V.T @ z.T)  # (latent_dim, n_times)

    @torch.no_grad()
    def forecast(
        self, x0: np.ndarray, n_steps: int, device: str = "cpu"
    ) -> np.ndarray:
        """Free-run forecast from x0 in original scale. Returns (n_features, n_steps+1)."""
        self.eval()
        mu = self.x_mean.to(device)
        sigma = self.x_std.to(device)
        x0_n = (torch.from_numpy(x0.astype(np.float32)).to(device) - mu) / sigma
        z = self.encode(x0_n[None, :])
        outs_n = [self.decode(z)[0]]
        for _ in range(n_steps):
            z = z @ self.K.T
            outs_n.append(self.decode(z)[0])
        stack_n = torch.stack(outs_n, dim=1)  # (1, n_steps+1, n_features) → squeeze
        stack = stack_n * sigma[:, None] + mu[:, None]
        return stack.cpu().numpy()  # (n_features, n_steps+1)


# ------------------------------------------------------------------
@dataclass
class KoopmanAEInfo:
    losses: list[float]
    x_mean: np.ndarray
    x_std: np.ndarray


def _make_trajectory_batches(
    X: np.ndarray, horizon: int, batch_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample (batch, horizon+1, n_features) trajectory windows from X."""
    n_features, n_times = X.shape
    max_start = n_times - horizon - 1
    starts = rng.integers(0, max_start + 1, size=batch_size)
    return np.stack(
        [X[:, s : s + horizon + 1].T for s in starts], axis=0
    ).astype(np.float32)


def fit_koopman_ae(
    X: np.ndarray,
    latent_dim: int = 8,
    horizon: int = 4,
    encoder_hidden: tuple[int, ...] = (64, 64),
    epochs: int = 400,
    batches_per_epoch: int = 32,
    batch_size: int = 64,
    lr: float = 1e-3,
    w_recon: float = 1.0,
    w_linear: float = 1.0,
    w_pred: float = 1.0,
    device: str = "cpu",
    rng_seed: int = 0,
    verbose: bool = False,
) -> tuple[KoopmanAutoencoder, KoopmanAEInfo]:
    """Train a Koopman autoencoder (invariant manifold + eigenfunctions)."""
    torch.manual_seed(rng_seed)
    rng = np.random.default_rng(rng_seed)

    n_features = X.shape[0]
    x_mean = X.mean(axis=1, keepdims=True)
    x_std = X.std(axis=1, keepdims=True) + 1e-8
    Xn = ((X - x_mean) / x_std).astype(np.float32)

    model = KoopmanAutoencoder(
        input_dim=n_features,
        latent_dim=latent_dim,
        encoder_hidden=encoder_hidden,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses: list[float] = []
    for epoch in range(epochs):
        ep_loss = 0.0
        for _ in range(batches_per_epoch):
            traj = torch.from_numpy(
                _make_trajectory_batches(Xn, horizon, batch_size, rng)
            ).to(device)  # (B, horizon+1, F)

            out = model(traj, horizon=horizon)

            # L_recon: decoder(encoder(x_0)) ≈ x_0
            recon_loss = ((out["x_recon"] - traj[:, 0, :]) ** 2).mean()

            # L_linear: Koopman-stepped z ≈ true encoded z
            linear_loss = ((out["z_pred"] - out["z_true"]) ** 2).mean()

            # L_pred: multi-step decode ≈ true future states
            pred_loss = ((out["x_pred"] - out["x"]) ** 2).mean()

            loss = w_recon * recon_loss + w_linear * linear_loss + w_pred * pred_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()

        ep_loss /= batches_per_epoch
        losses.append(ep_loss)
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:4d}  loss {ep_loss:.4e}")

    info = KoopmanAEInfo(
        losses=losses,
        x_mean=x_mean.squeeze(-1),
        x_std=x_std.squeeze(-1),
    )
    return model, info
