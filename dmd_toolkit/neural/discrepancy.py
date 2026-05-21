"""Discrepancy modelling (Ebers, Steele, Kutz 2024).

Two-stage modelling: fit a base model (here any of the DMD variants), compute
its residual on the training data, then fit a neural residual model on the
residual. Total prediction = base + residual.

The base model only needs to expose `.reconstruct(t)` -> (n_features, len(t)).
ExactDMD, OptimizedDMD, BOPDMD all do.

Usage
-----
>>> import dmd_toolkit as dmd
>>> from dmd_toolkit.neural import DiscrepancyModel
>>> ed = dmd.exact_dmd(X, rank=4, dt=0.05)
>>> disc = DiscrepancyModel(base=ed).fit(X, t)
>>> X_hat = disc.reconstruct(t)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn


class _BaseModel(Protocol):
    def reconstruct(self, t: np.ndarray) -> np.ndarray: ...


def _mlp(dims: list[int]) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class _ResidualNet(nn.Module):
    """Maps time scalar -> residual vector of length n_features."""

    def __init__(self, n_features: int, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        self.net = _mlp([1, *hidden, n_features])

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


@dataclass
class DiscrepancyModel:
    base: _BaseModel
    hidden: tuple[int, ...] = (64, 64)
    epochs: int = 800
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cpu"
    rng_seed: int = 0
    verbose: bool = False

    _net: _ResidualNet | None = field(default=None, init=False, repr=False)
    _t_mean: float = field(default=0.0, init=False)
    _t_std: float = field(default=1.0, init=False)
    _r_mean: np.ndarray | None = field(default=None, init=False, repr=False)
    _r_std: np.ndarray | None = field(default=None, init=False, repr=False)
    losses: list[float] = field(default_factory=list, init=False, repr=False)

    def fit(self, X: np.ndarray, t: np.ndarray) -> "DiscrepancyModel":
        torch.manual_seed(self.rng_seed)
        X = np.asarray(X)
        t = np.asarray(t)
        X_base = np.real(self.base.reconstruct(t))
        residual = (X.real - X_base).astype(np.float32)  # (n_features, n_times)

        self._t_mean = float(t.mean())
        self._t_std = float(t.std()) + 1e-8
        t_n = ((t - self._t_mean) / self._t_std).astype(np.float32)[:, None]
        self._r_mean = residual.mean(axis=1, keepdims=True)
        self._r_std = residual.std(axis=1, keepdims=True) + 1e-8
        target = ((residual - self._r_mean) / self._r_std).T  # (n_times, n_features)

        n_features = X.shape[0]
        self._net = _ResidualNet(n_features=n_features, hidden=self.hidden).to(
            self.device
        )
        opt = torch.optim.Adam(
            self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        t_t = torch.from_numpy(t_n).to(self.device)
        y_t = torch.from_numpy(target.astype(np.float32)).to(self.device)
        self.losses = []
        for epoch in range(self.epochs):
            opt.zero_grad()
            pred = self._net(t_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            opt.step()
            self.losses.append(loss.item())
            if self.verbose and (
                epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1
            ):
                print(f"  epoch {epoch:4d}  loss {loss.item():.4e}")
        return self

    def residual(self, t: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Call fit() first.")
        t = np.asarray(t)
        t_n = ((t - self._t_mean) / self._t_std).astype(np.float32)[:, None]
        self._net.eval()
        with torch.no_grad():
            r = self._net(torch.from_numpy(t_n).to(self.device)).cpu().numpy().T
        return r * self._r_std + self._r_mean

    def reconstruct(self, t: np.ndarray) -> np.ndarray:
        return np.real(self.base.reconstruct(t)) + self.residual(t)
