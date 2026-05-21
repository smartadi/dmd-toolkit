"""Plotting helpers for DMD results. All return (fig, axes) for downstream tweaking."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def eigenvalue_spectrum(
    eigenvalues: np.ndarray,
    title: str = "DMD Eigenvalue Spectrum",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _ax(ax)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, alpha=0.4, label="unit circle")
    ax.scatter(eigenvalues.real, eigenvalues.imag, c="steelblue", zorder=3)
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend()
    return fig, ax


def mode_amplitudes(
    amplitudes: np.ndarray,
    frequencies: np.ndarray | None = None,
    title: str = "Mode Amplitudes",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _ax(ax)
    x = np.abs(frequencies) if frequencies is not None else np.arange(len(amplitudes))
    xlabel = "Frequency (Hz)" if frequencies is not None else "Mode index"
    ax.stem(x, np.abs(amplitudes), basefmt=" ")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("|Amplitude|")
    ax.set_title(title)
    return fig, ax


def reconstruction_vs_truth(
    truth: np.ndarray,
    recon: np.ndarray,
    t: np.ndarray | None = None,
    feature_idx: int = 0,
    title: str = "Reconstruction vs Truth",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _ax(ax)
    x = t if t is not None else np.arange(truth.shape[1])
    ax.plot(x, truth[feature_idx], label="Truth", lw=1.5)
    ax.plot(x, recon[feature_idx].real, "--", label="DMD recon", lw=1.5)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Feature {feature_idx}")
    ax.set_title(title)
    ax.legend()
    return fig, ax


def spacetime_heatmap(
    X: np.ndarray,
    t: np.ndarray | None = None,
    x: np.ndarray | None = None,
    title: str = "Space-Time",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _ax(ax)
    t_arr = t if t is not None else np.arange(X.shape[1])
    x_arr = x if x is not None else np.arange(X.shape[0])
    im = ax.pcolormesh(t_arr, x_arr, X.real, cmap="RdBu_r", shading="auto")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("Space")
    ax.set_title(title)
    return fig, ax


def forcing_signal(
    t: np.ndarray,
    forcing: np.ndarray,
    threshold: float | None = None,
    title: str = "HAVOK Forcing v_r(t)",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _ax(ax)
    ax.plot(t, forcing, lw=0.8, alpha=0.8, label="v_r(t)")
    if threshold is not None:
        ax.axhline(threshold, color="r", ls="--", lw=1, label=f"threshold={threshold:.3f}")
        ax.axhline(-threshold, color="r", ls="--", lw=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend()
    return fig, ax


def _ax(ax: plt.Axes | None) -> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    return fig, ax
