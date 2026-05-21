"""Synthetic datasets for testing DMD methods."""

from __future__ import annotations

import numpy as np


def two_mode_oscillator(
    nx: int = 200,
    nt: int = 400,
    dt: float = 0.05,
    freqs: tuple[float, float] = (1.0, 5.5),
    decay: tuple[float, float] = (0.0, -0.2),
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two spatially separated complex-exponential modes (Kutz textbook canonical example).

    Returns complex snapshot matrix so DMD can cleanly recover the eigenvalues.
    For real-valued data in practice, apply scipy.signal.hilbert() first or use
    Hankel embedding (see havok()).

    Returns
    -------
    X : (nx, nt) complex snapshot matrix
    x : (nx,) spatial grid
    t : (nt,) time grid
    """
    x = np.linspace(-10, 10, nx)
    t = np.arange(nt) * dt
    f1 = (1.0 / np.cosh(x + 5)).astype(complex)
    f2 = (1.0 / np.cosh(x - 5)).astype(complex)
    omega1 = decay[0] + 2j * np.pi * freqs[0]
    omega2 = decay[1] + 2j * np.pi * freqs[1]
    X = np.outer(f1, np.exp(omega1 * t)) + np.outer(f2, np.exp(omega2 * t))
    if noise > 0:
        rng = np.random.default_rng(seed)
        X = X + noise * rng.standard_normal(X.shape)
    return X, x, t


def multi_scale_signal(
    nx: int = 80,
    nt: int = 512,
    dt: float = 0.02,
    slow_freq: float = 0.3,
    fast_freq: float = 6.0,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two complex-exponential modes at widely separated time scales.

    Designed to showcase Multi-Resolution DMD (Kutz, Fu, Brunton 2016).
    The slow mode is a spatially smooth background; the fast mode is
    spatially localised "foreground" oscillation.
    """
    x = np.linspace(0, 2 * np.pi, nx)
    t = np.arange(nt) * dt
    f_slow = (np.sin(x) + 0.3).astype(complex)
    f_fast = np.exp(-((x - np.pi) ** 2) / 0.4).astype(complex)
    X = (
        np.outer(f_slow, np.exp(2j * np.pi * slow_freq * t))
        + np.outer(f_fast, np.exp(2j * np.pi * fast_freq * t))
    )
    if noise > 0:
        rng = np.random.default_rng(seed)
        X = X + noise * rng.standard_normal(X.shape)
    return X, x, t


def lorenz(
    n_steps: int = 10000,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    x0: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Lorenz '63 attractor, integrated with RK4. Returns (3, n_steps) and t."""
    def rhs(state):
        x, y, z = state
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    out = np.empty((3, n_steps))
    out[:, 0] = x0
    for i in range(n_steps - 1):
        s = out[:, i]
        k1 = rhs(s)
        k2 = rhs(s + 0.5 * dt * k1)
        k3 = rhs(s + 0.5 * dt * k2)
        k4 = rhs(s + dt * k3)
        out[:, i + 1] = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    t = np.arange(n_steps) * dt
    return out, t


def kuramoto_sivashinsky_snapshots(
    nx: int = 128,
    nt: int = 400,
    L: float = 22.0,
    dt: float = 0.25,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """KS equation u_t = -u u_x - u_xx - u_xxxx via exponential-time-differencing.

    Standard chaotic test case from the Kutz/Brunton book.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, L, nx, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(nx, d=L / nx)
    # Linear operator in Fourier space
    Lop = k**2 - k**4
    E = np.exp(dt * Lop)
    E2 = np.exp(dt * Lop / 2)

    # ETDRK4 coefficients via contour integral
    M = 16
    r = np.exp(1j * np.pi * (np.arange(M) + 0.5) / M)
    LR = dt * Lop[:, None] + r[None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = dt * np.real(
        np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1)
    )
    f2 = dt * np.real(
        np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1)
    )
    f3 = dt * np.real(
        np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1)
    )

    u = 0.01 * rng.standard_normal(nx)
    v = np.fft.fft(u)

    snapshots = np.empty((nx, nt))
    g = -0.5j * k

    for n in range(nt):
        snapshots[:, n] = np.real(np.fft.ifft(v))
        Nv = g * np.fft.fft(np.real(np.fft.ifft(v)) ** 2)
        a = E2 * v + Q * Nv
        Na = g * np.fft.fft(np.real(np.fft.ifft(a)) ** 2)
        b = E2 * v + Q * Na
        Nb = g * np.fft.fft(np.real(np.fft.ifft(b)) ** 2)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = g * np.fft.fft(np.real(np.fft.ifft(c)) ** 2)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3

    t = np.arange(nt) * dt
    return snapshots, x, t
