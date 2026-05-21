# dmd-toolkit

Python implementations of the five most-used Dynamic Mode Decomposition variants from Nathan Kutz's group at UW Applied Math, packaged for drop-in use on any snapshot dataset.

## What is DMD?

DMD finds a best-fit linear operator `A` such that `X[:, n+1] ≈ A X[:, n]` for a sequence of snapshots, then extracts its eigendecomposition. The result: a small set of spatial **modes** with associated complex **eigenvalues** that decode into frequencies and growth/decay rates. It's a data-driven Koopman-operator approximation — you get linear-system tools (eigenvalues, modes, forecasting) for nonlinear data, without ever writing down a model.

Originated by Schmid (2010) for fluid dynamics. The variants below are refinements that handle noise, multi-scale dynamics, chaos, and uncertainty.

## Variants implemented

| Class | Best for | Paper |
|---|---|---|
| `ExactDMD` | Clean, evenly-sampled snapshots | Tu, Rowley, Luchtenburg, Brunton, Kutz (2014) |
| `OptimizedDMD` | Noisy data, uneven sampling | Askham & Kutz, SIAM JADS (2018) |
| `BOPDMD` | Uncertainty quantification, forecasting | Sashidhar & Kutz, Phil. Trans. R. Soc. A (2022) |
| `MultiResolutionDMD` | Multi-scale data (slow + fast modes) | Kutz, Fu, Brunton, SIAM JADS (2016) |
| `HAVOK` | Chaotic / intermittently-forced scalar signals | Brunton, Brunton, Proctor, Kaiser, Kutz, Nat. Commun. (2017) |

## Install

```bash
cd ~/Documents/Yazdanlab/dmd-toolkit
python3 -m venv .venv
.venv/bin/pip install -e .
```

### Neural variants (optional)

The five neural methods in `dmd_toolkit.neural` (SHRED, DPK, Discrepancy, SKS,
KoopmanAutoencoder — see `METHODS.md`) require PyTorch. Install with the
`[neural]` extra:

```bash
.venv/bin/pip install -e ".[neural]"
```

**Note on CUDA:** the `[neural]` extra pulls torch from PyPI, which gives you
the **CPU** build by default. For GPU support, install torch separately from
the PyTorch index before (or after) the `[neural]` install — for CUDA 12.8:

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Match the `cuXXX` suffix to your driver's max CUDA version (`nvidia-smi`).
PEP 508 has no way to pin a custom index per-extra, so this remains a manual
step.

## 30-second quickstart

```python
import dmd_toolkit as dmd

# Any (features × time) snapshot matrix
X, x, t = dmd.data.two_mode_oscillator(noise=0.05)

# Exact DMD — recover modes/eigenvalues
ed = dmd.exact_dmd(X, rank=2, dt=0.05)
print(ed.frequencies)        # [1.0, 5.5] Hz — recovered
print(ed.growth_rates)       # decay rates

X_hat = ed.reconstruct(t)    # rebuild snapshots from modes
forecast = ed.predict(50)    # 50 steps into the future

# Plots
dmd.plot.eigenvalue_spectrum(ed.eigenvalues)
dmd.plot.spacetime_heatmap(X, t=t, x=x)
```

Full demo of all five variants at `examples/quickstart.py`. Run it:
```bash
.venv/bin/python examples/quickstart.py
```
Saves `quickstart_output.png` with reconstructions, eigenvalue spectra, mode amplitudes, and HAVOK forcing.

## Input convention

All snapshot-based methods accept:
- `(features, time)` 2D arrays — preferred
- 1D arrays — treated as single-feature time series
- 3D+ arrays — non-time axes flattened (last axis must be time)

Convert anything explicitly with `dmd.utils.to_snapshots(data)`.

### Real-valued data note

DMD assumes complex-exponential modes. For real-valued data with oscillations, you have three options:
1. **Hilbert transform**: `from scipy.signal import hilbert; X_complex = hilbert(X, axis=-1)`
2. **Hankel embedding** via `HAVOK` (for scalar signals)
3. Accept that real eigenvalues will appear (they correspond to cosines, not exponentials)

## Common kwargs

| Param | Methods | Meaning |
|---|---|---|
| `rank` | all | int = fixed rank, float ∈ (0,1) = energy fraction, `None` = full SVD |
| `dt` | ExactDMD, mrDMD, HAVOK | sample spacing in time (seconds) |
| `t` | OptimizedDMD, BOPDMD | explicit time array (allows uneven sampling) |

## API reference

### `ExactDMD(rank, dt)` — `dmd.exact_dmd(X, rank, dt)`

The canonical algorithm (Tu et al. 2014). Fast, deterministic, sensitive to noise.

```python
ed = dmd.exact_dmd(X, rank=2, dt=0.05)
ed.modes          # (n_features, rank) complex — spatial structures
ed.eigenvalues    # (rank,) complex — discrete-time
ed.omega          # (rank,) complex — continuous-time (log(λ)/dt)
ed.frequencies    # (rank,) real — Hz, = Im(omega)/(2π)
ed.growth_rates   # (rank,) real — Re(omega)
ed.amplitudes     # (rank,) complex — initial-condition projection
ed.reconstruct(t) # (n_features, len(t)) complex
ed.predict(n)     # forecast n steps from end of training data
```

### `OptimizedDMD(rank, init, max_iter, tol)` — `dmd.optimized_dmd(X, rank, t)`

Variable-projection nonlinear least squares (Askham & Kutz 2018). Substantially less bias than exact DMD when data is noisy, and handles arbitrary time samples.

```python
od = dmd.optimized_dmd(X, rank=2, t=t)
od.omega          # optimized continuous-time eigenvalues
od.modes
od.amplitudes
od.converged      # bool — did the solver succeed
od.reconstruct(t)
```

`init` can be `"exact"` (warm-start from exact DMD, default) or an array of initial omega guesses.

### `BOPDMD(rank, n_trials, subsample_fraction, rng_seed)` — `dmd.bop_dmd(X, rank, t)`

Bagging + optimized DMD (Sashidhar & Kutz 2022). Fits `n_trials` optimized DMDs on random subsets, averages eigenvalues, reports per-mode std for uncertainty quantification.

```python
bop = dmd.bop_dmd(X, rank=2, t=t, n_trials=32, subsample_fraction=0.6)
bop.omega_mean             # ensemble mean — best estimate
bop.omega_std              # temporal uncertainty per mode
bop.modes_mean
bop.amplitudes_mean
bop.reconstruct(t)
bop.forecast_ensemble(t)   # (n_trials, n_features, len(t)) — for uncertainty bands
bop.trials                 # raw per-trial (omega, modes, amps) tuples
```

### `MultiResolutionDMD(max_levels, rank, slow_cutoff, dt)` — `dmd.mr_dmd(X, max_levels, rank, dt)`

Recursive separation by time-scale (Kutz, Fu, Brunton 2016). Best when data has widely separated time-scales (slow background + fast events). At each level, fits DMD on a window, retains "slow" modes (modes with ≤ `slow_cutoff` cycles in the current window), subtracts their reconstruction, and recurses on left/right halves.

```python
mr = dmd.mr_dmd(X, max_levels=6, rank=2, dt=0.02, slow_cutoff=5.0)
mr.levels        # list of dicts: {level, t_offset, modes, omega, amplitudes, omega_cutoff}
mr.reconstruct(n_times)
```

### `HAVOK(delays, rank, dt)` — `dmd.havok(x, delays, rank, dt)`

Hankel Alternative View Of Koopman (Brunton et al. 2017). For chaotic scalar signals (e.g., Lorenz). Builds a Hankel matrix of time-delayed copies, takes SVD, fits linear dynamics on the leading `rank-1` SVD modes, and treats the last mode as intermittent forcing — which spikes correspond to lobe-switching events in chaotic attractors.

```python
hav = dmd.havok(lorenz_x, delays=100, rank=15, dt=0.01)
hav.A              # (rank-1, rank-1) linear state matrix
hav.B              # (rank-1, 1) forcing input matrix
hav.forcing        # (T_eff,) — v_r(t) intermittent forcing signal
hav.embedding      # (T_eff, rank-1) low-dim attractor coordinates
hav.forcing_threshold(quantile=0.95)
hav.forcing_events(quantile=0.95)   # indices where |forcing| spikes
```

## Synthetic datasets

```python
# Two complex-exponential modes, sech spatial profiles — clean DMD test
X, x, t = dmd.data.two_mode_oscillator(nx=200, nt=400, dt=0.05, noise=0.05)

# Slow background + fast localized oscillation — designed for mrDMD
X, x, t = dmd.data.multi_scale_signal(slow_freq=0.3, fast_freq=6.0)

# Lorenz '63 chaotic attractor (RK4 integration) — for HAVOK
X, t = dmd.data.lorenz(n_steps=10000, dt=0.01)

# Kuramoto–Sivashinsky chaotic PDE (ETDRK4 spectral solver) — heavy chaos
X, x, t = dmd.data.kuramoto_sivashinsky_snapshots(nx=128, nt=400)
```

## Plotting helpers

All return `(fig, ax)` so you can tweak further with matplotlib:

```python
dmd.plot.eigenvalue_spectrum(ed.eigenvalues)           # unit circle + λ scatter
dmd.plot.mode_amplitudes(ed.amplitudes, ed.frequencies) # stem plot of |b| vs Hz
dmd.plot.reconstruction_vs_truth(X, X_hat, t=t)        # single-feature time series overlay
dmd.plot.spacetime_heatmap(X, t=t, x=x)                # pcolormesh
dmd.plot.forcing_signal(t, hav.forcing, threshold)     # HAVOK forcing with event threshold
```

## Project structure

```
dmd-toolkit/
├── README.md
├── pyproject.toml
├── requirements.txt           # numpy, scipy, matplotlib
├── dmd_toolkit/
│   ├── __init__.py
│   ├── exact.py              # ExactDMD
│   ├── optimized.py          # OptimizedDMD
│   ├── bop.py                # BOPDMD
│   ├── multires.py           # MultiResolutionDMD
│   ├── havok.py              # HAVOK
│   ├── data.py               # synthetic datasets
│   ├── utils.py              # to_snapshots, truncated_svd, hankel, reconstruct
│   └── plot.py               # matplotlib helpers
├── examples/
│   └── quickstart.py         # runnable end-to-end demo
└── tests/
    └── test_exact.py         # pytest smoke tests
```

## Tests

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests/ -v
```

All eight tests pass: ExactDMD reconstruction error, frequency recovery, rank-by-energy-fraction, mrDMD smoke, HAVOK smoke, Hankel shape, snapshot coercion, SVD truncation.

## References

- Schmid (2010) — *Dynamic mode decomposition of numerical and experimental data*. JFM 656, 5–28. *(Original DMD)*
- Tu, Rowley, Luchtenburg, Brunton, Kutz (2014) — *On dynamic mode decomposition: theory and applications*. J. Comp. Dyn. 1, 391–421. *(Exact DMD)*
- Kutz, Fu, Brunton (2016) — *Multiresolution dynamic mode decomposition*. SIAM JADS 15, 713–735.
- Kutz, Brunton, Brunton, Proctor (2016) — *Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems*. SIAM. *(The textbook.)*
- Brunton, Brunton, Proctor, Kaiser, Kutz (2017) — *Chaos as an intermittently forced linear system*. Nature Communications 8, 19. *(HAVOK)*
- Askham & Kutz (2018) — *Variable projection methods for an optimized dynamic mode decomposition*. SIAM JADS 17, 380–416. *(Optimized DMD)*
- Sashidhar & Kutz (2022) — *Bagging, optimized dynamic mode decomposition (BOP-DMD) for robust, stable forecasting with spatial and temporal uncertainty quantification*. Phil. Trans. R. Soc. A 380, 20210199.

## Future work (not implemented)

- **Extended DMD (EDMD)** with nonlinear observables / dictionary learning
- **DMD with control (DMDc)** for input–output systems
- **Compressed DMD** for high-dim subsampled data
- **Online / streaming DMD** for real-time updates
- **Sparsity-promoting DMD (spDMD)** — selects dominant modes via ℓ1-penalized fit

PyDMD (https://github.com/PyDMD/PyDMD) has many of these if you need them off-the-shelf.
