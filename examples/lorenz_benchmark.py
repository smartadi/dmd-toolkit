"""Benchmark all DMD variants on Lorenz '63 chaotic data.

Lorenz is genuinely chaotic — no linear operator can capture it perfectly.
Standard DMD methods will produce structured but inaccurate reconstructions.
HAVOK is the right tool for this dataset (Brunton et al. 2017).

Metrics reported per method:
  - Fit time (seconds)
  - Reconstruction relative Frobenius error on training window
  - One-step-ahead RMSE on test window (where applicable)
  - Free-running forecast relative error over test window
  - Dominant frequencies recovered
"""

from __future__ import annotations

import time
import sys
import pathlib

import numpy as np
from scipy.signal import hilbert
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import dmd_toolkit as dmd


# ──────────────────────────────────────────────────
# 1. Generate Lorenz '63 data
# ──────────────────────────────────────────────────
N_TOTAL = 5000
N_TRAIN = 4000
DT = 0.01

X_real, t = dmd.data.lorenz(n_steps=N_TOTAL, dt=DT)
print(f"Lorenz data shape: {X_real.shape}  (x, y, z over {N_TOTAL} steps)")

# DMD assumes complex-exponential modes. Hilbert transform gives the analytic
# signal so DMD on each spatial component is meaningful.
X = hilbert(X_real, axis=-1)

X_train = X[:, :N_TRAIN]
X_test = X[:, N_TRAIN:]
t_train = t[:N_TRAIN]
t_test = t[N_TRAIN:]
N_TEST = X_test.shape[1]


def rel_err(true: np.ndarray, est: np.ndarray) -> float:
    return float(np.linalg.norm(true - est, "fro") / np.linalg.norm(true, "fro"))


results: dict[str, dict] = {}


# ──────────────────────────────────────────────────
# 2. Exact DMD
# ──────────────────────────────────────────────────
t0 = time.perf_counter()
ed = dmd.exact_dmd(X_train, rank=3, dt=DT)
fit_time = time.perf_counter() - t0

X_recon = ed.reconstruct(t_train)
recon_err = rel_err(X_train, X_recon)
X_pred = ed.predict(N_TEST)
pred_err = rel_err(X_test, X_pred)

results["ExactDMD"] = {
    "fit_time": fit_time,
    "recon_err": recon_err,
    "pred_err": pred_err,
    "freqs": ed.frequencies,
    "growth": ed.growth_rates,
}


# ──────────────────────────────────────────────────
# 3. Optimized DMD
# ──────────────────────────────────────────────────
t0 = time.perf_counter()
od = dmd.optimized_dmd(X_train, rank=3, t=t_train)
fit_time = time.perf_counter() - t0

X_recon = od.reconstruct(t_train)
recon_err = rel_err(X_train, X_recon)
X_pred = od.reconstruct(t_test)
pred_err = rel_err(X_test, X_pred)

results["OptimizedDMD"] = {
    "fit_time": fit_time,
    "recon_err": recon_err,
    "pred_err": pred_err,
    "freqs": od.frequencies,
    "growth": od.growth_rates,
    "converged": od.converged,
}


# ──────────────────────────────────────────────────
# 4. BOP-DMD
# ──────────────────────────────────────────────────
t0 = time.perf_counter()
bop = dmd.bop_dmd(X_train, rank=3, t=t_train, n_trials=16, rng_seed=0)
fit_time = time.perf_counter() - t0

X_recon = bop.reconstruct(t_train)
recon_err = rel_err(X_train, X_recon)
X_pred = bop.reconstruct(t_test)
pred_err = rel_err(X_test, X_pred)

results["BOPDMD"] = {
    "fit_time": fit_time,
    "recon_err": recon_err,
    "pred_err": pred_err,
    "freqs": np.imag(bop.omega_mean) / (2 * np.pi),
    "growth": np.real(bop.omega_mean),
    "freq_std": np.imag(bop.omega_std) / (2 * np.pi),
    "n_trials": len(bop.trials),
}


# ──────────────────────────────────────────────────
# 5. Multi-Resolution DMD
# ──────────────────────────────────────────────────
t0 = time.perf_counter()
mr = dmd.mr_dmd(X_train, max_levels=6, rank=3, dt=DT, slow_cutoff=5.0)
fit_time = time.perf_counter() - t0

X_recon = mr.reconstruct(N_TRAIN)
recon_err = rel_err(X_train, X_recon)
# mrDMD doesn't naturally extend past the training window; skip pred

results["mrDMD"] = {
    "fit_time": fit_time,
    "recon_err": recon_err,
    "pred_err": float("nan"),
    "n_chunks": len(mr.levels),
    "n_modes_captured": sum(L["modes"].shape[1] for L in mr.levels),
}


# ──────────────────────────────────────────────────
# 6. HAVOK — on Lorenz x-coordinate only
# ──────────────────────────────────────────────────
t0 = time.perf_counter()
hav = dmd.havok(X_real[0, :N_TRAIN], delays=100, rank=15, dt=DT)
fit_time = time.perf_counter() - t0

# HAVOK fit-quality metric: how well does dv/dt = A v + B v_r fit?
state = hav.embedding
forcing = hav.forcing[:, None]
dstate = np.gradient(state, DT, axis=0)
predicted = state @ hav.A.T + forcing @ hav.B.T
linear_resid = rel_err(dstate, predicted)
events = hav.forcing_events(0.95)

results["HAVOK"] = {
    "fit_time": fit_time,
    "recon_err": linear_resid,  # residual of linear model fit
    "pred_err": float("nan"),
    "n_forcing_events": len(events),
    "forcing_threshold": hav.forcing_threshold(0.95),
    "rank": hav.rank,
}


# ──────────────────────────────────────────────────
# 7. Print results table
# ──────────────────────────────────────────────────
print("\n" + "=" * 76)
print(f"{'Method':<14} {'Fit (s)':>9} {'Recon err':>11} {'Pred err':>11}  Notes")
print("-" * 76)
for name, r in results.items():
    fit = f"{r['fit_time']:.3f}"
    recon = f"{r['recon_err']:.3f}"
    pred = "—" if np.isnan(r["pred_err"]) else f"{r['pred_err']:.3f}"
    notes = []
    if "freqs" in r:
        top_freqs = ", ".join(f"{f:+.2f}Hz" for f in r["freqs"][:3])
        notes.append(top_freqs)
    if "converged" in r:
        notes.append(f"converged={r['converged']}")
    if "n_chunks" in r:
        notes.append(f"{r['n_chunks']} chunks, {r['n_modes_captured']} modes")
    if "n_forcing_events" in r:
        notes.append(f"{r['n_forcing_events']} forcing events @ thresh={r['forcing_threshold']:.3f}")
    print(f"{name:<14} {fit:>9} {recon:>11} {pred:>11}  {'; '.join(notes)}")
print("=" * 76)


# ──────────────────────────────────────────────────
# 8. Visualisation
# ──────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))

# Top row: Lorenz attractor + x(t) train/test split
ax1 = fig.add_subplot(2, 3, 1, projection="3d")
ax1.plot(X_real[0], X_real[1], X_real[2], lw=0.4, alpha=0.6)
ax1.set_title("Lorenz '63 attractor")
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")

ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(t, X_real[0], lw=0.5, label="x(t)")
ax2.axvline(t[N_TRAIN], color="r", ls="--", lw=1, label="train/test")
ax2.set_xlabel("t"); ax2.set_ylabel("x"); ax2.legend()
ax2.set_title("x(t) with train/test split")

# ExactDMD reconstruction vs truth
ax3 = fig.add_subplot(2, 3, 3)
X_ed_recon = ed.reconstruct(t_train).real
ax3.plot(t_train, X_real[0, :N_TRAIN], lw=0.6, label="truth")
ax3.plot(t_train, X_ed_recon[0], "--", lw=0.6, label="ExactDMD")
ax3.set_xlabel("t"); ax3.set_ylabel("x"); ax3.legend()
ax3.set_title(f"ExactDMD reconstruction (rel err {results['ExactDMD']['recon_err']:.2f})")

# Eigenvalue spectra
ax4 = fig.add_subplot(2, 3, 4)
theta = np.linspace(0, 2 * np.pi, 200)
ax4.plot(np.cos(theta), np.sin(theta), "k--", lw=0.5, alpha=0.4)
ax4.scatter(ed.eigenvalues.real, ed.eigenvalues.imag, label="ExactDMD", s=80, marker="o")
ax4.scatter(np.exp(od.omega * DT).real, np.exp(od.omega * DT).imag, label="OptDMD", s=60, marker="s")
ax4.set_aspect("equal")
ax4.set_xlabel("Re"); ax4.set_ylabel("Im")
ax4.set_title("Eigenvalue spectra")
ax4.legend()

# mrDMD: number of modes per recursion depth
ax5 = fig.add_subplot(2, 3, 5)
by_depth: dict[int, int] = {}
for L in mr.levels:
    by_depth[L["level"]] = by_depth.get(L["level"], 0) + L["modes"].shape[1]
depths = sorted(by_depth)
ax5.bar(depths, [by_depth[d] for d in depths], color="steelblue")
ax5.set_xlabel("Recursion depth"); ax5.set_ylabel("# modes captured")
ax5.set_title(f"mrDMD: {results['mrDMD']['n_modes_captured']} modes across {results['mrDMD']['n_chunks']} chunks")

# HAVOK forcing
ax6 = fig.add_subplot(2, 3, 6)
t_hav = t_train[: len(hav.forcing)]
ax6.plot(t_hav, hav.forcing, lw=0.5, color="steelblue")
thresh = hav.forcing_threshold(0.95)
ax6.axhline(thresh, color="r", ls="--", lw=1)
ax6.axhline(-thresh, color="r", ls="--", lw=1)
ax6.set_xlabel("t"); ax6.set_ylabel("v_r(t)")
ax6.set_title(f"HAVOK forcing: {results['HAVOK']['n_forcing_events']} events")

plt.tight_layout()
out_path = pathlib.Path(__file__).parent / "lorenz_benchmark.png"
plt.savefig(out_path, dpi=120)
print(f"\nSaved {out_path.name}")
