"""Quickstart — run all four Kutz-canon DMD variants on synthetic data.

Usage:
    cd ~/Documents/Yazdanlab/dmd-toolkit
    pip install -e .
    python examples/quickstart.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import dmd_toolkit as dmd

# ──────────────────────────────────────────────────
# 1. Synthetic dataset: two oscillating modes
# ──────────────────────────────────────────────────
X, x, t = dmd.data.two_mode_oscillator(nx=200, nt=400, dt=0.05, noise=0.05)
print(f"Snapshot matrix: {X.shape}  (features × time)")

# ──────────────────────────────────────────────────
# 2. Exact DMD
# ──────────────────────────────────────────────────
ed = dmd.exact_dmd(X, rank=2, dt=0.05)  # data is rank-2 (two complex-exp modes)
X_ed = ed.reconstruct(t)
err_ed = np.linalg.norm(X - X_ed, "fro") / np.linalg.norm(X, "fro")
print(f"Exact DMD relative error : {err_ed:.4f}")
print(f"  Frequencies (Hz)       : {ed.frequencies.round(3)}")

# ──────────────────────────────────────────────────
# 3. Optimized DMD
# ──────────────────────────────────────────────────
od = dmd.optimized_dmd(X, rank=2, t=t)
X_od = od.reconstruct(t)
err_od = np.linalg.norm(X - X_od, "fro") / np.linalg.norm(X, "fro")
print(f"Optimized DMD rel. error : {err_od:.4f}  (converged={od.converged})")

# ──────────────────────────────────────────────────
# 4. Multi-resolution DMD on multi-scale data
# ──────────────────────────────────────────────────
X_ms, _, t_ms = dmd.data.multi_scale_signal(slow_freq=0.3, fast_freq=6.0)
mr = dmd.mr_dmd(X_ms, max_levels=6, rank=2, dt=0.02)
X_mr = mr.reconstruct(X_ms.shape[1])
err_mr = np.linalg.norm(X_ms - X_mr, "fro") / np.linalg.norm(X_ms, "fro")
by_depth = {}
for L in mr.levels:
    if L["modes"].size:
        f = np.abs(L["omega"].imag).max() / (2 * np.pi)
        by_depth.setdefault(L["level"], []).append(f)
print(f"mrDMD relative error     : {err_mr:.4f}  ({len(mr.levels)} chunks)")
for depth in sorted(by_depth):
    fs = by_depth[depth]
    print(f"  depth {depth}: captured up to {max(fs):.2f} Hz across {len(fs)} chunks")

# ──────────────────────────────────────────────────
# 5. HAVOK on Lorenz x-coordinate
# ──────────────────────────────────────────────────
lorenz_data, t_lorenz = dmd.data.lorenz(n_steps=8000, dt=0.01)
hav = dmd.havok(lorenz_data[0], delays=100, rank=15, dt=0.01)
thresh = hav.forcing_threshold(0.95)
events = hav.forcing_events(0.95)
print(f"HAVOK: {len(events)} intermittent forcing events above {thresh:.4f}")

# ──────────────────────────────────────────────────
# 6. Quick plots
# ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

dmd.plot.spacetime_heatmap(X, t=t, title="Truth", ax=axes[0, 0])
dmd.plot.spacetime_heatmap(X_ed, t=t, title="Exact DMD recon", ax=axes[0, 1])
dmd.plot.spacetime_heatmap(X_od, t=t, title="Opt DMD recon", ax=axes[0, 2])

dmd.plot.eigenvalue_spectrum(ed.eigenvalues, title="ExactDMD eigenvalues", ax=axes[1, 0])
dmd.plot.mode_amplitudes(
    ed.amplitudes, frequencies=ed.frequencies, title="Mode amplitudes", ax=axes[1, 1]
)
dmd.plot.forcing_signal(
    t_lorenz[: len(hav.forcing)],
    hav.forcing,
    threshold=thresh,
    title="HAVOK forcing (Lorenz)",
    ax=axes[1, 2],
)

plt.tight_layout()
plt.savefig("quickstart_output.png", dpi=120)
print("Saved quickstart_output.png")
