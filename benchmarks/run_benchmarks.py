"""Benchmark all 10 DMD variants across synthetic datasets and noise levels.

Run:
    .venv/bin/python benchmarks/run_benchmarks.py
    .venv/bin/python benchmarks/run_benchmarks.py --epochs 200 --out results.csv

Writes a tidy CSV (one row per method × dataset × noise level) with:
    method, dataset, noise, fit_time_s, recon_rmse, forecast_rmse,
    n_params, notes

Notes
-----
- recon_rmse: RMSE of model output on the training portion (first 80% of t).
- forecast_rmse: RMSE on the held-out tail (last 20% of t), where the method
  supports free-running prediction. NaN otherwise.
- All RMSEs are computed on the *real part* of the data, normalised by the
  feature-wise std of the training portion, so values are comparable across
  datasets of very different scale.
- n_params: rank for classical methods, total trainable params for neural.
- HAVOK is scalar-only; runs on the Lorenz x-coordinate.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import dmd_toolkit as dmd
from dmd_toolkit.neural import (
    DiscrepancyModel,
    fit_dpk,
    fit_koopman_ae,
    fit_shred,
    fit_sks,
    predict_shred,
)

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    name: str
    X: np.ndarray           # native dtype (complex for synthetic oscillators, real for sims)
    X_real: np.ndarray      # always real-valued — for neural methods
    t: np.ndarray
    dt: float
    rank: int
    scalar: np.ndarray | None = None  # for HAVOK


def make_datasets(noise: float, seed: int) -> dict[str, Dataset]:
    """Datasets ship complex-where-natural plus a real view.

    Classical DMD is happiest on complex-exponential snapshots (its native
    ansatz). Neural methods need real float32 input. Each Dataset carries
    both: `X` is complex for the oscillator datasets, real for the simulators;
    `X_real` is always real.
    """
    out: dict[str, Dataset] = {}

    Xc, _, t = dmd.data.two_mode_oscillator(
        nx=120, nt=300, dt=0.05, noise=noise, seed=seed
    )
    out["two_mode"] = Dataset("two_mode", Xc, Xc.real, t, 0.05, rank=4)

    Xc, _, t = dmd.data.multi_scale_signal(
        nx=80, nt=300, dt=0.02, noise=noise, seed=seed
    )
    out["multi_scale"] = Dataset("multi_scale", Xc, Xc.real, t, 0.02, rank=4)

    X, t = dmd.data.lorenz(n_steps=2000, dt=0.01)
    if noise > 0:
        rng = np.random.default_rng(seed)
        X = X + noise * X.std() * rng.standard_normal(X.shape)
    out["lorenz"] = Dataset("lorenz", X, X, t, 0.01, rank=3, scalar=X[0])

    X, _, t = dmd.data.kuramoto_sivashinsky_snapshots(nx=64, nt=300)
    if noise > 0:
        rng = np.random.default_rng(seed)
        X = X + noise * X.std() * rng.standard_normal(X.shape)
    out["ks"] = Dataset("ks", X, X, t, float(t[1] - t[0]), rank=8)

    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def normalised_rmse(true: np.ndarray, pred: np.ndarray, scale: float) -> float:
    """RMSE on real parts, divided by `scale` (training-portion std)."""
    a = np.real(true)
    b = np.real(pred)
    return float(np.sqrt(np.mean((a - b) ** 2)) / max(scale, 1e-12))


def split_time(t: np.ndarray, X: np.ndarray, train_frac: float = 0.8):
    n = X.shape[-1]
    n_train = int(n * train_frac)
    t_tr, t_te = t[:n_train], t[n_train:]
    if X.ndim == 1:
        X_tr, X_te = X[:n_train], X[n_train:]
    else:
        X_tr, X_te = X[:, :n_train], X[:, n_train:]
    return X_tr, X_te, t_tr, t_te


def torch_n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------
def run_exact(ds: Dataset, X_tr, X_te, t_tr, t_te, scale):
    t0 = time.perf_counter()
    model = dmd.exact_dmd(X_tr, rank=ds.rank, dt=ds.dt)
    fit_s = time.perf_counter() - t0
    # reconstruct/forecast in absolute time relative to t_tr[0] (model evolves from t=0)
    recon = model.reconstruct(t_tr - t_tr[0])
    fc = model.reconstruct(t_te - t_tr[0])
    return dict(
        fit_time_s=fit_s,
        recon_rmse=normalised_rmse(X_tr, recon, scale),
        forecast_rmse=normalised_rmse(X_te, fc, scale),
        n_params=ds.rank,
        notes="",
    )


def run_optimized(ds: Dataset, X_tr, X_te, t_tr, t_te, scale):
    t0 = time.perf_counter()
    try:
        model = dmd.optimized_dmd(X_tr, rank=ds.rank, t=t_tr)
        fit_s = time.perf_counter() - t0
        recon = model.reconstruct(t_tr)
        fc = model.reconstruct(t_te)
        return dict(
            fit_time_s=fit_s,
            recon_rmse=normalised_rmse(X_tr, recon, scale),
            forecast_rmse=normalised_rmse(X_te, fc, scale),
            n_params=ds.rank,
            notes="" if model.converged else "did_not_converge",
        )
    except Exception as e:
        return dict(
            fit_time_s=time.perf_counter() - t0,
            recon_rmse=math.nan,
            forecast_rmse=math.nan,
            n_params=ds.rank,
            notes=f"error:{type(e).__name__}",
        )


def run_bop(ds: Dataset, X_tr, X_te, t_tr, t_te, scale):
    t0 = time.perf_counter()
    try:
        model = dmd.bop_dmd(X_tr, rank=ds.rank, t=t_tr, n_trials=16, rng_seed=0)
        fit_s = time.perf_counter() - t0
        recon = model.reconstruct(t_tr)
        fc = model.reconstruct(t_te)
        omega_unc = float(np.mean(np.abs(model.omega_std)))
        return dict(
            fit_time_s=fit_s,
            recon_rmse=normalised_rmse(X_tr, recon, scale),
            forecast_rmse=normalised_rmse(X_te, fc, scale),
            n_params=ds.rank,
            notes=f"omega_std_mean={omega_unc:.3g}",
        )
    except Exception as e:
        return dict(
            fit_time_s=time.perf_counter() - t0,
            recon_rmse=math.nan,
            forecast_rmse=math.nan,
            n_params=ds.rank,
            notes=f"error:{type(e).__name__}",
        )


def run_mrdmd(ds: Dataset, X_tr, X_te, t_tr, t_te, scale):
    t0 = time.perf_counter()
    model = dmd.mr_dmd(X_tr, max_levels=4, rank=ds.rank, dt=ds.dt, slow_cutoff=5.0)
    fit_s = time.perf_counter() - t0
    recon = model.reconstruct(X_tr.shape[1])
    # no native forecast; extrapolate slow modes manually onto t_te grid
    n_modes = sum(L["modes"].shape[1] for L in model.levels)
    return dict(
        fit_time_s=fit_s,
        recon_rmse=normalised_rmse(X_tr, recon, scale),
        forecast_rmse=math.nan,
        n_params=n_modes,
        notes=f"n_levels={len(model.levels)}",
    )


def run_havok(ds: Dataset, X_tr, X_te, t_tr, t_te, scale):
    """HAVOK is scalar-only; runs on ds.scalar (e.g. Lorenz x)."""
    if ds.scalar is None:
        return None
    x = ds.scalar
    n = len(x)
    n_train = int(n * 0.8)
    x_tr = x[:n_train]
    t0 = time.perf_counter()
    model = dmd.havok(x_tr, delays=80, rank=15, dt=ds.dt)
    fit_s = time.perf_counter() - t0
    # Recon metric: one-step linear-ODE prediction error in embedding space
    dV = np.gradient(model.embedding, ds.dt, axis=0)
    pred = model.embedding @ model.A.T + model.forcing[:, None] @ model.B.T
    emb_scale = float(model.embedding.std() + 1e-12)
    embed_rmse = float(np.sqrt(np.mean((dV - pred) ** 2)) / emb_scale)
    return dict(
        fit_time_s=fit_s,
        recon_rmse=embed_rmse,
        forecast_rmse=math.nan,
        n_params=(model.A.size + model.B.size),
        notes="scalar; recon_rmse=embedding-ODE residual",
    )


def run_shred(ds: Dataset, X_tr, X_te, t_tr, t_te, scale, epochs):
    if ds.X.shape[0] < 4:
        return dict(  # too few features to sample sensors
            fit_time_s=math.nan, recon_rmse=math.nan, forecast_rmse=math.nan,
            n_params=0, notes="skipped:too_few_features",
        )
    n_feat = ds.X.shape[0]
    sensor_idx = np.linspace(0, n_feat - 1, min(8, n_feat // 2), dtype=int).tolist()
    lags = min(30, X_tr.shape[1] // 4)
    t0 = time.perf_counter()
    model, info = fit_shred(
        X_tr, sensor_indices=sensor_idx, lags=lags, epochs=epochs,
        device=DEVICE, rng_seed=0,
    )
    fit_s = time.perf_counter() - t0
    # Recon error on training: decode each training sensor window back to state
    from dmd_toolkit.neural.shred import make_sensor_lag_dataset
    inputs, targets = make_sensor_lag_dataset(X_tr, sensor_idx, lags)
    inputs_n = (inputs - info.sensor_mean) / info.sensor_std
    import torch as _t
    with _t.no_grad():
        preds = model(_t.from_numpy(inputs_n).to(DEVICE)).cpu().numpy()
    preds = preds * info.x_std + info.x_mean
    train_rmse = normalised_rmse(targets, preds, scale)
    # "Forecast" for SHRED = reconstruction on test sensor windows
    test_rmse = math.nan
    if X_te.shape[1] > lags:
        # concatenate tail of train to give first lags samples for first test target
        bridge = np.concatenate([X_tr[:, -(lags - 1):], X_te], axis=1)
        inputs_te, targets_te = make_sensor_lag_dataset(bridge, sensor_idx, lags)
        inputs_te_n = (inputs_te - info.sensor_mean) / info.sensor_std
        with _t.no_grad():
            preds_te = model(_t.from_numpy(inputs_te_n).to(DEVICE)).cpu().numpy()
        preds_te = preds_te * info.x_std + info.x_mean
        test_rmse = normalised_rmse(targets_te, preds_te, scale)
    return dict(
        fit_time_s=fit_s,
        recon_rmse=train_rmse,
        forecast_rmse=test_rmse,
        n_params=torch_n_params(model),
        notes=f"sensors={len(sensor_idx)},lags={lags}",
    )


def run_dpk(ds: Dataset, X_tr, X_te, t_tr, t_te, scale, epochs):
    latent = min(8, max(2, ds.X.shape[0]))
    t0 = time.perf_counter()
    model, info = fit_dpk(
        X_tr, latent_dim=latent, horizon=8, epochs=epochs,
        device=DEVICE, rng_seed=0,
    )
    fit_s = time.perf_counter() - t0
    # Recon: encode-decode each training column
    import torch as _t
    Xn = (X_tr - info.x_mean[:, None]) / info.x_std[:, None]
    with _t.no_grad():
        z = model.encode(_t.from_numpy(Xn.T.astype(np.float32)).to(DEVICE))
        mean, _ = model.decode(z)
        mean_np = mean.cpu().numpy() * info.x_std[None, :] + info.x_mean[None, :]
    recon = mean_np.T
    train_rmse = normalised_rmse(X_tr, recon, scale)
    # Forecast: roll forward from last training snapshot
    mean_fc, _ = model.forecast(X_tr[:, -1], n_steps=X_te.shape[1] - 1, device=DEVICE)
    fc = mean_fc.T  # (n_features, n_steps+1)
    if fc.shape[1] > X_te.shape[1]:
        fc = fc[:, : X_te.shape[1]]
    test_rmse = normalised_rmse(X_te[:, : fc.shape[1]], fc, scale)
    return dict(
        fit_time_s=fit_s,
        recon_rmse=train_rmse,
        forecast_rmse=test_rmse,
        n_params=torch_n_params(model),
        notes=f"latent_dim={latent}",
    )


def run_discrepancy(ds: Dataset, X_tr, X_te, t_tr, t_te, scale, epochs):
    try:
        base = dmd.exact_dmd(X_tr, rank=ds.rank, dt=ds.dt)
    except Exception as e:
        return dict(
            fit_time_s=math.nan, recon_rmse=math.nan, forecast_rmse=math.nan,
            n_params=0, notes=f"base_failed:{type(e).__name__}",
        )
    t0 = time.perf_counter()
    disc = DiscrepancyModel(
        base=base, epochs=max(50, epochs * 2), device=DEVICE, rng_seed=0,
    ).fit(X_tr, t_tr)
    fit_s = time.perf_counter() - t0
    recon = disc.reconstruct(t_tr)
    fc = disc.reconstruct(t_te)
    return dict(
        fit_time_s=fit_s,
        recon_rmse=normalised_rmse(X_tr, recon, scale),
        forecast_rmse=normalised_rmse(X_te, fc, scale),
        n_params=torch_n_params(disc._net) if disc._net else 0,
        notes=f"base=ExactDMD(rank={ds.rank})",
    )


def run_sks(ds: Dataset, X_tr, X_te, t_tr, t_te, scale, epochs):
    if ds.X.shape[0] < 4:
        return dict(
            fit_time_s=math.nan, recon_rmse=math.nan, forecast_rmse=math.nan,
            n_params=0, notes="skipped:too_few_features",
        )
    n_feat = ds.X.shape[0]
    sensor_idx = np.linspace(0, n_feat - 1, min(8, n_feat // 2), dtype=int).tolist()
    lags = min(30, X_tr.shape[1] // 4)
    latent = 6
    t0 = time.perf_counter()
    sks = fit_sks(
        X_tr, sensor_indices=sensor_idx, lags=lags, latent_dim=latent,
        epochs=epochs, dt=ds.dt, device=DEVICE, rng_seed=0,
    )
    fit_s = time.perf_counter() - t0
    # Recon on training via SHRED decoder
    z_recon = sks.latent_traj
    full = sks.decode_latent(z_recon)  # (n_features, n_samples)
    # targets are X[:, lags-1:] from make_sensor_lag_dataset
    n_samples = full.shape[1]
    targets = X_tr[:, lags - 1 : lags - 1 + n_samples]
    train_rmse = normalised_rmse(targets, full, scale)
    # Forecast: roll Koopman from last latent for X_te.shape[1] steps
    z_fc = sks.forecast_latent(n_steps=X_te.shape[1] - 1, mode="koopman")
    fc = sks.decode_latent(z_fc)
    if fc.shape[1] > X_te.shape[1]:
        fc = fc[:, : X_te.shape[1]]
    test_rmse = normalised_rmse(X_te[:, : fc.shape[1]], fc, scale)
    nonzero = int(np.count_nonzero(sks.sindy_Xi))
    return dict(
        fit_time_s=fit_s,
        recon_rmse=train_rmse,
        forecast_rmse=test_rmse,
        n_params=torch_n_params(sks.shred) + sks.K.size + sks.sindy_Xi.size,
        notes=f"sensors={len(sensor_idx)},lags={lags},latent={latent},sindy_nz={nonzero}",
    )


def run_koopman_ae(ds: Dataset, X_tr, X_te, t_tr, t_te, scale, epochs):
    latent = min(8, max(2, ds.X.shape[0]))
    t0 = time.perf_counter()
    model, info = fit_koopman_ae(
        X_tr, latent_dim=latent, horizon=4, epochs=epochs,
        device=DEVICE, rng_seed=0,
    )
    fit_s = time.perf_counter() - t0
    # Recon: encode-decode training columns
    import torch as _t
    x_mean = info.x_mean[:, None]
    x_std = info.x_std[:, None]
    Xn = (X_tr - x_mean) / x_std
    with _t.no_grad():
        z = model.encode(_t.from_numpy(Xn.T.astype(np.float32)).to(DEVICE))
        recon_n = model.decode(z).cpu().numpy().T
    recon = recon_n * x_std + x_mean
    train_rmse = normalised_rmse(X_tr, recon, scale)
    # Forecast: free-run from last training column
    model.x_mean.copy_(_t.from_numpy(info.x_mean.astype(np.float32)).to(DEVICE))
    model.x_std.copy_(_t.from_numpy(info.x_std.astype(np.float32)).to(DEVICE))
    fc = model.forecast(X_tr[:, -1], n_steps=X_te.shape[1] - 1, device=DEVICE)
    if fc.shape[1] > X_te.shape[1]:
        fc = fc[:, : X_te.shape[1]]
    test_rmse = normalised_rmse(X_te[:, : fc.shape[1]], fc, scale)
    evals = np.abs(model.eigenvalues)
    return dict(
        fit_time_s=fit_s,
        recon_rmse=train_rmse,
        forecast_rmse=test_rmse,
        n_params=torch_n_params(model),
        notes=f"latent_dim={latent},|eig|_max={float(evals.max()):.3g}",
    )


METHODS = [
    ("ExactDMD", run_exact, False),
    ("OptimizedDMD", run_optimized, False),
    ("BOPDMD", run_bop, False),
    ("MultiResDMD", run_mrdmd, False),
    ("HAVOK", run_havok, False),
    ("SHRED", run_shred, True),
    ("DPK", run_dpk, True),
    ("Discrepancy", run_discrepancy, True),
    ("SKS", run_sks, True),
    ("KoopmanAE", run_koopman_ae, True),
]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Epochs for neural methods (default 50; use 200+ for paper-grade).",
    )
    parser.add_argument(
        "--noise", type=float, nargs="+", default=[0.0, 0.05, 0.15],
        help="Noise levels to sweep (default: 0.0 0.05 0.15).",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent / "results.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["two_mode", "multi_scale", "lorenz", "ks"],
        help="Subset of datasets to run.",
    )
    args = parser.parse_args()

    print(f"Device: {DEVICE}  | epochs={args.epochs}  | noise={args.noise}")
    print(f"Datasets: {args.datasets}\n")

    rows = []
    total = len(METHODS) * len(args.datasets) * len(args.noise)
    counter = 0

    for noise in args.noise:
        datasets = make_datasets(noise=noise, seed=0)
        for ds_name in args.datasets:
            if ds_name not in datasets:
                continue
            ds = datasets[ds_name]
            # Classical: use ds.X (complex where natural). Neural: use ds.X_real.
            X_tr_c, X_te_c, t_tr, t_te = split_time(ds.t, ds.X)
            X_tr_r, X_te_r, _, _ = split_time(ds.t, ds.X_real)
            scale_c = float(np.real(X_tr_c).std()) + 1e-12
            scale_r = float(X_tr_r.std()) + 1e-12

            for method_name, runner, is_neural in METHODS:
                counter += 1
                print(
                    f"[{counter:3d}/{total}] {method_name:14s} on {ds_name:12s} "
                    f"noise={noise:.2f} ...",
                    end="", flush=True,
                )
                try:
                    if is_neural:
                        out = runner(ds, X_tr_r, X_te_r, t_tr, t_te, scale_r, args.epochs)
                    else:
                        out = runner(ds, X_tr_c, X_te_c, t_tr, t_te, scale_c)
                    if out is None:
                        print(" SKIP")
                        continue
                except Exception as e:
                    print(f" ERR ({type(e).__name__}: {e})")
                    out = dict(
                        fit_time_s=math.nan, recon_rmse=math.nan,
                        forecast_rmse=math.nan, n_params=0,
                        notes=f"unhandled:{type(e).__name__}:{e}"[:120],
                    )

                row = dict(
                    method=method_name, dataset=ds_name, noise=noise, **out
                )
                rows.append(row)
                print(
                    f" {out['fit_time_s']:6.2f}s  "
                    f"recon={out['recon_rmse']:.3g}  "
                    f"fc={out['forecast_rmse']:.3g}"
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method", "dataset", "noise", "fit_time_s",
        "recon_rmse", "forecast_rmse", "n_params", "notes",
    ]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nWrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
