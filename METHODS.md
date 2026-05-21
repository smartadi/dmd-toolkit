# DMD Toolkit — Methods Reference

Ten variants split into a **classical core** (`dmd_toolkit`, numpy/scipy only) and
a **neural** sub-package (`dmd_toolkit.neural`, requires `torch`). Each entry
lists the idea, algorithm, key parameters, and the Python entry point.

For mathematical preliminaries (snapshot matrices, Koopman theory, the DMD
ansatz) see `../dynamic_mode_decomposition.md`.

---

## Classical core

### 1. Exact DMD

**Reference:** Tu, Rowley, Luchtenburg, Brunton, Kutz — *J. Comput. Dyn.* 1 (2014).
**Module:** `dmd_toolkit.exact` · **API:** `ExactDMD`, `exact_dmd(data, rank, dt)`

The textbook DMD. Given snapshot pairs $X_1, X_2$:

1. $X_1 = U \Sigma V^*$ (truncated SVD at `rank`)
2. $\tilde A = U^* X_2 V \Sigma^{-1}$
3. Eigendecompose $\tilde A W = W \Lambda$
4. Modes $\Phi = X_2 V \Sigma^{-1} W$
5. Continuous-time eigenvalues $\omega = \log(\Lambda)/\Delta t$
6. Amplitudes $b = \Phi^\dagger x_1$

**Use when:** evenly sampled snapshots, low noise, you want spatial modes +
frequencies fast. `frequencies` and `growth_rates` are exposed as properties.

---

### 2. Optimized DMD

**Reference:** Askham & Kutz — *SIAM J. Appl. Dyn. Syst.* 17 (2018).
**Module:** `dmd_toolkit.optimized` · **API:** `OptimizedDMD`, `optimized_dmd(data, rank, t)`

Variable-projection nonlinear least squares:

$$
\min_{\omega,\,B} \big\| X - \Phi(\omega)\, B \big\|_F^2,
\quad \Phi(\omega)_{ij} = e^{\omega_j t_i}
$$

Inner $B$ solved by linear LS once $\omega$ is fixed; outer optimisation over
$\omega$ via `scipy.optimize.least_squares` (Levenberg–Marquardt).

**Use when:** noisy data, **unevenly spaced samples**, or you need a tighter
fit than exact DMD. Pass `t=` for non-uniform sampling. Init `"exact"` warm-
starts from Exact DMD eigenvalues.

---

### 3. BOP-DMD (Bagging, Optimized)

**Reference:** Sashidhar & Kutz — *Phil. Trans. R. Soc. A* 380 (2022).
**Module:** `dmd_toolkit.bop` · **API:** `BOPDMD`, `bop_dmd(data, rank, t)`

Bootstrap-aggregating wrapper around Optimized DMD: fits `n_trials` models on
random column subsets (`subsample_fraction` of snapshots) and reports
$\bar\omega$, $\sigma_\omega$, mean modes, mean amplitudes.

`forecast_ensemble(t)` returns the full $(n_\text{trials}, n_\text{features}, |t|)$
tensor for uncertainty bands.

**Use when:** you need **uncertainty quantification** on DMD eigenvalues, or
the data is short / noisy enough that a single OptDMD fit is unstable. Greedy
eigenvalue alignment matches modes across trials (mis-orders near-degenerate
eigenvalues — see `_align_trials`).

---

### 4. Multi-Resolution DMD

**Reference:** Kutz, Fu, Brunton — *SIAM J. Appl. Dyn. Syst.* 15 (2016).
**Module:** `dmd_toolkit.multires` · **API:** `MultiResolutionDMD`, `mr_dmd(data, max_levels, ...)`

Recursive time-windowed DMD:

1. Fit Exact DMD on the current window.
2. Extract "slow" modes where $|\omega_\text{imag}| \le 2\pi \cdot (\text{slow\_cutoff}/\text{window})$.
3. Subtract their reconstruction.
4. Recurse on the left and right halves of the residual, up to `max_levels`.

Stored as a list of `(level, t_offset, modes, omega, amplitudes)` dicts;
`reconstruct(n_times)` sums all level contributions onto a uniform grid.

**Use when:** dynamics span **multiple timescales** (slow + fast transients,
non-stationary signals) where one global DMD smears the spectrum.

---

### 5. HAVOK — Hankel Alternative View Of Koopman

**Reference:** Brunton, Brunton, Proctor, Kutz — *Nature Comm.* 8 (2017).
**Module:** `dmd_toolkit.havok` · **API:** `HAVOK`, `havok(x, delays, rank, dt)`

For a **scalar** chaotic time series $x(t)$:

1. Hankel embed → $H$ with `delays` time-lags.
2. SVD: $H = U \Sigma V^*$, keep `rank` columns of $V$.
3. Linear regression on the first $r-1$ modes:
   $\dot v_{1:r-1} = A\,v_{1:r-1} + B\,v_r$
4. The $r$-th mode $v_r(t)$ is the **intermittent forcing**.

`forcing_events(quantile=0.95)` flags bursts (e.g. Lorenz lobe-switches).

**Use when:** chaotic / intermittent scalar signals where standard DMD fails
because the dynamics aren't linear in the observable space.

---

## Neural (`dmd_toolkit.neural`, requires torch)

All five lazy-import torch, so the classical core works without it.

### 6. SHRED — SHallow REcurrent Decoder

**Reference:** Williams, Kutz et al. — *Proc. R. Soc. A* (2024); ROM variant *Nature Comm.* (2025).
**Module:** `dmd_toolkit.neural.shred` · **API:** `SHRED`, `fit_shred(X, sensor_indices, lags, ...)`

Reconstructs the full high-dim state from a sparse sensor time-window:

```
sensor_window (lags × n_sensors)  →  LSTM  →  final hidden h
h  →  MLP (decoder_hidden)  →  full state (n_features)
```

Architecture: LSTM (`hidden_size=64`, 2 layers) + 3-layer MLP decoder
(default `(350, 400)` hidden). Trained with MSE on standardised targets.
`predict_shred(model, info, window)` decodes a single window back to the full
field in original scale.

**Use when:** you have very few sensors and need the **whole field**.
Foundation block for SKS (#9).

---

### 7. Deep Probabilistic Koopman (DPK)

**Reference:** Mallen & Kutz — *Int. J. Forecasting* 40 (2024).
**Module:** `dmd_toolkit.neural.dpk` · **API:** `DeepProbKoopman`, `fit_dpk(X, latent_dim, ...)`

Long-horizon forecasting with **calibrated uncertainty**. Encoder $x \to z$,
linear latent operator $K$, two decoders for mean and log-variance:

$$
\mathcal L \;=\; w_\text{recon}\,\|\hat\mu - x\|^2 \;+\; w_\text{linear}\,\|z_{t+h} - K^h z_t\|^2 \;+\; w_\text{nll}\,\tfrac12(\log\sigma^2 + (x-\mu)^2/\sigma^2)
$$

Rollout training: sample random $(x_0, \text{traj})$ pairs, predict
`horizon`-step trajectories in latent, decode to $(\mu, \sigma)$.
`model.forecast(x0, n_steps)` returns mean/std in original scale.

**Use when:** you need **prediction intervals**, not just point forecasts.
The simplified single-$K$ version here handles non-periodic systems; the full
Mallen–Kutz formulation supports time-periodic $K(t)$.

---

### 8. Discrepancy Modelling

**Reference:** Ebers, Steele, Kutz (2024).
**Module:** `dmd_toolkit.neural.discrepancy` · **API:** `DiscrepancyModel(base=...)`

Hybrid: a classical base model (any `ExactDMD`/`OptimizedDMD`/`BOPDMD`) +
neural **residual** in time:

$$
\hat x(t) \;=\; \underbrace{\text{base.reconstruct}(t)}_\text{linear part} \;+\; \underbrace{f_\theta(t)}_\text{NN residual}
$$

$f_\theta$ is a small tanh MLP $t \to \mathbb R^{n_\text{features}}$ trained
to MSE on the residual. Standardisation on both $t$ and residual.

**Use when:** you have a serviceable physics-flavoured base model (DMD) but
want a learned correction for the parts it can't capture (nonlinearity,
slow drift). Cheap to fit, interpretable split.

---

### 9. SINDy + Koopman + SHRED (SKS)

**Reference:** Gao, Williams, Kutz — *L4DC* (2025).
**Module:** `dmd_toolkit.neural.sks` · **API:** `SINDyKoopmanSHRED`, `fit_sks(X, sensor_indices, ...)`

Stacks on SHRED (#6):

1. Train SHRED on sparse sensors.
2. Run the sensor windows through the trained LSTM and take the **final
   hidden state** as a low-dim latent trajectory $Z \in \mathbb R^{n\times d}$.
3. In that latent space, fit **both**:
   - **SINDy:** polynomial library $\Theta(Z)$ (order `sindy_order`), sparse
     coefficients via STLSQ (sequential thresholded LS).
   - **Koopman:** linear $K$ such that $Z_{t+1} \approx K Z_t$ (DMD on the
     latent), plus its eigendecomposition.
4. Forecast in latent via either symbolic SINDy (forward-Euler) or linear
   Koopman, then decode through SHRED.

`forecast_latent(n_steps, mode="koopman"|"sindy")` + `decode_latent(Z)`.

**Use when:** sparse sensors **plus** you want an **interpretable** latent
model — SINDy gives symbolic equations, Koopman gives a spectrum.

---

### 10. Invariant Manifolds + Koopman Eigenfunctions

**Reference:** Morrison & Kutz — *SIAM J. Appl. Dyn. Syst.* (2024). Architecture
extends Lusch, Brunton, Kutz — *Nature Comm.* 9 (2018).
**Module:** `dmd_toolkit.neural.manifold` · **API:** `KoopmanAutoencoder`, `fit_koopman_ae(X, latent_dim, ...)`

Deep Koopman autoencoder with three losses:

$$
\mathcal L \;=\; \underbrace{\|\text{dec}(\text{enc}(x_t)) - x_t\|^2}_\text{recon (manifold)}
\;+\; \underbrace{\|\text{enc}(x_{t+1}) - K\,\text{enc}(x_t)\|^2}_\text{Koopman linearity}
\;+\; \underbrace{\|\text{dec}(K^h\,\text{enc}(x_t)) - x_{t+h}\|^2}_\text{multi-step prediction}
$$

Tanh-MLP encoder + symmetric decoder + learned $K$ (init $0.9 I$ + small noise).
After training:

- `.eigenvalues` / `.eigenvectors` — spectrum of $K$
- `.eigenfunctions(X)` — $\phi_j(x) = v_j \cdot \text{encoder}(x)$, the
  Koopman eigenfunctions restricted to the learned manifold
- `.forecast(x0, n_steps)` — free-run rollout in original scale

**Use when:** you want a **nonlinear coordinate change** that genuinely
linearises the dynamics, with explicit access to Koopman eigenfunctions —
the heaviest method here, best left until you actually need eigenfunctions
on the manifold rather than just forecasts.

---

---

### 11. piDMD — Physics-Informed DMD

**Reference:** Baddoo, Herrmann, McKeon, Kutz, Brunton — *Proc. R. Soc. A* 479 (2023).
**Module:** `dmd_toolkit.pidmd` · **API:** `piDMD`, `pidmd(data, constraint, rank, dt)`

Standard DMD finds the unconstrained least-squares operator. piDMD instead
solves

$$\min_{A \in \mathcal C} \|X_2 - A X_1\|_F$$

where the constraint set $\mathcal C$ encodes known physics. Five constraints:

| `constraint=` | Structure | Closed form |
|---|---|---|
| `"unitary"` | $A^* A = I$ (energy-preserving) | Procrustes: SVD of $X_2 X_1^*$ |
| `"symmetric"` | $A = A^*$ (self-adjoint, real spectrum) | Lyapunov eq. $AG + GA = M + M^*$ |
| `"skew_symmetric"` | $A = -A^*$ (purely imaginary spectrum) | Sylvester eq. $AG - GA = M - M^*$ |
| `"diagonal"` | Per-feature scalar fit | Channel-wise ratio |
| `"circulant"` | Translation-invariant | DFT-domain diagonal |

For the non-local constraints, fitting happens in an SVD-reduced subspace
(`rank` controls truncation); modes are lifted back to full space.

**Use when:** you have physical knowledge of the operator structure. `"unitary"`
enforces energy conservation (Hamiltonian systems, lossless wave propagation);
`"symmetric"` is appropriate for self-adjoint PDE operators; `"diagonal"` is the
fastest zero-coupling baseline.

---

### 12. Kernel DMD

**References:**
- Williams, Rowley, Kevrekidis — *J. Nonlinear Sci.* 25 (2015).
- Panda, Singh, Kutz — arxiv:2505.06806 (2025).

**Module:** `dmd_toolkit.kernel` · **API:** `KernelDMD`, `kernel_dmd(data, kernel, rank, dt)`

Lifts snapshots through a (possibly infinite-dimensional) feature map
$\psi(x)$ implicitly via kernel evaluations
$k(x, y) = \langle\psi(x), \psi(y)\rangle$, then fits a linear Koopman
operator in feature space. Two Gram matrices

$$G_{ij} = k(x_i, x_j), \qquad A_{ij} = k(x_{i+1}, x_j)$$

define a reduced Koopman matrix $\hat K = \Sigma^{-1} Q^\top A Q \Sigma^{-1}$
where $G = Q \Sigma^2 Q^\top$. Eigenvalues of $\hat K$ approximate the Koopman
spectrum; eigenfunctions are evaluated on new points via
$\phi_j(x) = v_j^* (\Sigma^+ Q^\top k(X_1, x))$.

Koopman modes $\xi_j$ for the identity observable are recovered by projecting
training snapshots onto the eigenfunction matrix.

Kernels supported: `"rbf"` (Gaussian, median-heuristic bandwidth by default),
`"linear"` (equivalent to DMD on $\text{Re}(X)$), `"poly"` (polynomial
degree `degree` with bias `coef0`).

Panda/Singh/Kutz (2025) regularisation: a Tikhonov ridge $\epsilon I$ on $G$
before pseudo-inversion stabilises the spectrum when $G$ is near-singular.

**Use when:** the dynamics are nonlinear and you want Koopman eigenfunctions
without manually specifying a dictionary. RBF kernel adapts to the data
geometry; use `rank` to control approximation quality vs. cost.

---

## Quick chooser

| Need | Use |
|---|---|
| Modes + freqs, evenly sampled, low noise | **ExactDMD** |
| Noisy or unevenly sampled snapshots | **OptimizedDMD** |
| Eigenvalue uncertainty bands | **BOPDMD** |
| Multi-scale / non-stationary signals | **MultiResolutionDMD** |
| Chaotic scalar signal | **HAVOK** |
| Physics constraints (energy-preserving, symmetric, …) | **piDMD** |
| Nonlinear Koopman without a manual dictionary | **KernelDMD** |
| Full field from a handful of sensors | **SHRED** |
| Probabilistic forecast with intervals | **DeepProbKoopman** |
| DMD + learned correction | **DiscrepancyModel** |
| Sparse sensors + symbolic / spectral interpretation | **SINDyKoopmanSHRED** |
| Nonlinear manifold + explicit Koopman eigenfunctions | **KoopmanAutoencoder** |
