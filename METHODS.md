# DMD Toolkit — Methods Reference

Thirteen variants split into a **classical core** (`dmd_toolkit`, numpy/scipy only) and
a **neural** sub-package (`dmd_toolkit.neural`, requires `torch`).

For background on Koopman theory see Brunton & Kutz, *Data-Driven Science and Engineering*
(Cambridge UP, 2019), Ch. 7.

---

## Notation

| Symbol | Meaning |
|---|---|
| $X_1 = [x_0, \ldots, x_{m-1}]$ | "left" snapshot matrix, $(n \times m)$ |
| $X_2 = [x_1, \ldots, x_m]$ | "right" snapshot matrix, $(n \times m)$ |
| $A$ | full-state propagation operator: $x_{k+1} \approx A x_k$ |
| $\tilde A$ | projected (rank-$r$) operator |
| $\Phi$ | DMD mode matrix, $(n \times r)$ |
| $\Lambda$ | diagonal matrix of discrete eigenvalues $\lambda_j$ |
| $\omega_j$ | continuous-time eigenvalue $\omega_j = \ln\lambda_j/\Delta t$ |
| $b$ | amplitude vector: $x_0 \approx \Phi b$ |
| $U_r\Sigma_r V_r^*$ | rank-$r$ truncated SVD of $X_1$ |

Reconstruction formula (common to all classical methods):

$$
x(t) = \Phi\,\operatorname{diag}(e^{\omega_j t})\,b
      = \sum_{j=1}^{r} \phi_j\,b_j\,e^{\omega_j t}
$$

---

## Classical core

### 1. Exact DMD

**Reference:** Tu, Rowley, Luchtenburg, Brunton, Kutz —
*J. Comput. Dyn.* 1(2), 391–421 (2014). §2.

**Module:** `dmd_toolkit.exact` · **API:** `ExactDMD`, `exact_dmd(data, rank, dt)`

#### Problem statement

Find the best-fit linear operator in the least-squares sense:

$$
A^* = \operatorname*{argmin}_{A} \|X_2 - A X_1\|_F^2 = X_2 X_1^{+}
$$

Because $n$ can be large, solve via the rank-$r$ SVD $X_1 \approx U_r\Sigma_r V_r^*$:

#### Algorithm (Tu et al. 2014, Def. 2.2 & Thm. 2.4)

1. $X_1 = U_r\Sigma_r V_r^*$ (truncated SVD)
2. $\tilde A = U_r^* X_2 V_r \Sigma_r^{-1}$ — projected operator
3. $\tilde A W = W\Lambda$ — eigendecomposition
4. **Exact DMD modes:** $\Phi = X_2 V_r \Sigma_r^{-1} W$

   These satisfy $A\Phi = \Phi\Lambda$ exactly on $\operatorname{range}(X_1)$,
   unlike the "projected" modes $U_r W$ (Tu et al. 2014, Thm. 2.4).

5. $\omega_j = \ln\lambda_j / \Delta t$
6. $b = \Phi^+ x_0$

**Use when:** evenly sampled snapshots, low-to-moderate noise.

---

### 2. Optimized DMD

**Reference:** Askham & Kutz —
*SIAM J. Appl. Dyn. Syst.* 17(2), 1100–1126 (2018). §2.

**Module:** `dmd_toolkit.optimized` · **API:** `OptimizedDMD`, `optimized_dmd(data, rank, t)`

#### Problem statement

Replace the discrete-eigenvalue constraint with a continuous one and optimize jointly:

$$
\min_{\alpha \in \mathbb{C}^r,\; B \in \mathbb{C}^{r \times n}}
\bigl\| X^{\top} - \Phi(\alpha,t)\,B \bigr\|_F^2,
\qquad
\Phi(\alpha,t)_{ij} = e^{\alpha_j t_i}
$$

where $X^{\top}$ is $(m \times n)$, rows are snapshots at times $t_1,\ldots,t_m$.

#### Variable projection (VARPRO)

For fixed $\alpha$, $B$ is solved by linear LS:

$$
B^*(\alpha) = \Phi(\alpha,t)^{+} X^{\top}
$$

Substituting back gives a nonlinear residual in $\alpha$ only:

$$
\rho(\alpha) = \bigl(I - \Phi\Phi^{+}\bigr) X^{\top}
$$

Outer minimization over $\alpha \in \mathbb{C}^r$ via Levenberg–Marquardt
(Askham & Kutz 2018, §2.2). Handles **unevenly spaced** $t_i$ naturally.

**Use when:** noisy data, non-uniform sampling, or you need a tighter
fit than Exact DMD. `init="exact"` warm-starts from Exact DMD eigenvalues.

---

### 3. BOP-DMD

**Reference:** Sashidhar & Kutz —
*Phil. Trans. R. Soc. A* 380, 20210199 (2022). §3.

**Module:** `dmd_toolkit.bop` · **API:** `BOPDMD`, `bop_dmd(data, rank, t)`

#### Bootstrap aggregation over snapshot subsets

For $b = 1,\ldots,N_b$:
1. Sample column index set $\mathcal{I}_b \subset \{1,\ldots,m\}$
   of size $\lfloor\eta m\rfloor$ (default $\eta = 0.6$)
2. Fit OptimizedDMD on $(X[:,\mathcal{I}_b],\, t[\mathcal{I}_b])$ → $\alpha_b, \Phi_b$

Aggregate statistics (Sashidhar & Kutz 2022, Eq. 3.2–3.3):

$$
\bar\alpha = \frac{1}{N_b}\sum_{b=1}^{N_b}\alpha_b,
\qquad
\sigma_\alpha^2 = \frac{1}{N_b}\sum_{b=1}^{N_b}\|\alpha_b - \bar\alpha\|^2
$$

`forecast_ensemble(t)` returns the full $(N_b, n, |t|)$ tensor;
`omega_std` reports $\sigma_\alpha$ per eigenvalue.

**Use when:** eigenvalue uncertainty quantification is needed, or the data
is short/noisy enough that a single OptDMD fit is unreliable.

---

### 4. Multi-Resolution DMD

**Reference:** Kutz, Fu, Brunton —
*SIAM J. Appl. Dyn. Syst.* 15(2), 713–735 (2016). §3.

**Module:** `dmd_toolkit.multires` · **API:** `MultiResolutionDMD`, `mr_dmd(data, ...)`

#### Recursive multi-scale decomposition

At each level $\ell$ and window $[t_\ell, t_\ell + T_\ell]$, $T_\ell = T/2^\ell$:

1. Fit Exact DMD → modes $\Phi^{(\ell)}$, eigenvalues $\lambda^{(\ell)}$
2. Classify modes as **slow** if (Kutz et al. 2016, Eq. 3.1):

$$
|\operatorname{Im}(\omega_j^{(\ell)})| \;\le\; \rho \cdot \frac{2\pi}{T_\ell}
$$

3. Subtract slow reconstruction from the window:

$$
X^{(\ell)}_{\rm slow}(t) = \sum_{j\,\text{slow}} \phi_j^{(\ell)}\,b_j^{(\ell)}\,e^{\omega_j^{(\ell)}(t - t_\ell)}
$$

4. Recurse on the residual $X^{(\ell+1)} = X^{(\ell)} - X^{(\ell)}_{\rm slow}$,
   splitting into left and right half-windows.

Final reconstruction sums all slow-mode contributions across all levels:

$$
x(t) = \sum_{\ell}\sum_{j\,\in\,\text{slow}_\ell(t)}
        \phi_j^{(\ell)}\,b_j^{(\ell)}\,e^{\omega_j^{(\ell)}(t-t_\ell)}
$$

**Use when:** data spans multiple timescales (slow background + fast
transients, non-stationary signals). A single global DMD smears the spectrum.

---

### 5. HAVOK — Hankel Alternative View Of Koopman

**Reference:** Brunton, Brunton, Proctor, Kaiser, Kutz —
*Nat. Commun.* 8, 19 (2017). §Methods.

**Module:** `dmd_toolkit.havok` · **API:** `HAVOK`, `havok(x, delays, rank, dt)`

#### Time-delay (Hankel) embedding

For a scalar signal $x(t_k)$, form the $d \times M$ Hankel matrix
(Takens 1981; Arbabi & Mezić 2017):

$$
H = \begin{bmatrix}
x(t_1)   & x(t_2)   & \cdots & x(t_M) \\
x(t_2)   & x(t_3)   & \cdots & x(t_{M+1}) \\
\vdots   &           &        & \vdots \\
x(t_d)   & x(t_{d+1})& \cdots & x(t_{M+d-1})
\end{bmatrix}
$$

SVD: $H = U\Sigma V^{\top}$. Let $v_1(t),\ldots,v_r(t)$ be the first $r$ columns
of $V$ ("delay coordinates").

#### HAVOK regression (Brunton et al. 2017, Eq. 2)

$$
\frac{d}{dt}\begin{pmatrix}v_1\\\vdots\\v_{r-1}\end{pmatrix}
= A\begin{pmatrix}v_1\\\vdots\\v_{r-1}\end{pmatrix}
+ B\,v_r(t)
$$

$A\in\mathbb{R}^{(r-1)\times(r-1)}$, $B\in\mathbb{R}^{(r-1)\times 1}$ fit by linear
regression on the time-derivative of $V$. The $r$-th mode $v_r(t)$ is the
**intermittent chaotic forcing** — lobe-switch events in Lorenz correspond to
bursts in $v_r$.

**Use when:** scalar chaotic / intermittent signals where standard DMD
fails because the dynamics are nonlinear in the raw observable.

---

### 6. Multi-Shot DMD

**References:**
- Tu et al. (2014), §1 (snapshot-matrix formulation).
- Brunton, Johnson, Ojemann, Kutz —
  *J. Neurosci. Methods* 261, 1–9 (2016). §2.3 (trial-stratified data).

**Module:** `dmd_toolkit.multishot` · **API:** `MultiShotDMD`, `multi_shot_dmd(data, rank, dt)`

#### Shared operator, per-trial initial conditions

Given data $X \in \mathbb{C}^{n \times K \times T}$ (channels × trials × time),
build within-trial snapshot pairs — **never across the trial boundary**:

$$
\mathbf{X}_1 = \bigl[X_1^{(1)}\;\big|\;\cdots\;\big|\;X_1^{(K)}\bigr],
\qquad
\mathbf{X}_2 = \bigl[X_2^{(1)}\;\big|\;\cdots\;\big|\;X_2^{(K)}\bigr]
$$

where $X_i^{(k)} = X[:,k,i{-}1:T{-}(2{-}i)]$. Fit Exact DMD on
$(\mathbf{X}_1, \mathbf{X}_2)$ to get a single shared $(\Phi, \Lambda)$.

Per-trial amplitudes (the only trial-specific quantity):

$$
b^{(k)} = \Phi^{+}\,x^{(k)}(0), \qquad k = 1,\ldots,K
$$

Reconstruction per trial:

$$
x^{(k)}(t) = \Phi\,\operatorname{diag}(e^{\omega_j t})\,b^{(k)}
$$

Fitting $K$ trials jointly regularises the DMD via a larger effective snapshot
count $K(T-1)$ without polluting $X_1$ with boundary steps.

**Use when:** trial-stratified neuroscience / control data where dynamics
are assumed shared across trials and only initial conditions vary.
For uncertainty, wrap in BOP-DMD with trials as the bootstrap units.

---

### 7. piDMD — Physics-Informed DMD

**Reference:** Baddoo, Herrmann, McKeon, Kutz, Brunton —
*Proc. R. Soc. A* 479, 20220576 (2023).

**Module:** `dmd_toolkit.pidmd` · **API:** `piDMD`, `pidmd(data, constraint, rank, dt)`

#### Constrained operator problem (Baddoo et al. 2023, §2)

$$
A^* = \operatorname*{argmin}_{A\,\in\,\mathcal{M}} \|X_2 - A X_1\|_F^2
$$

where $\mathcal{M}$ is a matrix manifold encoding prior physics. Working in the
rank-$r$ SVD subspace (let $Y_i = U_r^* X_i$, $G = Y_1 Y_1^*$, $M = Y_2 Y_1^*$):

| Constraint | Manifold $\mathcal{M}$ | Closed-form solution |
|---|---|---|
| `"unitary"` | $A^*A = I$ | Procrustes: $A^* = U_p V_p^*$ where $Y_2 Y_1^* = U_p\Sigma_p V_p^*$ |
| `"symmetric"` | $A = A^*$ | Lyapunov: $AG + GA = M + M^*$ |
| `"skew_symmetric"` | $A = -A^*$ | Sylvester: $AG - GA = M - M^*$ |
| `"diagonal"` | $A_{ij}=0,\;i\neq j$ | $a_i = \langle x_2^{(i)}, x_1^{(i)}\rangle / \|x_1^{(i)}\|^2$ |
| `"circulant"` | $A = F^{-1}\operatorname{diag}(\hat a)F$ | Per-bin: $\hat a_k = \langle\hat x_2^{(k)},\hat x_1^{(k)}\rangle/\|\hat x_1^{(k)}\|^2$ |

The unitary constraint enforces exact energy conservation
($\|Ax\|=\|x\|$, relevant for Hamiltonian/Schrödinger dynamics).
The circulant constraint is exact for spatially periodic, translation-invariant PDEs.

**Use when:** physical symmetry is known a priori. Especially effective in
low-data / high-noise regimes where constraint acts as implicit regularisation
(Baddoo et al. 2023, §4).

---

### 8. Kernel DMD

**References:**
- Williams, Rowley, Kevrekidis —
  *J. Nonlinear Sci.* 25(6), 1307–1346 (2015). §3.
- Panda, Singh, Kutz — arxiv:2505.06806 (2025). §2.

**Module:** `dmd_toolkit.kernel` · **API:** `KernelDMD`, `kernel_dmd(data, kernel, rank, dt)`

#### RKHS lift and kernel Gram matrices (Williams et al. 2015, §3.1)

Let $\psi: \mathbb{R}^n \to \mathcal{H}$ be the feature map of a reproducing-kernel
Hilbert space with kernel $k(x,y) = \langle\psi(x),\psi(y)\rangle_{\mathcal{H}}$.
Build the two Gram matrices ($m$ training pairs):

$$
G_{ij} = k(x_i, x_j), \qquad A_{ij} = k(x_{i+1}, x_j),
\qquad i,j = 1,\ldots,m
$$

The reduced Koopman matrix is $\hat K = G^{+} A$ (Williams et al. 2015, Eq. 3.6).

#### Numerically stable eigendecomposition (Panda et al. 2025, §2.2)

Regularise: $G_\epsilon = G + \epsilon I$. Eigen-decompose the symmetric
$G_\epsilon = Q\Sigma^2 Q^{\top}$ (keeping top $r$ values), then form the
similarity-equivalent reduced operator:

$$
\hat K_r = \Sigma^{-1} Q^{\top} A\, Q\, \Sigma^{-1}
$$

whose eigenvalues $\lambda_j$ equal those of $G_\epsilon^{+} A$ on the retained
subspace. Eigenvectors $\hat K_r\, v = \lambda v$ define the Koopman
eigenfunctions at any point $x$:

$$
\varphi_j(x) = v_j^*\,\Sigma^{-1}\,Q^{\top}\,k(X_1, x)
$$

where $k(X_1,x) = (k(x_1,x),\ldots,k(x_m,x))^{\top}$.

#### Koopman modes for the identity observable

For the observable $g(x) = x$, modes $\xi_j$ are found by LS projection:

$$
X_1 \approx \sum_j \xi_j\,\varphi_j(X_1), \qquad
\Xi = X_1\,\Phi_{\rm train}^{+}
$$

where $\Phi_{\rm train}$ collects $\varphi_j(x_i)$ for all training points.

**Kernels:** `"rbf"` ($k(x,y) = e^{-\gamma\|x-y\|^2}$, median-heuristic
$\gamma$ by default); `"poly"` ($k = (\gamma x^\top y + c_0)^d$);
`"linear"` ($k = x^\top y$, equivalent to DMD on real data).

**Use when:** dynamics are nonlinear and you want Koopman eigenfunctions
without specifying a dictionary. Panda 2025 regularisation handles near-singular
Gram matrices arising in high-dimensional or sparse-snapshot settings.

---

## Neural (`dmd_toolkit.neural`, requires PyTorch)

All five lazy-import torch so the classical core runs without it.

---

### 9. SHRED — SHallow REcurrent Decoder

**References:**
- Williams, Zhe, Kutz, Brunton —
  *Sci. Adv.* 9, eadi1973 (2023). §Materials and Methods.
- Callaham, Koch, Brunton, Kutz —
  *Nat. Commun.* 15, 5808 (2024). (ROM extension.)

**Module:** `dmd_toolkit.neural.shred` · **API:** `SHRED`, `fit_shred(X, sensor_indices, lags, ...)`

#### Sensor + time-window to full field

Let $y(t) = C x(t) \in \mathbb{R}^p$ be sparse sensor measurements ($p \ll n$).
SHRED learns the map from a window of $L$ sensor readings to the full state:

$$
\hat x_t = D_\theta\!\left(h_t\right),
\qquad
h_t = \operatorname{LSTM}_\phi\!\left([y_{t-L+1},\ldots,y_t]\right)
$$

where $h_t \in \mathbb{R}^s$ is the final LSTM hidden state and $D_\theta$
is a shallow MLP decoder. Trained end-to-end by:

$$
\mathcal{L} = \frac{1}{T}\sum_{t=L}^{T}\|\hat x_t - x_t\|_2^2
$$

Architecture default: LSTM ($s = 64$, 2 layers) + MLP decoder with hidden
widths $(350, 400)$. All inputs/targets standardised.

**Use when:** very few sensors, need the whole spatial field.
Foundation block for the SKS method (#12).

---

### 10. Deep Probabilistic Koopman (DPK)

**Reference:** Mallen & Kutz —
*Int. J. Forecasting* 40(2), 435–451 (2024). §3.

**Module:** `dmd_toolkit.neural.dpk` · **API:** `DeepProbKoopman`, `fit_dpk(X, latent_dim, ...)`

#### Koopman-structured latent space with calibrated uncertainty

Encoder $\phi_\theta: x \mapsto z \in \mathbb{R}^d$; linear latent propagator $K$;
two decoders for mean $\mu_\varphi$ and log-variance $\sigma^2_\psi$
(Mallen & Kutz 2024, §3.1):

$$
z_{t+h} = K^h z_t
$$

$$
\mathcal{L} =
  w_r \|{\mu_\varphi(\phi_\theta(x_t)) - x_t}\|^2
  + w_\ell \sum_{h=1}^{H}\|z_{t+h} - K^h z_t\|^2
  + w_n \sum_{h=1}^{H}
    \frac{1}{2}\!\left[\log\sigma^2_{t+h}
    + \frac{(x_{t+h} - \hat\mu_{t+h})^2}{\sigma^2_{t+h}}\right]
$$

Training samples random $(x_0, h)$ pairs and rolls out $H$-step trajectories
in latent space. `model.forecast(x0, n_steps)` returns $(\mu, \sigma)$ in
original scale.

**Use when:** long-horizon forecast with **prediction intervals** is required.

---

### 11. Discrepancy Modelling

**Reference:** Ebers, Steele, Brunton, Kutz —
*Proc. R. Soc. A* 479, 20230168 (2023). §2.

**Module:** `dmd_toolkit.neural.discrepancy` · **API:** `DiscrepancyModel(base=...)`

#### Hybrid: physics base + neural residual (Ebers et al. 2023, Eq. 2.1)

$$
\hat x(t) = \underbrace{M_{\rm base}(t)}_{\text{DMD reconstruction}}
           + \underbrace{f_\theta(t)}_{\text{NN residual}}
$$

$f_\theta: t \mapsto \mathbb{R}^n$ is a small tanh-MLP trained on the
time-series of residuals $r(t) = x(t) - M_{\rm base}(t)$:

$$
\mathcal{L} = \sum_{t}\|r(t) - f_\theta(t)\|^2
$$

Both $t$ and $r$ are standardised; base model can be any
`ExactDMD`/`OptimizedDMD`/`BOPDMD`.

**Use when:** you have a serviceable linear base model but want a cheap
learned correction for the nonlinear/stochastic residual. The split keeps
interpretability: inspect the base model for the dominant dynamics, the
NN for structured error.

---

### 12. SINDy + Koopman + SHRED (SKS)

**Reference:** Gao, Williams, Kutz —
*L4DC Proceedings* (2025). §3.

**Module:** `dmd_toolkit.neural.sks` · **API:** `SINDyKoopmanSHRED`, `fit_sks(X, ...)`

#### Sparse symbolic + linear latent models on SHRED's latent space

1. Train SHRED (#9) on sparse sensor windows → final hidden state
   $Z = (z_1,\ldots,z_T) \in \mathbb{R}^{T\times d}$ as latent trajectory.

2. **Koopman in latent:** fit DMD to $(Z_1, Z_2)$:

$$
z_{t+1} \approx K z_t, \qquad K = Z_2 Z_1^{+}
$$

3. **SINDy in latent** (Brunton, Proctor, Kutz — *Science* 351, 2016):
   build polynomial library $\Theta(Z)$ and solve via sequential-thresholded
   LS (STLSQ):

$$
\dot Z \approx \Theta(Z)\,\Xi, \qquad
\Xi_j = \operatorname*{argmin}_{\xi}\|\dot Z_j - \Theta(Z)\xi\|^2
\;\text{ s.t. }\;\|\xi\|_0 \leq s
$$

`forecast_latent(n, mode="koopman"|"sindy")` rolls the chosen model
forward, then `decode_latent(Z)` maps back through the SHRED decoder.

**Use when:** sparse sensors + need an **interpretable** latent model
(SINDy gives symbolic equations; Koopman gives a spectrum).

---

### 13. Koopman Autoencoder

**References:**
- Lusch, Brunton, Kutz —
  *Nat. Commun.* 9, 4950 (2018). §Methods.
- Morrison & Kutz —
  *SIAM J. Appl. Dyn. Syst.* 23(3), 2265–2294 (2024). §3.

**Module:** `dmd_toolkit.neural.manifold` · **API:** `KoopmanAutoencoder`, `fit_koopman_ae(X, ...)`

#### Deep encoder–decoder with linear Koopman constraint (Lusch et al. 2018, Eq. 1–3)

Let $\phi_\theta: \mathbb{R}^n \to \mathbb{R}^d$ (encoder) and
$\varphi_\psi: \mathbb{R}^d \to \mathbb{R}^n$ (decoder), with learnable
$K \in \mathbb{R}^{d\times d}$:

$$
\mathcal{L} =
  \underbrace{\|\varphi_\psi(\phi_\theta(x_t)) - x_t\|^2}_{\text{reconstruction}}
  + w_1\underbrace{\|\phi_\theta(x_{t+1}) - K\phi_\theta(x_t)\|^2}_{\text{Koopman linearity}}
  + w_2\underbrace{\sum_{h=1}^{H}\|\varphi_\psi(K^h\phi_\theta(x_t)) - x_{t+h}\|^2}_{\text{multi-step prediction}}
$$

Morrison & Kutz (2024) extend to **invariant manifolds**: the decoder is
constrained to map exactly onto a learned invariant subset of the state space,
giving Koopman eigenfunctions that are globally valid rather than locally linear.

After training, eigendecompose $K$:

$$
K v_j = \lambda_j v_j, \qquad
\varphi_j(x) = v_j^*\,\phi_\theta(x)
$$

$\varphi_j$ are the approximate Koopman eigenfunctions; $\lambda_j$ the
Koopman eigenvalues (Morrison & Kutz 2024, Def. 3.2).

**Use when:** nonlinear coordinate change that genuinely linearises the
dynamics, with explicit access to Koopman eigenfunctions. Most expensive
method here; use when eigenfunctions on the manifold are the target, not
just forecasts.

---

## Quick chooser

| Need | Use |
|---|---|
| Modes + freqs, evenly sampled, low noise | **ExactDMD** |
| Noisy or unevenly sampled snapshots | **OptimizedDMD** |
| Eigenvalue uncertainty bands | **BOPDMD** |
| Multi-scale / non-stationary signals | **MultiResolutionDMD** |
| Chaotic scalar signal | **HAVOK** |
| Trial-stratified data (channel × trial × time) | **MultiShotDMD** |
| Physics constraints (energy-preserving, symmetric, …) | **piDMD** |
| Nonlinear Koopman without a manual dictionary | **KernelDMD** |
| Full field from a handful of sensors | **SHRED** |
| Probabilistic forecast with intervals | **DeepProbKoopman** |
| DMD + learned correction for nonlinear residual | **DiscrepancyModel** |
| Sparse sensors + symbolic / spectral latent model | **SINDyKoopmanSHRED** |
| Nonlinear manifold + explicit Koopman eigenfunctions | **KoopmanAutoencoder** |
