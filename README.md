# dde

`dde` is a Python package for Deep Discrete Encoder models, including
parametric Normal, Bernoulli, and Poisson DDE estimation,
semi-parametric DDE estimation for continuous observations and copula/rank
likelihood DDE estimation for mixed-scale observations.

Install in editable mode from this directory:

```bash
pip install -e .
```

Then import from any script or notebook:

```python
from dde import fit, select_hyperparameters
```

A runnable walkthrough (kernel catalogue, a simulated Normal DDE, `fit()` on
the fixed and CSP paths, and `select_hyperparameters` on explicit and
automatic grids) is in the executed notebook `examples/package_demo.ipynb` (about two minutes to rerun):

```bash
jupyter nbconvert --to notebook --execute --inplace examples/package_demo.ipynb
```

A fixed-width Normal DDE supplies its latent widths and layer penalties.  A
CSP fit omits `K_list`; the package initializes the recursive maximal widths
`floor(J / 3), floor(floor(J / 3) / 3), ...` and returns the CSP-active widths
in `result.diagnostics["K_list"]`:

```python
result = fit(
    Y,
    method="DDE",
    family="Normal",
    D=3,
    dimension_method="csp",
    t_spike=[0.01, 0.03, 0.005],  # Laplace spike scales, not rates
    temp=0.9,
    tau=0.3,
    C=1,
    tol=0.005,
    max_iter=50,
    random_state=123,
)
```

Normal CSP uses the CopDDE D-layer CSP algorithm (`csp_backend="copula"`,
the default): the MATLAB `get_SAEM_RL_CSP_D` kernel with its rank-likelihood
resampling removed, because the Normal DDE observes its Gaussian layer.  The
fit runs on column-standardized observations, the unit-variance scale of the
CopDDE latent layer on which the CopDDE `t_spike` and `tau` values are
calibrated, and initializes with the Normal family's data initializer
(`csp_initializer="family"`; the CopDDE `Normal_init_D` rules remain
available as `csp_initializer="copula"`).  Layers 1 to D-1 are solved with
an exact proximal weighted-lasso step (`csp_mstep_solver="fista_lower"`,
`csp_fista_maxiter=300`) under a penalty proportional to N
(`csp_penalty_scaling_power=1`, `csp_penalty_scaling_layers="lower"`,
`csp_penalty_scaling_reference=500`); the deepest layer keeps the CopDDE
inexact L-BFGS-B step (`optimizer_maxiter=15`) with the unscaled penalty.
Observed-layer coefficients and variances are returned on the original
scale; the hard threshold `tau` acts on standardized loadings.  `tol` is the
rolling standard deviation of the relative coefficient change (window 7)
that stops the kernel.  `csp_mstep_solver="lbfgsb"` together with
`csp_penalty_scaling_power=0` restores the CopDDE M-step in every layer;
`csp_slab_shape` and `csp_slab_rate` set the CUSP slab hyperprior (CopDDE
1 and 5); `standardize=False` and `csp_threshold_rule` expose the remaining
adaptations.  Run `scripts/evaluate_normal_csp_copula_kernel.py` to
compare configurations' mean fitted widths against the true widths on the
CopDDE paper designs; `Results/normal_dde_csp_copula_kernel/` holds the
studies behind these defaults.

SemiDDE supports the same maximum-width CSP initialization while retaining
its semiparametric pseudo-observation update.  `D` fixes the hierarchy depth;
the fitted active widths are returned in `result.K_list`:

```python
result = fit(
    Y_continuous,
    model="SemiDDE",
    D=2,
    dimension_method="csp",
    t_spike=[0.04, 0.08],
    temp=0.8,
    tau=0.3,
    C=1,
    tol=0.005,
    max_iter=50,
    random_state=123,
)
```

Run `scripts/evaluate_semidde_csp_recovery.py` for checkpointed recovery
experiments under nonlinear continuous margins. Its primary summary columns
are the exact dimension-shrinkage recovery rate and the mean L1 distance
between fitted and true width vectors.

Normal CSP has the copula kernel only.  Bernoulli and Poisson CSP provide
three MCEM backends.  ``"mcem"`` retains the original simultaneous MCEM updates,
``"mcem_reinitialize"`` activates CSP layer by layer and rebuilds each waiting
parent, and ``"mcem_cap_only"`` uses the same staged schedule while preserving
eligible parent states and only enforcing the child-implied structural cap.
Bernoulli and Poisson CSP select ``"mcem"`` by default when ``csp_backend``
is omitted; the staged backends remain explicit options.

Bernoulli DDE uses the same fixed and CSP interfaces, requires a binary 0/1
matrix, and has no `gamma` dispersion parameter. For fixed-width parametric
DDEs, `tol` is the RMSE of the postprocessed parameter update and defaults to
`0.005`:

```python
result = fit(
    Y_binary,
    method="DDE",
    family="Bernoulli",
    D=2,
    dimension_method="fixed",
    K_list=[6, 2],
    pen_list=[0.1, 0.1],
    tau=0.3,
    C=1,
    tol=0.005,
)
```

For a lightweight fixed-width convergence trace, pass `keep_trace=True`.
The returned `result.diagnostics["trace"]` records per-sweep update RMSEs,
E/M-step timings, and accumulator sizes. It does not retain coefficient
matrices or latent draws, so tracing does not scale with parameter-history
storage.

Bernoulli CSP uses the simultaneous MCEM path with a persistent systematic
coordinate-Gibbs transition: its
`C` draws are consecutive sweeps, and the last draw becomes the next MCEM
state.  The transition respects CSP stage caps and the current annealing
temperature.  For simultaneous MCEM with three or more layers, its automatic
defaults delay irreversible CSP shrinkage for 10 iterations when at least 20
outer iterations are requested, and scale the CSP penalty by
`max(1, (N / 1000) ** (2 / 3))`.  Supplying
`mcem_shrinkage_burn` or `mcem_penalty_strength` overrides the corresponding
automatic value.  Two-layer Bernoulli CSP retains its established settings.

Poisson DDE uses a canonical log link, requires finite nonnegative integer
counts, and has no `gamma` dispersion parameter. Fixed-width estimation
supports any positive number of latent layers:

```python
result = fit(
    Y_counts,
    method="DDE",
    family="Poisson",
    D=3,
    dimension_method="fixed",
    K_list=[9, 3, 1],
    pen_list=[0.1, 0.1, 0.1],
    tau=0.3,
    C=1,
    tol=0.005,
)
```

Poisson CSP uses the same maximum-width initialization and cumulative
shrinkage state as the other parametric families. Its observed M-step solves
the tempered Poisson likelihood with CSP-weighted L1 penalties by bounded
backtracking FISTA, while its latent E-step uses persistent systematic Gibbs:

```python
result = fit(
    Y_counts,
    method="DDE",
    family="Poisson",
    D=3,
    dimension_method="csp",
    t_spike=[0.01, 0.03, 0.005],
    temp=0.9,
    tau=0.3,
    C=1,
    random_state=123,
)
```

Run `scripts/evaluate_poisson_csp_recovery.py` for checkpointed dimension
recovery experiments. Its primary summary columns are the exact shrinkage
recovery rate and the mean L1 distance between fitted and true width vectors.

## Hyperparameter selection

`select_hyperparameters` accepts the same `method`, `family`, `D`, and
`dimension_method` path choices as `fit`. Use `criterion="ebic"` for a
full-data information criterion or `"bic"`/`"loglik"` for K-fold scores.
CopDDE also supports `"pmse"`, which compares simulated and observed data
with a cross-validated random forest. Fixed DDE and SemiDDE use `pen_list`;
CSP paths use `t_spike` and `temp`. DDE and SemiDDE CSP derive maximal widths
from `D`, while CopDDE can derive them or accept `K_list`.

Supply explicit candidate rows or a mapping of axes. `per_layer` crosses each
layer's values independently, including all interactions with `tau` and
`temp`:

```python
selected = select_hyperparameters(
    Y_binary, method="DDE", family="Bernoulli", D=2,
    dimension_method="fixed", criterion="ebic",
    fixed_hp={"K_list": [6, 2]},
    candidates={
        "pen_list": {"per_layer": [[0.1, 0.2], [0.2, 0.4]]},
        "tau": [0.2, 0.3],
    },
    base_fit_kwargs={"max_iter": 50, "C": 1},
    random_state=123,
)
table = selected.diagnostics["cv_table"]  # eight fitted candidates
best = selected.diagnostics["hyperparameter_selection"].best_candidate
```

When `candidates` is omitted, `auto_grid` crosses layer-specific penalty
exponents and thresholds; CSP defaults cross all layer-specific spike choices.
`auto_grid_copula_csp` also accepts `temp_values` and `tau_values` to build
the complete grid used in a specific analysis. Every evaluated row records its
score and elapsed fit/score times. The selected model is fitted on all data
when `refit=True` (the default). For larger CopDDE grids, set `n_jobs` to
parallelize the independent pMSE repeats; `n_repeats` and
`pmse_n_estimators` control the accuracy and cost of that score.
