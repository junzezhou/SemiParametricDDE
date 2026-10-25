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
its semiparametric pseudo-observation update.  SemiDDE CSP runs on the same
copula kernel (`csp_backend="copula"`, the default), with the Normal-family
engine defaults and the precision-weighted `g` step after every committed
iteration; `csp_backend="mcem"` selects the former MCEM path.  `D` fixes the
hierarchy depth; the fitted active widths are returned in `result.K_list`:

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

Normal CSP has the copula kernel only.  Bernoulli and Poisson CSP also
default to the copula kernel (`csp_backend="copula"`) with a logistic or
log-link observed layer and their own defaults: the sequential mean-field
transition (`csp_latent_transition="sequential_mean_field"`), a sqrt(N)
penalty in every layer (`csp_penalty_scaling_power=0.5`,
`csp_penalty_scaling_layers="all"`), hard thresholding of every layer
(`csp_threshold_rule="all"`), the dense marginal-likelihood deletion test for
the deepest layer (`csp_top_prune="bic_dense"`), and 100 iterations without
early stop (`max_it=100`, `tol=0`); `Results/csp_redesign/bench_v2` holds the
20-replicate N-trend study behind these defaults.  The former MCEM backends
stay available: ``"mcem"`` retains the original simultaneous MCEM updates,
``"mcem_reinitialize"`` activates CSP layer by layer and rebuilds each waiting
parent, and ``"mcem_cap_only"`` uses the same staged schedule while preserving
eligible parent states and only enforcing the child-implied structural cap.

Bernoulli DDE uses the same fixed and CSP interfaces, requires a binary 0/1
matrix, and has no `gamma` dispersion parameter. For fixed-width parametric
DDEs, `tol` is the RMSE of the postprocessed parameter update and defaults to
`0.005`.  Fixed-width Bernoulli fits start from the anchor latent-class
initialization (`fixed_init_binarize="anchor_lca"`) and run ten unit-step
burn-in sweeps (`sa_burn_in=10`); see the fixed-width controls below:

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

## Visualization

Two interfaces summarize a fitted model (`fit` or `select_hyperparameters`
result).  `plot_weights` draws the estimated weight matrices from the observed
layer upward (one heatmap per layer, or a chosen `layer`); for CSP fits the
columns are labelled by their original latent indices.  `get_anchors` returns
the observed variables that anchor each shallow-layer latent feature, ranked
by the relative-loading score `r_jk = max(min_{l != k}(b_jk - b_jl), 0)`:

```python
from dde.visualization import plot_weights, get_anchors

figures = plot_weights(result, variable_names=Y.columns, output_dir="plots")
anchors = get_anchors(result, n_anchors=3, variable_names=Y.columns)
anchors[anchors["latent_feature"] == 1]
```

The earlier multi-panel report bundle (`visualization_report` and the
model-summary, latent-profile and theme-table plots) remains available from
`dde.visualization.legacy`.

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

Fixed-width fits use three controls on top of the MATLAB SAEM schedule
(`Results/dde_poisson_fixed_recon/REPORT.md`,
`Results/bernoulli_fixed_recon/REPORT.md` and
`Results/dde_normal_fixed_recon/REPORT.md` hold the studies behind them):
`sa_burn_in=10` runs the first ten sweeps with step size one before the `1/t`
stochastic approximation starts, so draws made from the initialization do not
keep a lasting weight in the averaged objectives; `fixed_init_binarize` reads
the initial latent states off each column's anchor variables, with
two-component mixtures on the observed layer (`"anchor_mixture"`: Poisson
mixtures for Poisson, Gaussian mixtures for Normal, whose deeper layers use
the latent class model) or a two-class latent class model over the binary
anchors (`"anchor_lca"`, Bernoulli), which recovers the column prevalence
that thresholding the projected working response at zero cannot (a rare
latent column otherwise starts at about 50 % and drifts into an always-on,
intercept-absorbing solution, or collapses while neighbouring columns absorb
its signal); and, for Poisson and Normal, `fixed_warm_start="layerwise"`
estimates the observed layer first with a flat latent prior, then the deeper
layers as a Bernoulli DDE on the resulting latent states, and only then runs
the full SAEM (binary data keep `fixed_warm_start="none"`: a flat-prior fit
of one binary layer loses the layers above it).  In the Bernoulli D=2..4
paper design the burn-in and the anchor initialization together lift the
layer-2 graph accuracy at D=3 from 0.67-0.71 to 0.79-0.89 and halve the
deep-layer coefficient RMSE, with the same cost per fit.  In the Normal D=4
paper design the three controls lift the layer-3/4 graph accuracy at
N=16000 from 0.55/0.70 to 0.93/0.96 at about three times the cost per fit;
at N=500 the mean deep-layer RMSE is higher than with the plain kernel (a
few fits lock a deep row with a rare child into a near-deterministic link
during the unit-step sweeps).  The Poisson observed-layer M-step keeps the
MATLAB per-row L-BFGS-B (`fixed_observed_solver="rowwise"`); `"active_set"`
solves each row's TLP problem exactly on its support with a KKT screening of
the zero coordinates and `"hybrid"` uses it after the burn-in only; both are
cheaper for very wide layers but lose observed-layer accuracy on the D=4
design, so they are options rather than defaults.  Pass `sa_burn_in=0,
fixed_init_binarize="projection", fixed_warm_start="none"` for the plain
MATLAB-style kernel; `result.diagnostics["warm_start"]` records the
warm-start stage iterations.

Fixed-width SemiDDE fits a Normal DDE to pseudo-observations `X = g(Y)` and
re-estimates the monotone transformations between SAEM fits
(`Results/semidde_fixed_recon/REPORT.md` holds the study behind the
defaults):

```python
result = fit(
    Y_continuous,
    model="SemiDDE",
    D=3,
    dimension_method="fixed",
    K_list=[18, 6, 2],
    pen_list=[N ** 0.25] * 3,
    tau=0.3,
    C=1,
    tol=0.01,
)
```

Each parent class `s` of `Y_j` (the nonzero slopes of row `j` of `B_1`)
gives its own estimate `mu_j(s) + sigma_j Phi^{-1}(F_hat_s(y))`; the default
`g_estimator="precision"` averages them pointwise with weights equal to
their asymptotic precision at the current pseudo-observation, projects the
result onto increasing functions of `Y_j` and standardizes it.
`g_estimator="average"` is the manuscript's estimator
`sum_s pi(s) Phi^{-1}(F_hat_s(y))`, which needs every class's CDF where the
class has no data and keeps an error that does not shrink with N.  The
default `schedule="staged"` alternates a one-layer SAEM of the observed
layer with the transformation update until the relative change of `X` is
below `g_tol=0.002` (at most `max_outer_iter=50` updates), then runs the
Normal fixed kernel's layer-wise warm start and a full SAEM on the final
`X`; `schedule="nested"` keeps the manuscript's outer loop of full inner
SAEM fits.  The initializer uses the anchor-mixture binarization
(`init_binarize`) and the SAEM fits use `sa_burn_in=10`, as for the Normal
family.  `result.diagnostics` records `g_iterations`, `g_converged`,
`g_relative_history` and the warm-start iterations.  Pass
`schedule="nested", g_estimator="average", sa_burn_in=0,
init_binarize="projection", g_tol=0.01` for the earlier kernel.

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
