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

The arguments of `fit` follow Table 1 of the package manuscript:

| Argument | Purpose |
|---|---|
| `model` | `"DDE"` (parametric), `"SemiDDE"` or `"CopDDE"` |
| `D` | number of binary latent layers |
| `family` | `"Normal"`, `"Bernoulli"` or `"Poisson"` (`model="DDE"` only) |
| `dimension_method` | `"fixed"` (widths given by `K_list`) or `"csp"` (widths inferred by cumulative shrinkage) |
| `K_list` | layer widths `[K_1, ..., K_D]`, shallowest first; for `"csp"` optional maximal widths |
| `G_list` | optional binary graphs of the layers (`"fixed"`), see below |
| `pen_list` | length-`D` truncated-lasso penalties (`"fixed"`) |
| `t_spike`, `temp` | length-`D` CSP spike scales and the initial annealing temperature (`"csp"`) |
| `C` | latent draws per stochastic E-step (default 1) |
| `tol` | convergence tolerance |
| `tau` | truncation threshold of the coefficients |
| `max_it` | maximum number of iterations |

`method=` (the former name of `model`) still works with a
`DeprecationWarning`, and the kernel spellings `max_iter`, `it` and
`max_inner_iter` are accepted for `max_it`.  Omitting `dimension_method`
warns (`FutureWarning`) and falls back to `"fixed"` (CopDDE: `"csp"`, or
`"fixed"` with `G_list`).  Every fit returns a `DDEResult` with the fields
`model` (`"DDE"`, `"SemiDDE"` or `"CopDDE"`), `prop`, `B_list`, `gamma`,
`A_list`, `K_list`, `transformed`, `diagnostics` (including `iterations`,
`active_by_layer` and `n_active_edges`) and `raw` (a dict of the kernel's
outputs).

A fixed-width Normal DDE supplies its latent widths and layer penalties.  A
CSP fit may omit `K_list`; the package then initializes the recursive maximal
widths `floor(J / 3), floor(floor(J / 3) / 3), ...` and returns the
CSP-active widths in `result.K_list`:

```python
result = fit(
    Y,
    model="DDE",
    family="Normal",
    D=3,
    dimension_method="csp",
    t_spike=[0.01, 0.03, 0.005],  # Laplace spike scales, not rates
    temp=0.9,
    tau=0.3,
    C=1,
    tol=0.005,
    max_it=50,
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
iteration (the former MCEM path was removed; git tag `csp-mcem-last`).  `D` fixes the
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
    max_it=50,
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
20-replicate N-trend study behind these defaults.  Bernoulli CSP also holds
the columns in the CUSP slab for the first fifth of the budget
(`csp_shrinkage_burn`: 20 of 100 iterations, in every layer except the
deepest layer of a model with three or more layers); without it the rarest
latents are pruned before their states settle and the width error does not
fall with N (`Results/dde_modification`).  The former MCEM CSP
backends (``"mcem"``, ``"mcem_reinitialize"``, ``"mcem_cap_only"``) were
removed; the git tag `csp-mcem-last` holds the last version with them.

Bernoulli DDE uses the same fixed and CSP interfaces, requires a binary 0/1
matrix, and has no `gamma` dispersion parameter. For fixed-width parametric
DDEs, `tol` is the RMSE of the postprocessed parameter update and defaults to
`0.005`.  Fixed-width Bernoulli fits start from the anchor latent-class
initialization (`fixed_init_binarize="anchor_lca"`) and run ten unit-step
burn-in sweeps (`sa_burn_in=10`); see the fixed-width controls below:

```python
result = fit(
    Y_binary,
    model="DDE",
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

Bernoulli and Poisson CSP share the copula kernel described above.  Its
other latent transitions stay available as options: the CopDDE synchronous
mean-field draws (`csp_latent_transition="synchronous_mean_field"`) and a
persistent systematic coordinate-Gibbs chain
(`csp_latent_transition="systematic_coordinate_gibbs"`).  Further options are
the slab-start burn-in (`csp_shrinkage_burn`, an integer or one value per
layer) and a stopping rule that also waits for stable widths
(`csp_convergence_width_window`, with `tol>0`).

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

`visualization_report` writes the standard bundle of diagnostic plots of a
SemiDDE or CopDDE fit (model summary, weight heatmaps, top loadings, item
clusters, latent profiles and, for CopDDE, an optional theme table) and
returns their paths keyed `model_summary`, `B1_heatmap`, ...:

```python
from dde.visualization import visualization_report

paths = visualization_report(result, output_dir="plots", prefix="semi",
                             feature_names=item_names)
```

The individual report helpers remain available from
`dde.visualization.legacy`.

Poisson DDE uses a canonical log link, requires finite nonnegative integer
counts, and has no `gamma` dispersion parameter. Fixed-width estimation
supports any positive number of latent layers:

```python
result = fit(
    Y_counts,
    model="DDE",
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
    model="DDE",
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

## Fixed (user-specified) graph

When the graph is known, as for a Q-matrix in a cognitive assessment, pass it
as `G_list` and only the coefficients on the graph are estimated (the
confirmatory analysis of the paper's TIMSS application). `G_list[0]` is the
`J x K_1` observed-layer graph and `G_list[d]` the `K_d x K_{d+1}` graph above
it, binary with a nonempty column for every latent variable; the widths follow
from the graphs:

```python
result = fit(
    Y_log_times,
    model="DDE",
    family="Normal",
    dimension_method="fixed",
    G_list=[Q, np.ones((7, 1))],
)
```

The fixed layers are neither penalized nor thresholded (`tau` and `pen_list`
may be omitted when every layer is fixed), and their coefficients outside the
graph are exactly zero. The start is graph-anchored: each latent column is
read off its children with the fewest parents, oriented so that the value one
raises their means. With every layer fixed and no layer wider than 10, the
default `graph_algorithm="auto"` runs an exact EM over each layer's `2**K_d`
states, which reaches the maximum-likelihood estimate on the graph; it keeps
the best of `n_init=10` starts (the graph-anchored one and random starts on
the graph, drawn from `random_state`), because the likelihood can be
multimodal on real data. Wider layers, or a `G_list` that fixes only the
shallowest layers (`K_list` then gives the free widths, which keep the
penalized estimation), use the fixed-width SAEM kernel with the graph held
(`graph_algorithm="saem"`, with `C=1` by default). SemiDDE (`model="SemiDDE"`,
fixed mode) accepts `G_list` with the SAEM path. The paper-format study is
`scripts/evaluate_fixed_graph_paper.py` and the TIMSS replication
`scripts/realdata_timss_fixed_graph.py`.

CopDDE takes `G_list` the same way:

```python
result = fit(X_counts, model="CopDDE", dimension_method="fixed",
             G_list=[G1, G2], random_state=0)
```

The fixed layers are fitted by the rank-likelihood kernel (`get_SAEM_RL_D`)
on their graph, unpenalized and unthresholded, with coordinate-Gibbs latent
draws, as in the CopDDE code's fixed-graph refit (`get_SAEM_RL_G.m`); the
defaults are `C=1`, `max_it=50`, `tol=0` and `temp=1`. The start matches the
spectral estimate's columns to the graph, which needs no child with a single
parent. With every layer fixed and deep layers no wider than 10, the layers
above the first are then refitted by exact EM on the first layer's states
after `graph_warmup_it=10` kernel iterations, keeping the best of `n_init=20`
starts. A binary graph fixes the labels of the latent variables but not their
0/1 coding, and variables with identical graph columns (and rows above) are
interchangeable. The paper-format study is
`scripts/evaluate_copdde_fixed_graph.py`.

## Hyperparameter selection

`select_hyperparameters` takes the same `model`, `family`, `D` and
`dimension_method` as `fit`, a `candidates` grid, and passes every other
keyword (`K_list`, `G_list`, `C`, `tol`, `max_it`, ...) to `fit` for every
candidate.  For fixed widths the grid holds `pen_list` and `tau`; for CSP,
`t_spike`, `tau` and `temp`.  A flat list of `pen_list` or `t_spike` values
holds one candidate per value, used in every layer; nested lists or
`{"per_layer": [[...], ...]}` give layer-specific vectors.  Every Cartesian
combination is fitted:

```python
selected = select_hyperparameters(
    Y,
    D=2,
    model="DDE",
    family="Normal",
    dimension_method="fixed",
    K_list=[10, 3],
    candidates={
        "pen_list": [.1, .05, .02, .01],
        "tau": [.1, .2, .3],
    },
    C=1,
    tol=.01,
    max_it=25,
)
selected.K_list                                   # [10, 3]
selected.raw["selection_method"]                  # 'fixed'
selected.raw["selected_hyperparameters"]          # e.g. {'pen_list': [0.05, 0.05], 'tau': 0.3}
table = selected.diagnostics["cv_table"]          # one row per candidate
```

The criterion follows the model.  DDE and SemiDDE candidates are compared by
the extended BIC of the full-data fit (`criterion="ebic"`, the default; the
`ebic` column of the table is on the usual smaller-is-better scale, while
`mean_score` is its negative per-observation value).  CopDDE candidates are
compared by predictive synthetic-data utility (`criterion="pmse"`, the
default): `n_repeats=25` synthetic data sets are drawn from each fit, pooled
with the observed data, and a random forest scored by `n_splits=5`-fold
cross-validation measures the propensity-score MSE; the smallest mean pMSE
wins.  `criterion="bic"` or `"loglik"` selects by `n_splits`-fold
cross-validated likelihood instead.

When `candidates` is omitted, `auto_grid` crosses layer-specific penalty
exponents and thresholds; CSP defaults cross all layer-specific spike choices.
`auto_grid_copula_csp` also accepts `temp_values` and `tau_values` to build
the complete grid used in a specific analysis. Every evaluated row records its
score and elapsed fit/score times. The selected model is fitted on all data
when `refit=True` (the default). Set `n_jobs` to evaluate candidates in
parallel worker processes; `n_repeats` and `pmse_n_estimators` control the
accuracy and cost of the pMSE score.  `fixed_hp` and `base_fit_kwargs` remain
accepted as older ways to pass the fixed `fit` arguments.
