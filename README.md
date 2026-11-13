# dde

`dde` is a Python package for fitting and interpreting **Deep Discrete Encoders
(DDEs)**: identifiable deep generative models in which `D` layers of binary
latent variables, `A(D) -> ... -> A(1)`, generate the observed variables `Y`.
Every layer is a sparse regression on the layer above, so a fitted model is a
set of sparse weight matrices `B(1), ..., B(D)` that can be read as a
multilevel network of latent features.

The package implements three formulations behind one interface:

| `model` | observed data | observation model |
|---|---|---|
| `"DDE"` | continuous, binary or count | parametric: `family="Normal"`, `"Bernoulli"` or `"Poisson"` |
| `"SemiDDE"` | continuous | a latent Normal DDE seen through unknown monotone transformations of each variable |
| `"CopDDE"` | continuous, discrete, ordinal or mixed | a Gaussian-copula DDE fitted with the rank likelihood |

Each model can be fitted with **fixed** latent widths (`dimension_method="fixed"`,
widths given in `K_list`) or with widths inferred by a **cumulative shrinkage
process** (`dimension_method="csp"`).  The package also provides
hyperparameter selection, a common result object and plots of the estimated
weights and anchor variables.

The methods are described in the package manuscript (in preparation, see
[Citation](#citation)).

## Installation

`dde` needs Python 3.9 or later (it is tested with Python 3.12) and
NumPy, SciPy, scikit-learn and joblib.  Install it from GitHub:

```bash
pip install "dde @ git+https://github.com/junzezhou/SemiParametricDDE.git"
```

or, to work on the code, clone the repository and install it in editable mode:

```bash
git clone https://github.com/junzezhou/SemiParametricDDE.git
cd SemiParametricDDE
pip install -e ".[viz,dev]"
```

The optional extras are `viz` (matplotlib and pandas, for the plots),
`init` (factor-analyzer, an optional initializer) and `dev` (pytest).

## Quick start

Simulate a two-layer Normal DDE with 18 observed variables and fit it with the
true widths:

```python
import numpy as np
from dde import fit
from dde.identifiability import build_true_params
from dde.simulation import generate_X_normal

np.random.seed(0)
prop, B_true, G_true, gamma = build_true_params([6, 2], 18, 4.0)  # widths (6, 2), J = 18
Y, A_true = generate_X_normal(2000, prop, B_true, gamma)

result = fit(Y, model="DDE", family="Normal", D=2, K_list=[6, 2],
             dimension_method="fixed", verbose=False)

print(result.K_list)                    # [6, 2]
print(result.B_list[0].shape)           # (18, 7): intercept + 6 loadings per variable
print(result.diagnostics["iterations"])
```

`build_true_params` and the `dde.simulation` generators produce the designs
used in the package's simulation studies; with real data only `Y` is needed.

To let the model choose the widths, use `dimension_method="csp"` and omit
`K_list`.  The fit then starts from the maximal identifiable widths
`floor(J/3), floor(floor(J/3)/3), ...` and switches unneeded latent columns
off.  Here 12 pure-noise variables are appended, so the fit starts from
widths (10, 3):

```python
Y_wide = np.column_stack([Y, np.random.standard_normal((Y.shape[0], 12))])
result_csp = fit(Y_wide, model="DDE", family="Normal", D=2,
                 dimension_method="csp", random_state=0, verbose=False)
print(result_csp.diagnostics["K_max_list"], "->", result_csp.K_list)
```

## The `fit()` interface

The arguments of `fit` follow Table 1 of the package manuscript:

| Argument | Purpose |
|---|---|
| `model` | `"DDE"` (parametric), `"SemiDDE"` or `"CopDDE"` |
| `D` | number of binary latent layers |
| `family` | `"Normal"`, `"Bernoulli"` or `"Poisson"` (`model="DDE"` only) |
| `dimension_method` | `"fixed"` (widths given by `K_list`) or `"csp"` (widths inferred by cumulative shrinkage); required |
| `K_list` | layer widths `[K_1, ..., K_D]`, shallowest first; for `"csp"` optional maximal widths |
| `G_list` | optional binary graphs of the layers (`"fixed"`), see [Known graphs](#known-graphs-g_list) |
| `pen_list` | truncated-lasso penalty of each layer (`"fixed"`) |
| `t_spike`, `temp` | CSP spike scales per layer and the initial annealing temperature (`"csp"`) |
| `C` | latent draws per stochastic E-step (default 1) |
| `tol` | convergence tolerance on the relative parameter change |
| `tau` | hard threshold applied to the estimated loadings |
| `max_it` | maximum number of iterations |
| `verbose` | print the iteration log (default `True`) |

Hyperparameters that are not given take these defaults (the same values
`select_hyperparameters` uses for what it does not tune):

| | fixed widths | CSP |
|---|---|---|
| DDE, SemiDDE | `pen_list = N**(1/4)` in every layer, `tau = 0.3`, `tol = 0.005`, `max_it = 50` | `t_spike = (0.03, 0.05)` for `D = 2`, `(0.01, 0.03, 0.005)` for `D = 3`; `temp = 0.7` (`D <= 2`) or `0.9`; `tau = 0.3` |
| CopDDE | `pen_list = N**(1/4)`, `tau = max(0.3, 3 N**-0.3)`, `tol = 0.005`, `temp = 1` | as above, with `tau = max(0.3, 3 N**-0.3)` |

`result.diagnostics["fit_arguments"]` records the hyperparameters a fit used and
`result.diagnostics["default_arguments"]` the ones it filled in.

**Reproducibility.**  Every fit draws from NumPy's global random generator, so
`np.random.seed(...)` before `fit` (or `select_hyperparameters`) reproduces it.
CSP and CopDDE fits also accept `random_state`, which seeds their starting
values; it does not replace `np.random.seed`.  The fixed-width DDE and SemiDDE
paths reject `random_state`.  Bit-for-bit agreement across machines also needs
single-threaded linear algebra (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`).

**The result object.**  Every fit returns a `DDEResult` with the fields

* `model` (`"DDE"`, `"SemiDDE"` or `"CopDDE"`) and `K_list` (the fitted widths);
* `B_list`: one matrix per layer; `B_list[0]` is `J x (1 + K_1)` and
  `B_list[d]` is `K_d x (1 + K_{d+1})`, with the intercepts in column 0;
* `prop`: the probabilities of the top-layer latent variables;
* `gamma`: observation variances (Normal DDE and SemiDDE);
* `A_list`: the final latent states, one `N x K_d` array per layer;
* `transformed`: the latent Gaussian pseudo-observations (SemiDDE and CopDDE);
* `diagnostics` (iterations, convergence, defaults used, `active_by_layer`,
  `n_active_edges`, ...) and `raw` (the kernel's own outputs).

For CSP fits, `B_list`, `A_list` and `prop` keep only the active latent columns;
`diagnostics["active_columns_original_index"]` maps them back to the starting
columns.

The former keyword `method=` (now `model`) still works with a
`DeprecationWarning`, and `max_iter`, `it` and `max_inner_iter` are accepted as
spellings of `max_it`.

## Models

The examples below continue the quick start (`Y`, `prop`, `B_true`, `gamma`
and `G_true` are defined there).

### Parametric DDE: binary and count data

Bernoulli DDEs take a 0/1 matrix and Poisson DDEs nonnegative integer counts;
neither has a variance parameter `gamma`.

```python
from dde.simulation import generate_X_bernoulli, generate_X_poisson

def logistic_layer(n_parent):
    # Lee & Gu (2026) design: three children per parent, logit intercepts -2/-4/-2
    slopes = np.zeros((3 * n_parent, n_parent))
    slopes[:n_parent] = slopes[n_parent:2 * n_parent] = 4 * np.eye(n_parent)
    slopes[2 * n_parent:] = 4 * np.eye(n_parent)
    for r in range(n_parent):
        slopes[2 * n_parent + r, (r + n_parent // 2) % n_parent] = 4 / 3
    intercept = np.r_[np.full(n_parent, -2.0), np.full(n_parent, -4.0), np.full(n_parent, -2.0)]
    return np.column_stack([intercept, slopes])

B_binary = [logistic_layer(6), logistic_layer(2)]   # binary data need larger logistic loadings
Y_binary, _, _ = generate_X_bernoulli(2000, prop, B_binary[0], B_binary[1])
fit_bernoulli = fit(Y_binary, model="DDE", family="Bernoulli", D=2, K_list=[6, 2],
                    dimension_method="fixed", verbose=False)

Y_counts, _ = generate_X_poisson(2000, prop, B_true, random_state=1)
fit_poisson = fit(Y_counts, model="DDE", family="Poisson", D=2, K_list=[6, 2],
                  dimension_method="fixed", verbose=False)
```

### SemiDDE: continuous data with unknown margins

SemiDDE assumes that each observed variable is an unknown increasing function
of a latent Gaussian variable that follows a Normal DDE.  It alternates
between fitting the Normal DDE to pseudo-observations `X = g(Y)` and
re-estimating the transformations `g`; `result.transformed` holds the final
pseudo-observations.

```python
Y_skewed = np.exp(Y)        # log-normal margins
fit_semi = fit(Y_skewed, model="SemiDDE", D=2, K_list=[6, 2],
               dimension_method="fixed", verbose=False)
print(np.corrcoef(fit_semi.transformed[:, 0], Y[:, 0])[0, 1])   # close to 1
```

### CopDDE: mixed, discrete or ordinal data

CopDDE treats the observed values only through their ranks within each
variable (the rank likelihood), so it accepts any mix of continuous, count,
ordinal and binary variables.  The loadings of the first layer are on the
scale of the latent Gaussian variables.

```python
from dde.simulation import generate_X_Cop_marg_emp_D

data = generate_X_Cop_marg_emp_D(2000, prop, np.full(18, 3.0), gamma, B_true,
                                 random_state=2)
Y_mixed = data["X"]         # zero-inflated Poisson, rounded gamma and Poisson columns
fit_cop = fit(Y_mixed, model="CopDDE", D=2, K_list=[6, 2],
              dimension_method="fixed", random_state=0, verbose=False)
```

`fit` does not accept `NaN` in `Y`.  For CopDDE, missing entries are handled
through the rank inputs: pass the within-variable ranks `R` (1, 2, ... for the
distinct observed values, `NaN` where missing) and the number of distinct
values `Rlevels`, together with a completed copy of `Y` that is used only for
the starting values:

```python
Y_missing = Y_mixed.copy()
Y_missing[np.random.default_rng(3).random(Y_missing.shape) < 0.05] = np.nan

R = np.full(Y_missing.shape, np.nan)
Rlevels = np.zeros(Y_missing.shape[1], dtype=int)
for j in range(Y_missing.shape[1]):
    observed = ~np.isnan(Y_missing[:, j])
    values, ranks = np.unique(Y_missing[observed, j], return_inverse=True)
    R[observed, j] = ranks + 1
    Rlevels[j] = len(values)
Y_start = np.where(np.isnan(Y_missing), np.nanmedian(Y_missing, axis=0), Y_missing)

fit_missing = fit(Y_start, model="CopDDE", D=2, K_list=[6, 2], dimension_method="fixed",
                  R=R, Rlevels=Rlevels, random_state=0, verbose=False)
```

### Known graphs (`G_list`)

When part of the structure is known, for example a Q-matrix linking test items
to skills, pass it as `G_list` (fixed widths only).  `G_list[0]` is the
`J x K_1` graph of the observed layer and `G_list[d]` the `K_d x K_{d+1}` graph
above it; `G = 1` marks an allowable connection and `G = 0` a coefficient held
at exactly zero.  The allowable coefficients are estimated by the same
penalized algorithm, so `pen_list` and `tau` can still remove one.  A
`G_list` shorter than `D` restricts the shallowest layers only, and `K_list`
gives the widths of the deeper layers.  All three models accept `G_list`.

```python
fit_graph = fit(Y, model="DDE", family="Normal", D=2, K_list=[6, 2],
                G_list=[G_true[0].astype(int)], dimension_method="fixed",
                verbose=False)
```

A graph that allows every connection restricts nothing and gives the same fit
as no graph.

## Hyperparameter selection

`select_hyperparameters` takes the same arguments as `fit` and a grid of
candidates, fits every candidate, scores it and refits the best one on all the
data.  For fixed widths the grid holds `pen_list` and `tau`; for CSP,
`t_spike`, `tau` and `temp`.  A flat list of `pen_list` or `t_spike` values gives
one candidate per value, used in every layer; nested lists give layer-specific
vectors.  `auto_grid` builds the default fixed-width grid (layer-specific
penalty exponents crossed with thresholds) and `auto_grid_copula_csp` the CSP
grid.

```python
from dde import select_hyperparameters, auto_grid

np.random.seed(0)
selected = select_hyperparameters(
    Y, model="DDE", family="Normal", D=2, K_list=[6, 2], dimension_method="fixed",
    candidates=auto_grid(Y.shape[0], D=2), random_state=0, verbose=False,
)
print(selected.raw["selected_hyperparameters"])
table = selected.diagnostics["cv_table"]          # one row per candidate
```

The criterion depends on the model:

* **DDE and SemiDDE**: the extended BIC of the full-data fit
  (`criterion="ebic"`, the default; smaller is better).  For SemiDDE the
  likelihood uses the transformation implied by the fit, `g_j = F_Zj^-1 o F_Yj`,
  with `F_Zj` the Gaussian-mixture margin of the fitted DDE.
* **CopDDE**: predictive synthetic-data utility (`criterion="pmse"`, the
  default).  `n_repeats=25` synthetic data sets are drawn from each fit, pooled
  with the observed data, and a random forest scored by 5-fold cross-validation
  gives the propensity-score mean squared error; the smallest mean pMSE wins.
* `criterion="bic"` or `"loglik"` selects by cross-validated likelihood instead.

`n_jobs` evaluates candidates in parallel processes.

## Visualization

The plotting functions need the `viz` extra (matplotlib and pandas).
`plot_weights` draws the estimated weight matrices, one heatmap per layer (or a
chosen `layer`); for CSP fits the columns keep their original indices and a
hatched column counts the columns that were shrunk away.  `get_anchors` lists,
for every latent variable of a layer, the observed variables that anchor it,
ranked by the relative loading `r_jk = max(min_{l != k} (b_jk - b_jl), 0)`.

```python
from dde.visualization import plot_weights, get_anchors, visualization_report

names = [f"item{j + 1}" for j in range(Y.shape[1])]
figures = plot_weights(result, variable_names=names, output_dir="plots")
anchors = get_anchors(result, n_anchors=3, variable_names=names)
print(anchors.head())

paths = visualization_report(result, output_dir="plots", prefix="normal",
                             feature_names=names)
```

Variables are labelled by the column names of the data frame the model was
fitted on, otherwise `y1, y2, ...`; `variable_names` overrides them.  For CSP
fits, `latent_feature` in the anchor table is the original (pre-shrinkage)
column index.  `visualization_report` writes a standard bundle of plots (model
summary, weight heatmaps, top loadings, item clusters and latent profiles) and
returns their paths.

## Implementation notes

The estimators follow the package manuscript; these are the main
implementation choices and the options that change them.

* **Fixed widths.**  Penalized stochastic approximation EM with a
  truncated-lasso penalty and a hard threshold `tau`.  Latent states are drawn
  by systematic coordinate-Gibbs sweeps.  A fit stops when the relative change
  of the parameters falls below `tol` or after `max_it` iterations.
  * Initialization: a spectral estimate whose latent states are read off each
    column's anchor variables, by two-component mixtures
    (`fixed_init_binarize="anchor_mixture"`; Normal, Poisson, SemiDDE) or a
    two-class latent class model (`"anchor_lca"`; Bernoulli).
  * `sa_burn_in=10` unit-step sweeps run before the `1/t` step size starts.
  * Normal and Poisson fits warm-start layer by layer
    (`fixed_warm_start="layerwise"`).
  * `sa_burn_in=0, fixed_init_binarize="projection", fixed_warm_start="none"`
    gives the plain kernel without these three steps.
* **SemiDDE.**  The pseudo-observations start from the normal scores of the
  ranks (`pseudo_init="rank_normal"`; `"rank_spectral"` uses the CopDDE
  spectral start).  The transformations are re-estimated by a precision-weighted
  average over parent classes (`g_estimator="precision"`; `"average"` is the
  manuscript's estimator).  The default `schedule="staged"` updates the
  transformations with a one-layer fit until they settle (`g_tol=0.002`) and then
  fits the full model; `schedule="nested"` is the manuscript's outer loop of
  full fits.
* **CopDDE.**  The latent Gaussian variables are redrawn from the rank
  likelihood at every iteration.  Fixed-width fits run at `temp=1` with
  coordinate-Gibbs latent draws (`latent_transition="synchronous_mean_field"`
  restores the draws of the original CopDDE code).
* **CSP.**  A cumulative shrinkage (CUSP) prior on the columns of each layer,
  starting from the maximal identifiable widths; the temperature rises from
  `temp` to 1 by 0.01 per iteration.  Normal, SemiDDE and CopDDE fits stop when
  every layer's relative change, averaged over the last seven iterations, is at
  most `tol` (default 0.005, at most 50 iterations).  Bernoulli and Poisson fits
  run 100 iterations and use sequential mean-field latent transitions, a
  `sqrt(N)` penalty scaling, thresholding of every layer and a deletion test for
  the top layer; Bernoulli fits also keep the columns in the slab for the first
  20 iterations (`csp_shrinkage_burn`).

## Repository layout

| Path | Contents |
|---|---|
| `src/dde/` | the package |
| `tests/` | the test suite (`pytest`) |
| `examples/package_demo.ipynb` | an executed walkthrough of the main functions |
| `scripts/` | simulation-study and real-data analysis scripts used to develop and evaluate the package; they are not installed with it |

Run the tests with `pytest` from the repository root.

## Citation

If you use `dde`, please cite the package manuscript (in preparation; the full
reference will be added here).

## License

MIT; see [LICENSE](LICENSE).
