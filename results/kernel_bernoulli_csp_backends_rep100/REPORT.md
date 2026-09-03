# Bernoulli CSP kernel backends: paired 100-replicate test

## Design

This experiment applies the same D2/D3 recovery design used for the
standalone Bernoulli CSP comparison to the three package-kernel backends:

- `mcem`: simultaneous MCEM;
- `mcem_reinitialize`: stagewise MCEM with parent reinitialization;
- `mcem_cap_only`: stagewise MCEM with child-implied caps and preserved state.

All backends use the same data and fit seed within each replicate.

| Setting | D2 | D3 |
|---|---:|---:|
| N | 500 | 500 |
| Replicates | 100 | 100 |
| True K | `(6, 2)` | `(18, 6, 2)` |
| Maximal K | `(9, 3)` | `(21, 7, 2)` |
| Penalty strength | 2 | 1 |
| Iterations | 40 | 50 |

Shared settings are `base_seed=20261301`, `C=1`, `tau=0.3`, temperature
`0.9`, data initialization, optimizer budget 50, SA burn 3, explicit
shrinkage burn 0, and convergence tolerance 0. Stagewise backends use the
current kernel defaults for transition timing and deeper-layer freezing.
Two workers were used for each concurrently running backend.

## Completion and numerical checks

- 600/600 fits completed successfully; no errors or nonfinite results.
- Every fit used its full iteration budget, as intended with tolerance 0.
- Probability saturation was zero for every backend/scenario aggregate.
- D2 stagewise fits all made their one required transition.
- D3 `mcem_reinitialize` made both transitions in 98/100 fits;
  `mcem_cap_only` did so in 97/100 fits.
- Data and fit seeds match exactly across all three backends.

## Aggregate results

### D2

| Backend | Mean K | Exact K | Mean dimension L1 | Under | Over | Graph accuracy | Latent accuracy | Coverage-adjusted latent | Log loss | Seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mcem` | `(5.79, 1.98)` | 78% | 0.25 | 22% | 1% | 0.9208 | 0.8675 | 0.8580 | 0.48423 | 0.570 |
| `mcem_reinitialize` | `(5.78, 1.78)` | 78% | 0.44 | 22% | 0% | 0.9059 | 0.8598 | 0.8368 | 0.48435 | 0.432 |
| `mcem_cap_only` | `(5.78, 1.64)` | 64% | 0.58 | 36% | 0% | 0.8898 | 0.8498 | 0.8164 | 0.48445 | 0.435 |

Exact-width 95% Wilson intervals are 68.9--85.0% for `mcem` and
`mcem_reinitialize`, and 54.2--72.7% for `mcem_cap_only`.

Layer-wise exact rates are:

- `mcem`: K1 79%, K2 96%;
- `mcem_reinitialize`: K1 78%, K2 78%;
- `mcem_cap_only`: K1 78%, K2 64%.

The reinitialization backend has the same aggregate exact rate as simultaneous
MCEM, but not strictly better recovery: relative to `mcem`, mean dimension L1
increases by 0.19 (paired 95% CI 0.10--0.28), graph accuracy falls by 1.49
percentage points, and coverage-adjusted latent accuracy falls by 2.13 points.
It is about 24% faster. Exact outcomes differ in only two paired replicates
(one win and one loss).

Cap-only is materially worse than both alternatives. Relative to `mcem`, its
exact rate drops from 78% to 64% (paired McNemar p=0.00052), dimension L1 rises
by 0.33, graph accuracy falls by 3.10 points, and coverage-adjusted latent
accuracy falls by 4.17 points. Its runtime is essentially the same as
reinitialization.

### D3

| Backend | Mean K | Exact K | Mean dimension L1 | Under | Over | Graph accuracy | Latent accuracy | Coverage-adjusted latent | Log loss | Seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mcem` | `(16.01, 5.12, 2.00)` | 1% | 2.91 | 99% | 2% | 0.9111 | 0.7855 | 0.7558 | 0.42922 | 2.717 |
| `mcem_reinitialize` | `(15.53, 4.95, 1.03)` | 0% | 4.49 | 100% | 0% | 0.8991 | 0.7729 | 0.7124 | 0.43227 | 2.378 |
| `mcem_cap_only` | `(15.54, 4.70, 1.04)` | 0% | 4.72 | 100% | 0% | 0.8986 | 0.7786 | 0.7127 | 0.43245 | 2.454 |

Layer-wise exact rates are:

- `mcem`: K1 3%, K2 27%, K3 100%;
- `mcem_reinitialize`: K1 0%, K2 1%, K3 3%;
- `mcem_cap_only`: K1 0%, K2 1%, K3 4%.

The rare K3=2 results in the stagewise backends mostly correspond to fits that
did not reach the final transition, not successful top-layer selection.

Relative to `mcem`, reinitialization increases mean dimension L1 by 1.58
(paired 95% CI 1.38--1.78), lowers graph accuracy by 1.20 percentage points,
lowers coverage-adjusted latent accuracy by 4.33 points, and raises conditional
log loss by 0.00305. It saves only about 12.5% runtime.

Cap-only increases dimension L1 by 1.81 (paired 95% CI 1.61--2.01), lowers
graph accuracy by 1.25 points, lowers coverage-adjusted latent accuracy by
4.31 points, and raises log loss by 0.00323. It saves about 9.7% runtime.
Neither stagewise backend beats `mcem` on dimension L1 in any replicate:
reinitialization is tied in 13 and worse in 87; cap-only is tied in 9 and worse
in 91.

## Interpretation

1. D2 is adequately recovered by simultaneous MCEM and reinitialization, but
   simultaneous MCEM retains substantially better graph and latent recovery.
2. Cap-only introduces additional upper-layer underselection even when K1 is
   correct; it should not be the preferred D2 backend under this protocol.
3. D3 remains a first-layer selection problem for all three kernels. The
   simultaneous path underselects K1 by about two variables on average, but
   preserves the true maximal K3.
4. Stagewise locking converts that lower-layer error into a hierarchy-wide
   failure: almost every completed staged fit ends at K3=1.
5. Unlike the standalone implementation, the optimized simultaneous kernel is
   already fast. The modest stagewise speed saving does not compensate for the
   statistical loss.

Under this matched explicit protocol, `mcem` is the preferred current kernel
backend. The next development target should be first-stage shrinkage/transition
calibration for D3, followed by uncertainty-aware cap propagation. Increasing
upper-stage sophistication cannot repair a K1 estimate already frozen near
15--16.

## Files

Each backend/scenario directory contains `fits.csv`, `summary.csv`,
`metadata.json`, and its automatically generated `REPORT.md`. The detailed
per-fit traces in `fits.csv` permit exact paired follow-up analyses.
