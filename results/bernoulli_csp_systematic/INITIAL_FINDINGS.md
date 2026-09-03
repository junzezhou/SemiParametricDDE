# Bernoulli CSP initial development findings

## Baseline protocol

- Scenarios retain the paper's sparse Bernoulli `B_s` coefficients and add
  uninformative observed variables so the fitted CSP hierarchy is genuinely
  overspecified.
- D=2: `J=27`, true width `(6,2)`, maximal fitted width `(9,3)`.
- D=3: `J=64`, true width `(18,6,2)`, maximal fitted width `(21,7,2)`.
- `N = 500, 1000, 2000, 4000, 8000`; three independent replicates each.
- Default simultaneous `mcem`, data initialization, `C=1`, at most 50 outer
  iterations, 100 FISTA iterations, `tau=0.3`, initial temperature `0.9`.

## Baseline results

| D | N | Successful | Exact-width rate | Mean dimension L1 error | Mean graph accuracy | Mean latent accuracy | Mean seconds | Mean peak RSS MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 500 | 3/3 | 1.000 | 0.000 | 0.847 | 0.874 | 1.05 | 187.6 |
| 2 | 1000 | 3/3 | 1.000 | 0.000 | 0.874 | 0.861 | 2.18 | 193.0 |
| 2 | 2000 | 3/3 | 0.667 | 0.333 | 0.952 | 0.879 | 4.32 | 199.5 |
| 2 | 4000 | 3/3 | 0.667 | 0.667 | 0.981 | 0.883 | 7.82 | 222.9 |
| 2 | 8000 | 3/3 | 0.667 | 0.667 | 0.966 | 0.869 | 14.99 | 251.2 |
| 3 | 500 | 3/3 | 0.000 | 3.667 | 0.910 | 0.735 | 3.19 | 197.2 |
| 3 | 1000 | 3/3 | 0.000 | 1.667 | 0.928 | 0.773 | 6.57 | 208.0 |
| 3 | 2000 | 3/3 | 0.000 | 2.333 | 0.933 | 0.758 | 11.20 | 226.4 |
| 3 | 4000 | 3/3 | 0.333 | 1.333 | 0.956 | 0.800 | 20.48 | 257.5 |
| 3 | 8000 | 3/3 | 0.333 | 0.667 | 0.961 | 0.789 | 36.73 | 287.0 |

All 30 fits completed with finite parameters. All 30 also reached the
50-iteration limit; none satisfied the current convergence rule.

Approximate empirical scaling over the tested range:

- D=2: `1.84 seconds / 1000 rows` and `8.57 MiB / 1000 rows` above process
  baseline.
- D=3: `4.41 seconds / 1000 rows` and `11.81 MiB / 1000 rows` above process
  baseline.

The fitted observed conditional log-loss is close to the oracle conditional
log-loss and substantially better than a prevalence-only model. No probability
saturation was observed. Numerical overflow and coefficient-bound saturation
are not the leading failures in this experiment.

## Focused diagnostics

### Longer C=1 runs

At `N=500`, increasing the budget from 50 to 150 iterations still did not
trigger convergence.

- D=2 ended at `(6,2)`; last-seven mean relative changes were approximately
  `(0.142, 0.100)`.
- D=3 ended at `(16,4,2)`; last-seven mean relative changes were approximately
  `(0.130, 0.102, 0.109)`.

The current tolerances (`mean < 0.01`, `std < 0.005`) are far below the
persistent MC fluctuation actually observed.

### Increasing MC samples

Using `C=5` for 150 iterations made estimation materially worse rather than
stabilizing it.

- D=2 ended at `(6,3)`, graph accuracy `0.600`, adjusted latent accuracy
  `0.663`, conditional log-loss `0.5143`.
- D=3 progressively shrank its first layer from 21 to 8 and ended at
  `(8,7,2)`, graph accuracy `0.803`, adjusted latent accuracy `0.602`,
  conditional log-loss `0.5024`.

This behavior is consistent with a problem in the current update
`A_list = majority(MC draws)`: increasing `C` turns the persistent state update
into a more deterministic posterior-mode threshold rather than a Monte Carlo
state transition.

### Initialization control

With padded truth initialization at D=3, N=500, the fit retained the correct
`(18,6,2)` width through iteration 26 but removed a true first-layer factor at
iteration 27, ending at `(17,6,2)`. Stopping at iteration 20 retained the exact
width and produced better sensitivity, latent recovery, and conditional
log-loss than the 50-iteration result.

Therefore data initialization contributes to D=3 under-selection, but it is
not the sole cause: the CSP iteration can delete true dimensions even from a
truth-aligned starting point.

### Spike-scale sensitivity

For the same D=3, N=500 replicate, multiplying all `t_spike` values by 0.5 or
2.0 left the selected width at `(16,4,2)`. Graph and prediction metrics changed
only slightly. The observed under-selection is not resolved by a modest global
spike-scale adjustment.

## Development blockers identified

1. The current convergence criterion is ineffective under the observed MC
   variation, so runtime is controlled by `max_it` rather than convergence.
2. `C>1` changes the persistent-state dynamics through majority voting and can
   cause severe progressive dimension collapse.
3. D=3 selection is sensitive to initialization and iteration count; later
   iterations are not reliably better estimates.
4. A fixed `t_spike` gives mild over-selection for some large-N D=2 fits and
   systematic under-selection for small/medium-N D=3 fits.
5. Runtime and memory scale regularly and no numerical failures were found;
   statistical update dynamics are the current priority.

## Recommended next experiment

Before expanding to many replicates, compare the existing majority-state CSP
against two controlled alternatives while keeping the objective unchanged:

1. persist the last MC draw, matching fixed SAEM state semantics;
2. use one valid systematic Gibbs chain and retain its last draw.

Run the three state-transition policies on the same seeds at D=2/3,
`N=(500,2000,8000)`, `C=(1,5)`, and record width trajectories, conditional
log-loss, graph recovery, and relative-change trajectories. Only after this
comparison should convergence thresholds or sample-size-dependent CSP scales
be calibrated.
