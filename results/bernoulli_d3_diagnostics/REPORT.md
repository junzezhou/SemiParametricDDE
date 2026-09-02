# Bernoulli DDE D=3 diagnostic report

Date: 2026-09-02

This report diagnoses two observed failures of the current fixed-width
Bernoulli DDE implementation:

1. D=3 parameter RMSE does not decrease as `N` increases.
2. D=3 Bernoulli runtime is much larger than D=3 Normal runtime.

The production kernel was not changed during this diagnostic. The controlled
trajectory benchmark is implemented by
`scripts/diagnose_bernoulli_d3_saem.py`; raw results are in
`saem_trajectories.csv`, `exact_gibbs_stationarity.json`, and the companion
directory `../bernoulli_d3_diagnostics_n4000/`.

## Main conclusion

The non-decreasing RMSE has a primary algorithmic cause: the current latent
"parallel Gibbs" sweep samples every latent variable from full conditionals
computed against the same old latent state. This synchronous transition does
not preserve the joint posterior. Its bias is larger at depth 3. A systematic
coordinate Gibbs sweep does preserve the posterior and restores a decreasing
RMSE trend under a truth-start control.

Two secondary effects amplify the failure:

- convergence is tested on raw, unthresholded coefficients before family
  postprocessing, using an unnormalised full Frobenius norm; and
- the recursively constructed D=3 data initializer improves for the first
  layer with `N`, but its deeper-layer accuracy plateaus.

The runtime gap has a separate, well-localised cause. Every Bernoulli iteration
performs 54 observed-layer L-BFGS-B logistic optimisations, whereas the D=3
Normal observed-layer update is closed form. The unsuitable convergence metric
then forces many Bernoulli fits to run all 50 iterations.

## Code comparison

### Current Python kernel

- `src/dde/algorithms/semi_saem.py:418-510` explicitly computes a synchronous
  sweep: all full conditionals use the incoming `A_list`.
- `src/dde/algorithms/semi_saem.py:595-601` then samples all layers independently
  from those probabilities.
- `src/dde/algorithms/semi_saem.py:645-648` computes convergence error from raw
  M-step coefficients.
- `src/dde/algorithms/semi_saem.py:656-659` thresholds/postprocesses those
  coefficients only after the error has been computed.
- `src/dde/algorithms/parametric_families.py:625-663` solves one bounded
  L-BFGS-B problem per observed Bernoulli response.
- `src/dde/algorithms/parametric_families.py:404-431` uses a closed-form
  observed-layer update for Normal when `D >= 3`.

### Original MATLAB deeper-Normal kernel

The only deeper MATLAB kernel in the reference tree is
`Deep-Discrete-Encoders/simulations/2. deeper models/Auxiliary code/get_SAEM_normal_D.m`.
It also computes all latent probabilities from the old states (lines 53-95),
so it does not provide an independent correctness check for the synchronous
transition. However, its stopping behavior differs materially:

- it thresholds/debiases before calculating error;
- it measures changes only on the previous nonzero support;
- it divides the squared error by the total parameter count; and
- it allows only about five iterations (`t < 6`).

The current Python Bernoulli path therefore combines the same biased
synchronous transition with a much longer run and a stricter, raw-scale error.
The closed-form Normal M-step and short MATLAB run can conceal the issue, but do
not establish validity of the transition.

MATLAB's `binary_init.m` also sets the first truncated-SVD rank to at least
`2*K_parent`; the current Python Bernoulli initializer uses a smaller minimum.
A small control indicates that `2*K` can improve the top D=3 support estimate,
but it is not uniformly better for the middle layer and cannot fix posterior
sampling bias by itself.

## Exact transition check

For a toy model with one observed bit and one bit in each latent layer, all
posterior states and transition probabilities can be enumerated exactly.

| Depth | Transition | max error after one posterior step | stationary TV from posterior |
|---:|---|---:|---:|
| 2 | current synchronous | 0.071832 | 0.171162 |
| 2 | systematic coordinate | 2.78e-17 | 8.33e-17 |
| 3 | current synchronous | 0.081861 | 0.236600 |
| 3 | systematic coordinate | 5.55e-17 | 2.71e-16 |

The current transition is therefore not an invariant MCMC kernel for the
posterior. This is a correctness defect rather than Monte Carlo noise or an
insufficient number of replications.

## Controlled D=3 SAEM trajectories

The following values are mean parameter RMSE. `data` uses the current
Bernoulli initializer. `truth` starts from the true coefficients, proportions,
and latent states, isolating the transition from initialization. `N=500` and
`N=2000` use five replicates; `N=4000` uses three. Every run is held to 15
iterations and uses the same M-step implementation.

| N | start | transition | iter 1 | iter 5 | iter 15 |
|---:|---|---|---:|---:|---:|
| 500 | data | current synchronous | 0.5225 | 0.5420 | 0.5716 |
| 500 | data | systematic coordinate | 0.5112 | 0.4934 | 0.4860 |
| 500 | truth | current synchronous | 0.0942 | 0.1366 | 0.1832 |
| 500 | truth | systematic coordinate | 0.0702 | 0.0772 | 0.0808 |
| 2000 | data | current synchronous | 0.4441 | 0.4743 | 0.5065 |
| 2000 | data | systematic coordinate | 0.4187 | 0.3954 | 0.3793 |
| 2000 | truth | current synchronous | 0.0756 | 0.1456 | 0.2132 |
| 2000 | truth | systematic coordinate | 0.0463 | 0.0550 | 0.0596 |
| 4000 | data | current synchronous | 0.4484 | 0.4760 | 0.5150 |
| 4000 | data | systematic coordinate | 0.4301 | 0.4089 | 0.3939 |
| 4000 | truth | current synchronous | 0.0726 | 0.1533 | 0.2260 |
| 4000 | truth | systematic coordinate | 0.0380 | 0.0421 | 0.0448 |

The truth-start control is decisive: systematic coordinate Gibbs gives
iteration-15 RMSE `0.0808 -> 0.0596 -> 0.0448` as `N` grows, while the current
transition moves away from the truth at every sample size. Data-start runs also
improve with the systematic transition, but the remaining `N=2000` versus
`N=4000` plateau points to the deeper initializer/local basin as a secondary
problem.

## M-step isolation

When the M-step is given true latent states, 20-replicate recovery is:

| N | graph accuracy | parameter RMSE |
|---:|---:|---:|
| 500 | 0.9999 | 0.0665 |
| 2000 | 0.9989 | 0.0492 |
| 8000 | 1.0000 | 0.0280 |

Thus the Bernoulli likelihood, penalised observed-layer M-step, and evaluation
pipeline display the expected consistency when latent-state error is removed.
The first-order statistical defect is in the E-step transition, not the
observed-layer likelihood or M-step.

## Convergence criterion

At iteration 15 of `N=2000`, data-start, current-transition runs, the mean
errors are:

| error definition | mean error |
|---|---:|
| raw pre-threshold full norm | 1.2512 |
| post-threshold full norm | 0.1847 |
| old-support parameter-normalised norm | 0.00540 |

At `N=4000` the corresponding values are `1.5387`, `0.2805`, and `0.00819`.
Small coefficients that will immediately be removed by the family
postprocessor keep the raw error above the D=3 tolerance (`1.0`). In the full
100-replicate benchmark, the max-iteration rate increases from 3% at `N=1000`
to 70% at `N=2000` and 100% from `N=4000` onward. A postprocessed and
parameter-scaled criterion would generally stop these fits in roughly 3-5
iterations rather than 50. Correcting the stopping rule is necessary for
runtime and prevents repeated biased sweeps, but it is not a substitute for a
valid transition.

## Runtime attribution

With the same D=3 dimensions `(J, K) = (54, [18, 6, 2])` at `N=2000`:

| family | initialization | kernel | iterations |
|---|---:|---:|---:|
| Bernoulli | 0.019 s | 29.92 s | 50 |
| Normal | 0.010 s | 1.07 s | 50 |

A one-iteration profile at `N=2000` gives:

| component | Bernoulli | Normal |
|---|---:|---:|
| total | 0.249 s | 0.0345 s |
| all numerical minimizers | 0.188 s (78 fits) | 0.0284 s (24 fits) |
| observed-layer M-step | 0.168 s (54 fits) | approximately closed-form cost |
| E-step | 0.0536 s | 0.0043 s |

In the 15-iteration trajectory benchmark, replacing the transition with a
correct systematic coordinate sweep raised mean total iteration time at
`N=2000` only from `0.489` to `0.503` seconds (about 3%). The observed-layer
Bernoulli M-step remained about `0.40` seconds per iteration. Correctness can
therefore be fixed without creating another order-of-magnitude slowdown.

## Initialization diagnostic

Across 100 fresh initializations, the current data initializer behaves as
follows:

| N | initial RMSE | graph accuracy | latent accuracy |
|---:|---:|---:|---:|
| 500 | 0.6377 | 0.8336 | 0.7947 |
| 1000 | 0.5768 | 0.8804 | 0.7997 |
| 2000 | 0.5350 | 0.9218 | 0.8057 |
| 4000 | 0.5292 | 0.9433 | 0.8054 |
| 8000 | 0.5275 | 0.9501 | 0.8062 |
| 16000 | 0.5312 | 0.9518 | 0.8064 |

First-layer support improves with `N`, while the deeper support and latent
accuracy plateau. This should be addressed after the invariant transition and
stopping rule are corrected, otherwise initializer experiments are confounded
by subsequent sampler drift.

## Recommended implementation order

1. Replace the shared synchronous latent update with a systematic coordinate
   Gibbs sweep, vectorised across observations. Update both coordinates within
   a layer and layers within a sweep in a fixed order. Apply the correction to
   the shared fixed-family runner; the CSP path should be audited separately
   for the same transition pattern.
2. Compute convergence after family postprocessing and scale it by parameter
   count. Preserve a clearly specified support convention. Add exact posterior
   invariance and truth-start monotonic-recovery tests.
3. Re-run the full D=2/D=3 Bernoulli recovery benchmark. Only then tune the
   deeper initializer, beginning with the MATLAB `2*K` rank rule for the top
   small layer and multiple deterministic rotation/restart candidates.
4. Optimise runtime after correctness: replace 54 independent observed-layer
   L-BFGS-B fits with batched IRLS/Newton or another vectorised solver, and
   profile the stochastic-approximation accumulator as its number of observed
   parent patterns grows.

## Useful MATLAB comparison

There is no Bernoulli D=3 MATLAB implementation in the reference repository.
A full 100-replicate MATLAB run is therefore not the most useful immediate
comparison. The highest-value cross-check would be:

- D=2 Bernoulli at `N=2000` and `N=8000`, about 20 replicates, using the paper's
  positive coefficients and data initialization;
- for three fixed seeds, save the initialized and per-iteration `A`, `B`,
  `prop`, thresholded `B`, convergence error, optimizer exit flags, and
  objective values; and
- if a D=3 Bernoulli MATLAB version is created, run both the original
  synchronous transition and a systematic coordinate transition from the same
  simulated `X` and initialization.

That output would be useful for checking `fmincon` versus L-BFGS-B and exact
initialization parity. It is not needed to establish the synchronous-kernel
defect, which is already demonstrated by exact transition enumeration.

## Reproduction

```bash
python scripts/diagnose_bernoulli_d3_saem.py \
  --sample-sizes 500 2000 --replicates 5 --iterations 15 --n-jobs 6

python scripts/diagnose_bernoulli_d3_saem.py \
  --sample-sizes 4000 --replicates 3 --iterations 15 --n-jobs 6 \
  --output-dir results/bernoulli_d3_diagnostics_n4000
```

Because timings depend on machine load, the statistical comparisons use paired
seeds and should be interpreted primarily through direction and effect size.
