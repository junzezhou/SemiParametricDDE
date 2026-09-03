# Bernoulli DDE CSP paired comparison

## Design

- Scenario: D=3, J=64, true widths `(18, 6, 2)`, maximal fitted widths
  `(21, 7, 2)`.
- Sample sizes: 500, 1000, 2000, 4000, and 8000.
- Replicates: 30 per sample size and method.
- Both methods use the same data and initialization seed for every replicate.
- Baseline: simultaneous MCEM, immediate shrinkage, penalty strength 1.
- Candidate: 10 initialization/M-step warm-up iterations followed by simultaneous
  MCEM shrinkage; penalty strength `max(1, (N / 1000) ** (2 / 3))`.
- Both methods use C=1, 50 outer iterations, and 100 optimizer iterations.

## Dimension recovery

| N | Baseline mean K | Candidate mean K | Baseline exact | Candidate exact | Baseline L1 | Candidate L1 | Paired better / same / worse |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | (15.967, 4.933, 2.000) | (16.500, 5.233, 2.000) | 0.000 | 0.000 | 3.100 | 2.333 | 17 / 9 / 4 |
| 1000 | (16.867, 5.067, 2.000) | (17.667, 5.500, 2.000) | 0.067 | 0.267 | 2.067 | 1.033 | 23 / 6 / 1 |
| 2000 | (16.867, 5.133, 2.000) | (17.767, 5.200, 2.000) | 0.033 | 0.267 | 2.133 | 1.033 | 22 / 8 / 0 |
| 4000 | (16.933, 5.033, 2.000) | (17.767, 5.300, 1.967) | 0.067 | 0.167 | 2.033 | 1.167 | 20 / 8 / 2 |
| 8000 | (17.267, 5.533, 2.000) | (17.600, 5.533, 2.000) | 0.267 | 0.267 | 1.333 | 1.067 | 12 / 13 / 5 |
| Overall | — | — | 0.087 | 0.193 | 2.133 | 1.327 | 94 / 44 / 12 |

The candidate lowers under-selection from 0.907 to 0.760. Over-selection rises
from 0.027 to 0.067, but total dimension error falls substantially. For
N>=1000 the candidate L1 errors are 1.033, 1.033, 1.167, and 1.067, so there is
no systematic deterioration with increasing N. N=500 remains the main gap.

## Estimation guardrails

| Metric | Baseline | Candidate | Candidate minus baseline |
|---|---:|---:|---:|
| Graph accuracy | 0.93425 | 0.94479 | +0.01054 |
| Sensitivity | 0.86256 | 0.90051 | +0.03795 |
| Specificity | 0.94063 | 0.94873 | +0.00810 |
| Latent coverage-adjusted accuracy | 0.76399 | 0.77299 | +0.00899 |
| Conditional log-loss | 0.42478 | 0.42195 | -0.00283 |
| Brier score | 0.13759 | 0.13644 | -0.00115 |
| Mean runtime ratio | 1.000 | 1.105 | +10.5% |

All 300 fits completed with finite values. Every fit reached the 50-iteration
limit, so convergence and rare tail failures remain open issues.

## Comparison with the existing Normal CSP development benchmark

The Normal benchmark uses a different observed dimension and maximal-width
initialization `(J=108, K_max=(36,12,4))`, so this is descriptive rather than a
controlled family comparison.

| N | Normal mean K | Normal L1 | Bernoulli candidate mean K | Bernoulli candidate L1 |
|---:|---:|---:|---:|---:|
| 500 | (18.0, 6.0, 2.1) | 0.5 | (16.500, 5.233, 2.000) | 2.333 |
| 1000 | (18.0, 6.2, 2.4) | 1.0 | (17.667, 5.500, 2.000) | 1.033 |
| 2000 | (18.0, 6.3, 2.1) | 1.0 | (17.767, 5.200, 2.000) | 1.033 |
| 4000 | (18.4, 7.5, 1.9) | 2.8 | (17.767, 5.300, 1.967) | 1.167 |
| 8000 | (18.5, 8.3, 1.6) | 3.8 | (17.600, 5.533, 2.000) | 1.067 |

Bernoulli remains worse at N=500, is close at N=1000/2000, and has less mean
dimension error than this Normal benchmark at N=4000/8000.

## Decision log

Retained:

- Bernoulli D>=3 simultaneous MCEM: warm-up 10 and N-scaled penalty with power
  2/3.
- Deterministic Bernoulli spectral initialization whenever `random_state` is
  supplied.

Not retained:

- `mcem_cap_only`: propagated upstream under-selection into deeper caps.
- Linear N scaling: systematically over-shrank.
- Warm-up without adequate N scaling: over-selected at large N.
- N-scaling powers 1/2 and 1/4: no reliable improvement over the retained
  candidate in their target regimes.
- Warm-up 20: moved error from under-selection toward over-selection without a
  net dimension gain.
- Two-iteration inactive patience: worsened both endpoint pilots; its code was
  removed.
- D=2 adaptive scaling: exchanged over-selection for under-selection with no
  net L1 or exact-width gain, so D=2 keeps the legacy path.

## Verification

- Full test suite: 172 passed.
- `git diff --check`: passed.
- Python compile check: passed.
- Ruff was unavailable in the current environment.
