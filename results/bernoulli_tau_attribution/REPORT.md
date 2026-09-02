# Bernoulli DDE tau-attribution report

Date: 2026-09-02

## Design

The experiment uses a paired 2-by-2 design on the D=2 paper model and the
D=3 recursive extension, with `N = [2000, 8000, 16000]` and 20 replicates per
cell.  Within each scenario, sample size, and replicate, the simulated data,
initialization inputs, and fit seed are shared.

- fitting tau: `3*N**(-0.3)` versus fixed `0.3`;
- scoring tau: `3*N**(-0.3)` versus fixed `0.3`.

There are 240 fits and 480 scored cells.  All completed successfully, and no
fit reached the iteration limit.

## Main graph result

The three cells below are sufficient for additive attribution:

- dynamic/dynamic: the current paper-schedule pipeline;
- dynamic/fixed: the same dynamic-tau fit rescored at 0.3;
- fixed/fixed: fixed 0.3 in both fitting and scoring.

| D | N | dynamic/dynamic | dynamic/fixed | fixed/fixed | total gain |
|---:|---:|---:|---:|---:|---:|
| 2 | 2,000 | 0.9550 | 0.9550 | 0.9517 | -0.0033 |
| 2 | 8,000 | 0.9250 | 0.9596 | 0.9713 | +0.0463 |
| 2 | 16,000 | 0.8888 | 0.9658 | 0.9775 | +0.0888 |
| 3 | 2,000 | 0.9228 | 0.9228 | 0.9207 | -0.0021 |
| 3 | 8,000 | 0.8942 | 0.9152 | 0.9379 | +0.0437 |
| 3 | 16,000 | 0.8745 | 0.9182 | 0.9396 | +0.0651 |

At N=2000, dynamic tau is 0.3068, so the near-zero difference is the intended
negative control.  At larger N, both parts of the tau change matter:

| D | N | scoring-only gain | fitting-path gain | scoring share of total |
|---:|---:|---:|---:|---:|
| 2 | 8,000 | +0.0346 | +0.0117 | 74.8% |
| 2 | 16,000 | +0.0771 | +0.0117 | 86.9% |
| 3 | 8,000 | +0.0210 | +0.0227 | 48.1% |
| 3 | 16,000 | +0.0437 | +0.0214 | 67.2% |

## False-positive attribution

Fixed tau primarily reduces false positives, with a small sensitivity cost.

| D | N | FP dynamic/dynamic | FP dynamic/fixed | FP fixed/fixed | FN dynamic/dynamic | FN fixed/fixed |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 8,000 | 9.00 | 4.85 | 3.25 | 0.00 | 0.20 |
| 2 | 16,000 | 13.35 | 4.10 | 2.50 | 0.00 | 0.20 |
| 3 | 8,000 | 113.15 | 90.15 | 64.40 | 2.35 | 3.40 |
| 3 | 16,000 | 135.65 | 87.70 | 62.65 | 1.40 | 3.30 |

For D=3 at N=16000, rescoring at 0.3 removes 47.95 false edges in the interval
between dynamic tau (0.1644) and 0.3.  Holding the scoring threshold at 0.3,
fitting at fixed 0.3 removes a further 25.05 false edges.  Thus the problem is
not only a reporting-threshold artifact; tau changes the fitted path.

## Layer attribution at D=3, N=16000

| layer | graph dynamic/dynamic | graph dynamic/fixed | graph fixed/fixed | FP dynamic/dynamic | FP fixed/fixed |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.9085 | 0.9508 | 0.9688 | 88.65 | 29.45 |
| 2 | 0.5856 | 0.6458 | 0.7019 | 44.10 | 30.75 |
| 3 | 0.7208 | 0.7292 | 0.7125 | 2.90 | 2.45 |

Most of the total improvement comes from the observed layer, but the middle
layer also improves materially.  Its final accuracy of 0.7019 and 30.75 false
positives show that fixed tau does not fully resolve the deeper-layer problem.

## RMSE and iterations

Using paired per-replicate RMSE differences, fixed/fixed versus
dynamic/dynamic gives:

| D | N | mean RMSE change | paired SE |
|---:|---:|---:|---:|
| 2 | 8,000 | -0.0121 | 0.0015 |
| 2 | 16,000 | -0.0160 | 0.0022 |
| 3 | 8,000 | -0.0082 | 0.0010 |
| 3 | 16,000 | -0.0080 | 0.0013 |

Iteration changes are small and inconsistent: +0.55 and +0.10 for D=2, and
-0.30 and -0.60 for D=3 at N=8000 and 16000.  The tau benefit is therefore not
mediated by simply running more iterations.  Mean latent accuracy is also
essentially unchanged, localizing the main effect to coefficient fitting and
support selection.

## Conclusion

The declining paper tau is the primary cause of the large-sample graph
degradation in the current Bernoulli pipeline.  At D=2, most of the effect is
the final graph threshold.  At D=3, both scoring and the fitting path are
important.  Fixed tau=0.3 substantially improves graph accuracy and RMSE, but
the D=3 middle layer remains overselected and should be investigated through
initialization and fixed-iteration trajectory controls.

Raw cells are in `replicates.csv`, aggregate cells in `summary.csv`, and paired
effects with standard errors in `paired_differences.csv`.  The experiment is
implemented by `scripts/diagnose_bernoulli_tau_attribution.py`.
